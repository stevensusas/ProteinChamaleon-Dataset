"""OpenAI Batch API client for bulk protein description generation.

Batch jobs are asynchronous — submit now, fetch results later (usually < 24 h).
Results are billed at 50% of the standard per-token rate.

Usage:
  export $(grep -v '^#' .env | xargs)

  # Submit a batch and get a batch ID
  python -m llm.batch --accessions P04637 P06400 Q09472 --submit

  # Check status of a running batch
  python -m llm.batch --status batch_abc123def456

  # Download and save results once the batch is complete
  python -m llm.batch --fetch batch_abc123def456
"""

from __future__ import annotations

import argparse
from datetime import datetime
import io
import json
import os
from pathlib import Path
import random
from typing import Any

import httpx

from graph.neo4j_client import Neo4jClient
from llm import LLMConfig
from llm.client import ProteinDescriptionLLMClient
from llm.prompts import SYSTEM_PROMPT, build_multi_protein_prompt


# ---------------------------------------------------------------------------
# Core client
# ---------------------------------------------------------------------------

class ProteinBatchClient:
    """
    Submit and retrieve OpenAI Batch API jobs for protein descriptions.

    Only works with the OpenAI backend (base_url must be api.openai.com).
    Groq does not support the Batch API.
    """

    _OPENAI_BASE = "https://api.openai.com/v1"

    def __init__(self, neo4j: Neo4jClient, config: LLMConfig) -> None:
        self.neo4j = neo4j
        self.config = config
        # Reuse the single-protein client for context-building helpers.
        self._llm = ProteinDescriptionLLMClient(neo4j=neo4j, config=config)
        self._http = httpx.Client(
            timeout=120.0,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )

    def close(self) -> None:
        self._llm.close()
        self._http.close()

    def __enter__(self) -> "ProteinBatchClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    @staticmethod
    def _url_manifest(protein: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Build a structure manifest using public download URLs instead of local paths.
        The model will emit these URLs as inline placeholders; callers can download
        them later on demand.
        """
        manifest: list[dict[str, Any]] = []

        for pdb_id in protein.get("pdb_ids", []):
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            manifest.append({
                "placeholder": f"[{url}]",
                "source": "pdb",
                "pdb_id": pdb_id,
                "url": url,
            })

        for af_id in protein.get("alphafold_entry_ids", []):
            # Standard AlphaFold DB URL pattern
            url = f"https://alphafold.ebi.ac.uk/files/{af_id}-model_v4.pdb"
            manifest.append({
                "placeholder": f"[{url}]",
                "source": "alphafold",
                "entry_id": af_id,
                "url": url,
            })

        return manifest

    # ------------------------------------------------------------------
    # Per-group request builders
    # ------------------------------------------------------------------

    def _make_request(self, custom_id: str, messages: list[dict]) -> dict:
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
        }

    def _build_single_line(self, acc: str, context: dict) -> dict:
        protein = context["protein"]
        manifest = self._url_manifest(protein)
        messages = self._llm._build_messages(acc, context, structure_files=manifest)
        return self._make_request(acc, messages)

    def _build_multi_line(self, group: list[str], contexts: dict) -> dict:
        # Fetch interaction edges between proteins in this group.
        with self.neo4j.driver.session() as s:
            rows = s.run(
                "MATCH (a:Protein)-[r:INTERACTS_WITH]-(b:Protein) "
                "WHERE a.accession IN $accs AND b.accession IN $accs "
                "RETURN a.accession AS source, b.accession AS target, properties(r) AS props",
                accs=group,
            )
            edges = [{"source": r["source"], "target": r["target"], "properties": r["props"]}
                     for r in rows]

        # Combined URL manifest across all proteins in the group.
        manifest: list[dict] = []
        for acc in group:
            manifest.extend(self._url_manifest(contexts[acc]["protein"]))

        packed = self._llm._context_for_network_prompt(contexts)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_multi_protein_prompt(
                accessions=group,
                contexts=packed,
                interaction_edges=edges,
                structure_manifest=manifest,
            )},
        ]
        custom_id = "multi_" + "_".join(group)
        return self._make_request(custom_id, messages)

    # ------------------------------------------------------------------
    # JSONL builders
    # ------------------------------------------------------------------

    def build_jsonl(self, accessions: list[str]) -> tuple[bytes, list[str]]:
        """Build a single-protein JSONL batch payload for a flat list of accessions."""
        lines, skipped = [], []
        for acc in accessions:
            context = self.neo4j.get_protein_context(acc)
            if not context.get("protein"):
                print(f"  [skip] {acc} — not found in graph")
                skipped.append(acc)
                continue
            lines.append(json.dumps(self._build_single_line(acc, context)))
        return "\n".join(lines).encode(), skipped

    def build_mixed_jsonl(
        self, groups: list[list[str]]
    ) -> tuple[bytes, list[list[str]]]:
        """
        Build a JSONL payload from a mixed list of groups (size 1–4).
        Single-protein groups use the single-protein prompt;
        multi-protein groups use the network prompt.

        Returns:
            (jsonl_bytes, skipped_groups)
        """
        lines, skipped = [], []
        for group in groups:
            # Fetch contexts for every accession in the group.
            contexts: dict[str, dict] = {}
            for acc in group:
                ctx = self.neo4j.get_protein_context(acc)
                if ctx.get("protein"):
                    contexts[acc] = ctx
            if len(contexts) < len(group):
                missing = [a for a in group if a not in contexts]
                print(f"  [skip group] missing in graph: {missing}")
                skipped.append(group)
                continue

            if len(group) == 1:
                acc = group[0]
                lines.append(json.dumps(self._build_single_line(acc, contexts[acc])))
            else:
                lines.append(json.dumps(self._build_multi_line(group, contexts)))

        return "\n".join(lines).encode(), skipped

    # ------------------------------------------------------------------
    # Interaction-aware disjoint sampling
    # ------------------------------------------------------------------

    def sample_disjoint_groups(
        self, per_size: int = 50, seed: int | None = None
    ) -> list[list[str]]:
        """
        Sample disjoint protein groups of sizes 1, 2, 3, and 4 from the graph.

        For groups of size > 1, the seed protein is chosen randomly and partners
        are its highest-scoring interactors not yet assigned to any group.

        Returns a flat list of groups (each a list of accessions), shuffled.
        """
        rng = random.Random(seed)

        # Fetch all proteins and their interaction partners ordered by score.
        with self.neo4j.driver.session() as s:
            rows = s.run(
                """
                MATCH (p:Protein)
                OPTIONAL MATCH (p)-[r:INTERACTS_WITH]-(q:Protein)
                WITH p.accession AS acc,
                     collect({partner: q.accession, score: coalesce(r.combined_score, 0)})
                         AS neighbors
                RETURN acc, neighbors
                """
            )
            graph: dict[str, list[str]] = {}
            for row in rows:
                acc = row["acc"]
                # Sort neighbors by score descending, drop nulls.
                neighbors = sorted(
                    [n for n in row["neighbors"] if n["partner"]],
                    key=lambda n: n["score"],
                    reverse=True,
                )
                graph[acc] = [n["partner"] for n in neighbors]

        all_proteins = list(graph.keys())
        rng.shuffle(all_proteins)

        used: set[str] = set()
        groups: list[list[str]] = []

        def pick_group(size: int) -> list[str] | None:
            for seed_acc in all_proteins:
                if seed_acc in used:
                    continue
                partners = [p for p in graph[seed_acc] if p not in used]
                if len(partners) < size - 1:
                    continue
                group = [seed_acc] + partners[: size - 1]
                used.update(group)
                return group
            return None

        for size in (1, 2, 3, 4):
            collected = 0
            for _ in range(per_size):
                group = pick_group(size)
                if group is None:
                    print(f"  [warn] only collected {collected}/{per_size} groups of size {size}")
                    break
                groups.append(group)
                collected += 1

        rng.shuffle(groups)
        return groups

    # ------------------------------------------------------------------
    # Submit (single-protein or mixed)
    # ------------------------------------------------------------------

    def submit(
        self,
        accessions: list[str] | None = None,
        groups: list[list[str]] | None = None,
        output_dir: Path | None = None,
    ) -> str:
        """
        Build, upload, and submit a batch job.

        Pass either:
          accessions — flat list for single-protein-only batch
          groups     — mixed list of groups (size 1-4) for a mixed batch

        Returns the batch ID.
        """
        output_dir = output_dir or (
            Path(__file__).resolve().parent.parent
            / "output"
            / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        if groups is not None:
            print(f"Building mixed JSONL for {len(groups)} group(s)…")
            jsonl_bytes, skipped = self.build_mixed_jsonl(groups)
            n_included = len(groups) - len(skipped)
            group_meta = groups
        elif accessions is not None:
            print(f"Building JSONL for {len(accessions)} accession(s)…")
            jsonl_bytes, skipped_accs = self.build_jsonl(accessions)
            n_included = len(accessions) - len(skipped_accs)
            skipped = [[a] for a in skipped_accs]
            group_meta = [[a] for a in accessions]
        else:
            raise ValueError("Provide either accessions or groups.")

        if n_included == 0:
            raise ValueError("No valid entries found — nothing to submit.")

        jsonl_path = output_dir / "batch_input.jsonl"
        jsonl_path.write_bytes(jsonl_bytes)
        print(f"Wrote {n_included} request(s) to {jsonl_path}")

        print("Uploading to OpenAI Files API…")
        upload_resp = self._http.post(
            f"{self._OPENAI_BASE}/files",
            files={"file": ("batch_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl")},
            data={"purpose": "batch"},
        )
        upload_resp.raise_for_status()
        file_id = upload_resp.json()["id"]
        print(f"File uploaded: {file_id}")

        print("Creating batch job…")
        batch_resp = self._http.post(
            f"{self._OPENAI_BASE}/batches",
            headers={"Content-Type": "application/json"},
            content=json.dumps({
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            }),
        )
        batch_resp.raise_for_status()
        batch = batch_resp.json()
        batch_id = batch["id"]

        meta = {
            "batch_id": batch_id,
            "file_id": file_id,
            "groups": group_meta,
            "skipped": skipped,
            "model": self.config.model,
            "submitted_at": datetime.utcnow().isoformat(),
            "output_dir": str(output_dir),
        }
        (output_dir / "batch_meta.json").write_text(json.dumps(meta, indent=2))

        print(f"\nBatch submitted: {batch_id}")
        print(f"Status: {batch.get('status')}")
        print(f"Metadata saved to {output_dir / 'batch_meta.json'}")
        print(f"\nTo check status:\n  python -m llm.batch --status {batch_id}")
        print(f"To fetch results when complete:\n  python -m llm.batch --fetch {batch_id}")

        return batch_id

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self, batch_id: str) -> dict[str, Any]:
        """Return the current status dict for a batch job."""
        resp = self._http.get(f"{self._OPENAI_BASE}/batches/{batch_id}")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Fetch results
    # ------------------------------------------------------------------

    def fetch(
        self,
        batch_id: str,
        output_dir: Path | None = None,
    ) -> dict[str, str]:
        """
        Download batch results once the job is complete.

        Returns a mapping of accession → description text.
        Results are also saved as individual <accession>_description.txt files.
        """
        batch = self.status(batch_id)
        state = batch.get("status")

        if state != "completed":
            raise RuntimeError(
                f"Batch {batch_id} is not complete yet (status: {state}). "
                "Try again later."
            )

        output_file_id = batch.get("output_file_id")
        if not output_file_id:
            raise RuntimeError(f"Batch {batch_id} has no output_file_id.")

        output_dir = output_dir or (
            Path(__file__).resolve().parent.parent / "output" / f"batch_{batch_id}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download the output JSONL.
        print(f"Downloading results from {output_file_id}…")
        dl_resp = self._http.get(f"{self._OPENAI_BASE}/files/{output_file_id}/content")
        dl_resp.raise_for_status()
        raw_jsonl = dl_resp.text

        output_jsonl_path = output_dir / "batch_output.jsonl"
        output_jsonl_path.write_text(raw_jsonl, encoding="utf-8")

        # Parse results.
        results: dict[str, str] = {}
        errors: list[dict[str, Any]] = []

        for line in raw_jsonl.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            custom_id = record.get("custom_id", "unknown")
            error = record.get("error")
            if error:
                errors.append({"accession": custom_id, "error": error})
                continue
            response_body = record.get("response", {}).get("body", {})
            choices = response_body.get("choices", [])
            if not choices:
                errors.append({"accession": custom_id, "error": "no choices in response"})
                continue
            text = choices[0].get("message", {}).get("content", "").strip()
            results[custom_id] = text

            # Save individual description files.
            desc_path = output_dir / f"{custom_id}_description.txt"
            desc_path.write_text(text, encoding="utf-8")

        # Save summary.
        summary = {
            "batch_id": batch_id,
            "fetched_at": datetime.utcnow().isoformat(),
            "total_requests": batch.get("request_counts", {}).get("total"),
            "succeeded": len(results),
            "failed": len(errors),
            "errors": errors,
        }
        (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2))

        print(f"\nResults saved to {output_dir}")
        print(f"  Succeeded: {len(results)}")
        if errors:
            print(f"  Failed:    {len(errors)}")
            for e in errors:
                print(f"    {e['accession']}: {e['error']}")
        for acc, text in results.items():
            print(f"\n{'=' * 60}")
            print(f"  {acc}")
            print("=" * 60)
            print(text)

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit or retrieve an OpenAI Batch API job for protein descriptions."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--submit",
        action="store_true",
        help="Submit a batch for explicit --accessions (single-protein only).",
    )
    group.add_argument(
        "--sample",
        action="store_true",
        help="Auto-sample disjoint groups of size 1–4 from the graph and submit.",
    )
    group.add_argument(
        "--status",
        metavar="BATCH_ID",
        help="Print the current status of an existing batch job.",
    )
    group.add_argument(
        "--fetch",
        metavar="BATCH_ID",
        help="Download and save results of a completed batch job.",
    )
    parser.add_argument(
        "--accessions",
        nargs="*",
        default=[],
        help="UniProt accessions (used with --submit).",
    )
    parser.add_argument(
        "--per-size",
        type=int,
        default=50,
        help="Number of groups per size class when using --sample (default: 50).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files (auto-generated if omitted).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    neo4j_uri  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASS", "password")
    config = LLMConfig.from_env()

    base_url = config.base_url.rstrip("/")
    if "openai.com" not in base_url:
        raise RuntimeError(
            f"Batch API is only supported with OpenAI (LLM_BASE_URL={base_url}). "
            "Set LLM_BASE_URL=https://api.openai.com/v1."
        )

    output_dir = Path(args.output_dir) if args.output_dir else None

    with Neo4jClient(uri=neo4j_uri, username=neo4j_user, password=neo4j_pass) as neo4j:
        with ProteinBatchClient(neo4j=neo4j, config=config) as client:

            if args.submit:
                if not args.accessions:
                    raise ValueError("--submit requires --accessions.")
                client.submit(accessions=args.accessions, output_dir=output_dir)

            elif args.sample:
                per_size = args.per_size
                print(f"Sampling {per_size} groups × 4 sizes = up to {per_size * 4} requests…")
                groups = client.sample_disjoint_groups(per_size=per_size)
                sizes = {1: 0, 2: 0, 3: 0, 4: 0}
                for g in groups:
                    sizes[len(g)] = sizes.get(len(g), 0) + 1
                print(f"  size-1: {sizes[1]}  size-2: {sizes[2]}  "
                      f"size-3: {sizes[3]}  size-4: {sizes[4]}")
                client.submit(groups=groups, output_dir=output_dir)

            elif args.status:
                batch = client.status(args.status)
                counts = batch.get("request_counts", {})
                print(f"Batch ID : {batch['id']}")
                print(f"Status   : {batch.get('status')}")
                print(f"Requests : total={counts.get('total')}  "
                      f"completed={counts.get('completed')}  "
                      f"failed={counts.get('failed')}")
                created = batch.get("created_at")
                if created:
                    print(f"Created  : {datetime.utcfromtimestamp(created).isoformat()} UTC")

            elif args.fetch:
                client.fetch(args.fetch, output_dir=output_dir)


if __name__ == "__main__":
    main()
