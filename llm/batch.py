"""OpenAI Batch API pipeline for bulk protein description generation.

Pipeline (--sample or --submit):
  1. Sample/resolve protein groups from Neo4j
  2. Download PDB + AlphaFold structures + feature slices per protein
  3. Build JSONL with local-path manifests
  4. Upload to OpenAI Files API and submit batch

Retrieve results (--fetch):
  5. Download output JSONL, save per-entry .txt files into results/

Usage:
  export $(grep -v '^#' .env | xargs)

  # Auto-sample 50 groups of each size and submit
  python -m llm.batch --sample

  # Explicit accessions (single-protein only)
  python -m llm.batch --submit --accessions P04637 P06400

  # Check status
  python -m llm.batch --status batch_abc123

  # Fetch results into the original run directory
  python -m llm.batch --fetch batch_abc123
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
from llm.single_protein import prepare_structure_files


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
        self._llm = ProteinDescriptionLLMClient(neo4j=neo4j, config=config)
        self._http = httpx.Client(
            timeout=120.0,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
        )

    def close(self) -> None:
        self._llm.close()
        self._http.close()

    def __enter__(self) -> "ProteinBatchClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Structure downloading
    # ------------------------------------------------------------------

    def _download_structures(
        self,
        contexts: dict[str, dict[str, Any]],
        structure_dir: Path,
        max_structures: int,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Download PDB/AlphaFold structures + feature slices for every accession.

        Returns a mapping of accession → structure manifest list.
        Each manifest entry has a 'placeholder' and 'path' field usable in prompts.
        """
        manifests: dict[str, list[dict[str, Any]]] = {}
        for acc, ctx in contexts.items():
            print(f"  Downloading structures for {acc}…")
            manifests[acc] = prepare_structure_files(
                accession=acc,
                protein=ctx["protein"],
                feature_rels=ctx.get("feature_relationships", []),
                structure_dir=structure_dir,
                max_structures=max_structures,
                output_subdir=acc,
            )
            n = len(manifests[acc])
            print(f"    → {n} structure file(s)")
        return manifests

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

    def _build_single_line(
        self,
        acc: str,
        context: dict,
        manifest: list[dict],
    ) -> dict:
        messages = self._llm._build_messages(acc, context, structure_files=manifest)
        return self._make_request(acc, messages)

    def _build_multi_line(
        self,
        group: list[str],
        contexts: dict,
        manifest: list[dict],
    ) -> dict:
        with self.neo4j.driver.session() as s:
            rows = s.run(
                "MATCH (a:Protein)-[r:INTERACTS_WITH]-(b:Protein) "
                "WHERE a.accession IN $accs AND b.accession IN $accs "
                "RETURN a.accession AS source, b.accession AS target, properties(r) AS props",
                accs=group,
            )
            edges = [
                {"source": r["source"], "target": r["target"], "properties": r["props"]}
                for r in rows
            ]

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
    # Interaction-aware disjoint sampling
    # ------------------------------------------------------------------

    def sample_disjoint_groups(
        self,
        per_size: int = 50,
        min_features: int = 5,
        require_structure: bool = True,
        seed: int | None = None,
    ) -> list[list[str]]:
        """
        Sample disjoint protein groups of sizes 1, 2, 3, and 4 from the graph.

        Only proteins with at least `min_features` HAS_FEATURE annotations and
        (optionally) at least one PDB or AlphaFold entry are eligible.
        Multi-protein groups contain proteins that actually interact.
        """
        rng = random.Random(seed)

        structure_filter = (
            "AND (size(p.pdb_ids) > 0 OR size(p.alphafold_entry_ids) > 0)"
            if require_structure else ""
        )
        partner_structure_filter = structure_filter.replace("p.", "q.")

        with self.neo4j.driver.session() as s:
            rows = s.run(
                f"""
                MATCH (p:Protein)
                WHERE size([(p)-[:HAS_FEATURE]->() | 1]) >= $min_features
                {structure_filter}
                OPTIONAL MATCH (p)-[r:INTERACTS_WITH]-(q:Protein)
                WHERE size([(q)-[:HAS_FEATURE]->() | 1]) >= $min_features
                {partner_structure_filter}
                WITH p.accession AS acc,
                     collect({{partner: q.accession, score: coalesce(r.combined_score, 0)}})
                         AS neighbors
                RETURN acc, neighbors
                """,
                min_features=min_features,
            )
            graph: dict[str, list[str]] = {}
            for row in rows:
                acc = row["acc"]
                neighbors = sorted(
                    [n for n in row["neighbors"] if n["partner"]],
                    key=lambda n: n["score"],
                    reverse=True,
                )
                graph[acc] = [n["partner"] for n in neighbors]

        eligible = list(graph.keys())
        print(f"  Eligible proteins: {len(eligible)} "
              f"(min_features={min_features}, require_structure={require_structure})")

        if not eligible:
            return []

        rng.shuffle(eligible)
        groups: list[list[str]] = []

        for size in (1, 2, 3, 4):
            candidates = [p for p in eligible if len(graph[p]) >= size - 1]
            rng.shuffle(candidates)
            collected = 0
            for seed in candidates[:per_size]:
                partners = graph[seed][: size - 1]
                groups.append([seed] + partners)
                collected += 1
            if collected < per_size:
                print(f"  [warn] only collected {collected}/{per_size} groups of size {size}")

        rng.shuffle(groups)
        return groups

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit(
        self,
        groups: list[list[str]],
        run_dir: Path | None = None,
        max_structures: int = 3,
        structures_dir: Path | None = None,
    ) -> str:
        """
        Full pipeline: download structures, build JSONL, upload, submit.

        Args:
            groups:         List of protein groups (size 1–4). Single-protein
                            groups use the single-protein prompt; larger groups
                            use the network prompt.
            run_dir:        Root directory for this run. Auto-generated if None.
            max_structures: Max PDB/AlphaFold files downloaded per protein.

        Returns the batch ID.
        """
        run_dir = run_dir or (
            Path(__file__).resolve().parent.parent
            / "output" / "batched"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        structure_dir = structures_dir or (run_dir / "structures")
        structure_dir.mkdir(parents=True, exist_ok=True)
        if structures_dir:
            print(f"Reusing existing structures from {structures_dir}")

        # ── Step 1: fetch Neo4j contexts for all unique accessions ────────────
        all_accs: list[str] = list({acc for g in groups for acc in g})
        print(f"\nFetching contexts for {len(all_accs)} unique protein(s)…")
        contexts: dict[str, dict[str, Any]] = {}
        for acc in all_accs:
            ctx = self.neo4j.get_protein_context(acc)
            if ctx.get("protein"):
                contexts[acc] = ctx
            else:
                print(f"  [skip] {acc} — not found in graph")

        # ── Step 2: download structures ───────────────────────────────────────
        print(f"\nDownloading structures into {structure_dir}…")
        manifests = self._download_structures(contexts, structure_dir, max_structures)

        # ── Step 3: build JSONL ───────────────────────────────────────────────
        print("\nBuilding JSONL…")
        lines: list[str] = []
        skipped: list[list[str]] = []
        group_index: list[dict] = []  # saved to metadata

        for group in groups:
            valid = [a for a in group if a in contexts]
            if len(valid) < len(group):
                missing = [a for a in group if a not in contexts]
                print(f"  [skip group] missing: {missing}")
                skipped.append(group)
                continue

            group_manifest = [entry for acc in valid for entry in manifests.get(acc, [])]

            if len(valid) == 1:
                acc = valid[0]
                req = self._build_single_line(acc, contexts[acc], group_manifest)
            else:
                req = self._build_multi_line(valid, {a: contexts[a] for a in valid}, group_manifest)

            lines.append(json.dumps(req))
            group_index.append({"custom_id": req["custom_id"], "group": valid, "size": len(valid)})

        if not lines:
            raise ValueError("No valid groups — nothing to submit.")

        # Deduplicate custom_ids (same protein/combo can appear in multiple groups).
        seen: dict[str, int] = {}
        deduped: list[str] = []
        for line in lines:
            req = json.loads(line)
            cid = req["custom_id"]
            if cid in seen:
                seen[cid] += 1
                req["custom_id"] = f"{cid}__{seen[cid]}"
            else:
                seen[cid] = 0
            deduped.append(json.dumps(req))
        lines = deduped

        # Update group_index custom_ids to match the deduped ones.
        for i, line in enumerate(lines):
            group_index[i]["custom_id"] = json.loads(line)["custom_id"]

        jsonl_bytes = "\n".join(lines).encode()
        jsonl_path = run_dir / "batch_input.jsonl"
        jsonl_path.write_bytes(jsonl_bytes)
        print(f"Wrote {len(lines)} request(s) to {jsonl_path}")

        # ── Step 4: upload and submit ─────────────────────────────────────────
        print("\nUploading to OpenAI Files API…")
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
            "run_dir": str(run_dir),
            "model": self.config.model,
            "submitted_at": datetime.utcnow().isoformat(),
            "groups": group_index,
            "skipped": skipped,
        }
        (run_dir / "batch_meta.json").write_text(json.dumps(meta, indent=2))

        print(f"\nBatch submitted: {batch_id}")
        print(f"Status: {batch.get('status')}")
        print(f"Run directory: {run_dir}")
        print(f"\nTo check status:\n  python -m llm.batch --status {batch_id}")
        print(f"To fetch results:\n  python -m llm.batch --fetch {batch_id}")

        return batch_id

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self, batch_id: str) -> dict[str, Any]:
        resp = self._http.get(f"{self._OPENAI_BASE}/batches/{batch_id}")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Fetch results
    # ------------------------------------------------------------------

    def fetch(
        self,
        batch_id: str,
        run_dir: Path | None = None,
    ) -> dict[str, str]:
        """
        Download and save batch results.

        If run_dir is not provided, looks for batch_meta.json in
        output/batched/*/ to find the original run directory.
        Results go into run_dir/results/{single,2_protein,3_protein,4_protein}/.
        """
        batch = self.status(batch_id)
        state = batch.get("status")
        if state != "completed":
            raise RuntimeError(
                f"Batch {batch_id} is not complete yet (status: {state}). Try again later."
            )

        output_file_id = batch.get("output_file_id")
        if not output_file_id:
            raise RuntimeError(f"Batch {batch_id} has no output_file_id.")

        # Try to find original run_dir from saved metadata.
        if run_dir is None:
            batched_root = Path(__file__).resolve().parent.parent / "output" / "batched"
            for meta_file in sorted(batched_root.glob("*/batch_meta.json"), reverse=True):
                meta = json.loads(meta_file.read_text())
                if meta.get("batch_id") == batch_id:
                    run_dir = Path(meta["run_dir"])
                    print(f"Found run directory: {run_dir}")
                    break

        if run_dir is None:
            run_dir = (
                Path(__file__).resolve().parent.parent / "output" / "batched" / f"fetch_{batch_id}"
            )

        subdirs = {
            1: run_dir / "results" / "single",
            2: run_dir / "results" / "2_protein",
            3: run_dir / "results" / "3_protein",
            4: run_dir / "results" / "4_protein",
        }
        for d in subdirs.values():
            d.mkdir(parents=True, exist_ok=True)

        print(f"Downloading results from {output_file_id}…")
        dl_resp = self._http.get(f"{self._OPENAI_BASE}/files/{output_file_id}/content")
        dl_resp.raise_for_status()
        raw_jsonl = dl_resp.text
        (run_dir / "batch_output.jsonl").write_text(raw_jsonl, encoding="utf-8")

        results: dict[str, str] = {}
        errors: list[dict[str, Any]] = []

        for line in raw_jsonl.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            custom_id = record.get("custom_id", "unknown")
            error = record.get("error")
            if error:
                errors.append({"id": custom_id, "error": error})
                continue
            choices = record.get("response", {}).get("body", {}).get("choices", [])
            if not choices:
                errors.append({"id": custom_id, "error": "no choices in response"})
                continue
            text = choices[0].get("message", {}).get("content", "").strip()
            results[custom_id] = text

            base_id = custom_id.split("__")[0]  # strip dedup suffix
            size = len(base_id[len("multi_"):].split("_")) if base_id.startswith("multi_") else 1
            size = max(1, min(4, size))
            (subdirs[size] / f"{custom_id}.txt").write_text(text, encoding="utf-8")

        counts = {k: 0 for k in subdirs}
        for cid in results:
            size = len(cid[len("multi_"):].split("_")) if cid.startswith("multi_") else 1
            counts[max(1, min(4, size))] += 1

        summary = {
            "batch_id": batch_id,
            "fetched_at": datetime.utcnow().isoformat(),
            "total_requests": batch.get("request_counts", {}).get("total"),
            "succeeded": len(results),
            "failed": len(errors),
            "errors": errors,
        }
        (run_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2))

        print(f"\nResults saved to {run_dir}/results/")
        for size, label in [(1, "single"), (2, "2_protein"), (3, "3_protein"), (4, "4_protein")]:
            print(f"  {label}/  — {counts[size]} file(s)")
        if errors:
            print(f"\n  Failed: {len(errors)}")
            for e in errors:
                print(f"    {e['id']}: {e['error']}")

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI Batch API pipeline for protein descriptions."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--sample",
        action="store_true",
        help="Auto-sample disjoint groups (sizes 1–4) from the graph, download structures, and submit.",
    )
    mode.add_argument(
        "--submit",
        action="store_true",
        help="Submit explicit --accessions as single-protein entries.",
    )
    mode.add_argument(
        "--status",
        metavar="BATCH_ID",
        help="Print current status of a batch job.",
    )
    mode.add_argument(
        "--fetch",
        metavar="BATCH_ID",
        help="Download and save results of a completed batch job.",
    )
    parser.add_argument("--accessions", nargs="*", default=[])
    parser.add_argument("--per-size", type=int, default=50,
                        help="Groups per size class for --sample (default: 50).")
    parser.add_argument("--min-features", type=int, default=5,
                        help="Min HAS_FEATURE annotations to be eligible (default: 5).")
    parser.add_argument("--no-structure-filter", action="store_true",
                        help="Allow proteins with no PDB/AlphaFold entries.")
    parser.add_argument("--max-structures", type=int, default=3,
                        help="Max structure files downloaded per protein (default: 3).")
    parser.add_argument("--structures-dir", default=None,
                        help="Reuse an existing structures directory (e.g. output/batched/20260406_185158/structures).")
    parser.add_argument("--run-dir", default=None,
                        help="Explicit run directory (auto-generated if omitted).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    neo4j_uri  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASS", "password")
    config = LLMConfig.from_env()

    if "openai.com" not in config.base_url.rstrip("/"):
        raise RuntimeError(
            f"Batch API requires OpenAI (LLM_BASE_URL={config.base_url}). "
            "Set LLM_BASE_URL=https://api.openai.com/v1."
        )

    run_dir = Path(args.run_dir) if args.run_dir else None

    with Neo4jClient(uri=neo4j_uri, username=neo4j_user, password=neo4j_pass) as neo4j:
        with ProteinBatchClient(neo4j=neo4j, config=config) as client:

            if args.sample:
                print(f"Sampling {args.per_size} groups × 4 sizes = up to {args.per_size * 4} requests…")
                groups = client.sample_disjoint_groups(
                    per_size=args.per_size,
                    min_features=args.min_features,
                    require_structure=not args.no_structure_filter,
                )
                sizes = {1: 0, 2: 0, 3: 0, 4: 0}
                for g in groups:
                    sizes[len(g)] += 1
                print(f"  size-1: {sizes[1]}  size-2: {sizes[2]}  "
                      f"size-3: {sizes[3]}  size-4: {sizes[4]}")
                structures_dir = Path(args.structures_dir) if args.structures_dir else None
                client.submit(groups=groups, run_dir=run_dir, max_structures=args.max_structures,
                              structures_dir=structures_dir)

            elif args.submit:
                if not args.accessions:
                    raise ValueError("--submit requires --accessions.")
                groups = [[acc] for acc in args.accessions]
                structures_dir = Path(args.structures_dir) if args.structures_dir else None
                client.submit(groups=groups, run_dir=run_dir, max_structures=args.max_structures,
                              structures_dir=structures_dir)

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
                client.fetch(args.fetch, run_dir=run_dir)


if __name__ == "__main__":
    main()
