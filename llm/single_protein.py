"""CLI for generating a single-protein structure-interleaved description.

Usage:
  source .env
  python -m llm.single_protein --accession P04637
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any

import httpx

from graph.neo4j_client import Neo4jClient
from llm import LLMConfig, ProteinDescriptionLLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a natural-language description for a protein accession."
    )
    parser.add_argument(
        "--accession",
        default="P04637",
        help="UniProt accession to describe (default: P04637).",
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=5,
        help="Maximum number of full structures to download per source (default: 5).",
    )
    return parser.parse_args()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _download_file(url: str, path: Path) -> bool:
    if path.exists() and path.stat().st_size > 0:
        return True  # already downloaded
    try:
        with httpx.Client(timeout=45.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
        path.write_bytes(resp.content)
        return True
    except Exception:
        return False


def _slice_pdb_by_residue_range(
    source_path: Path, output_path: Path, start: int, end: int
) -> bool:
    try:
        lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return False

    if start <= 0 or end <= 0 or end < start:
        return False

    kept: list[str] = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            resseq_raw = line[22:26].strip()
            try:
                resseq = int(resseq_raw)
            except ValueError:
                continue
            if start <= resseq <= end:
                kept.append(line)

    if not kept:
        return False

    output_text = "\n".join(kept + ["END"]) + "\n"
    output_path.write_text(output_text, encoding="utf-8")
    return True


def _fetch_uniprot_pdb_mappings(uniprot_acc: str, pdb_id: str) -> list[dict[str, int | str]]:
    """
    Fetch UniProt→PDB residue mappings from PDBe/SIFTS.
    Returns list of dicts with:
      chain_id, unp_start, unp_end, pdb_start, pdb_end
    """
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            payload = resp.json()
    except Exception:
        return []

    data = payload.get(pdb_id.lower(), {})
    uni_map = data.get("UniProt", {})
    acc_payload = uni_map.get(uniprot_acc, {})
    mappings = acc_payload.get("mappings", [])
    parsed: list[dict[str, int | str]] = []

    for m in mappings:
        chain_id = str(m.get("chain_id", "")).strip()
        unp_start = (
            m.get("unp_start")
            or (m.get("unp_start_residue_number") if isinstance(m, dict) else None)
            or (m.get("start", {}) or {}).get("residue_number")
        )
        unp_end = (
            m.get("unp_end")
            or (m.get("unp_end_residue_number") if isinstance(m, dict) else None)
            or (m.get("end", {}) or {}).get("residue_number")
        )
        pdb_start = (
            m.get("start", {}) or {}
        ).get("author_residue_number") or (m.get("start", {}) or {}).get("residue_number")
        pdb_end = (
            m.get("end", {}) or {}
        ).get("author_residue_number") or (m.get("end", {}) or {}).get("residue_number")

        if (
            chain_id
            and isinstance(unp_start, int)
            and isinstance(unp_end, int)
            and isinstance(pdb_start, int)
            and isinstance(pdb_end, int)
        ):
            parsed.append(
                {
                    "chain_id": chain_id,
                    "unp_start": unp_start,
                    "unp_end": unp_end,
                    "pdb_start": pdb_start,
                    "pdb_end": pdb_end,
                }
            )
    return parsed


def _project_uniprot_feature_to_pdb_ranges(
    feature_start: int,
    feature_end: int,
    mappings: list[dict[str, int | str]],
) -> list[tuple[str, int, int]]:
    """
    Project a UniProt feature span to one or more PDB chain residue ranges
    using SIFTS segment mappings.
    """
    projected: list[tuple[str, int, int]] = []
    if feature_start <= 0 or feature_end <= 0 or feature_end < feature_start:
        return projected

    for m in mappings:
        chain_id = str(m["chain_id"])
        unp_start = int(m["unp_start"])
        unp_end = int(m["unp_end"])
        pdb_start = int(m["pdb_start"])
        pdb_end = int(m["pdb_end"])

        overlap_start = max(feature_start, unp_start)
        overlap_end = min(feature_end, unp_end)
        if overlap_end < overlap_start:
            continue

        # Keep orientation if numbering direction is reversed.
        direction = 1 if pdb_end >= pdb_start else -1
        pdb_overlap_start = pdb_start + direction * (overlap_start - unp_start)
        pdb_overlap_end = pdb_start + direction * (overlap_end - unp_start)
        lo = min(pdb_overlap_start, pdb_overlap_end)
        hi = max(pdb_overlap_start, pdb_overlap_end)
        projected.append((chain_id, lo, hi))

    return projected


def _slice_pdb_by_chain_ranges(
    source_path: Path,
    output_path: Path,
    chain_ranges: list[tuple[str, int, int]],
) -> bool:
    try:
        lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return False

    if not chain_ranges:
        return False

    by_chain: dict[str, list[tuple[int, int]]] = {}
    for chain_id, start, end in chain_ranges:
        if start <= 0 or end <= 0 or end < start:
            continue
        by_chain.setdefault(chain_id, []).append((start, end))

    if not by_chain:
        return False

    kept: list[str] = []
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        chain_id = line[21:22].strip()
        if chain_id not in by_chain:
            continue
        resseq_raw = line[22:26].strip()
        try:
            resseq = int(resseq_raw)
        except ValueError:
            continue
        if any(start <= resseq <= end for start, end in by_chain[chain_id]):
            kept.append(line)

    if not kept:
        return False

    output_text = "\n".join(kept + ["END"]) + "\n"
    output_path.write_text(output_text, encoding="utf-8")
    return True


def prepare_structure_files(
    accession: str,
    protein: dict[str, Any],
    feature_rels: list[dict[str, Any]],
    structure_dir: Path,
    max_structures: int,
    file_prefix: str = "",
    output_subdir: str = "",
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    full_structure_files: list[Path] = []
    pdb_file_by_id: dict[str, Path] = {}
    prefix = f"{_safe_name(file_prefix)}_" if file_prefix else ""
    subdir = _safe_name(output_subdir) if output_subdir else ""
    target_dir = structure_dir / subdir if subdir else structure_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    pdb_ids = [str(x) for x in protein.get("pdb_ids", []) if x][:max_structures]
    for pdb_id in pdb_ids:
        file_name = f"{prefix}pdb_{_safe_name(pdb_id)}.pdb"
        rel_path = f"structure/{subdir}/{file_name}" if subdir else f"structure/{file_name}"
        out_path = target_dir / file_name
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        if _download_file(url, out_path):
            full_structure_files.append(out_path)
            pdb_file_by_id[pdb_id] = out_path
            manifest.append(
                {
                    "filename": file_name,
                    "placeholder": f"[{rel_path}]",
                    "source": "pdb",
                    "pdb_id": pdb_id,
                    "path": rel_path,
                }
            )

    alphafold_urls = [
        str(u) for u in protein.get("alphafold_pdb_urls", []) if isinstance(u, str) and u
    ][:max_structures]
    if not alphafold_urls and protein.get("alphafold_pdb_url"):
        alphafold_urls = [str(protein["alphafold_pdb_url"])]

    for idx, url in enumerate(alphafold_urls, start=1):
        file_name = f"{prefix}alphafold_{idx}.pdb"
        rel_path = f"structure/{subdir}/{file_name}" if subdir else f"structure/{file_name}"
        out_path = target_dir / file_name
        if _download_file(url, out_path):
            full_structure_files.append(out_path)
            manifest.append(
                {
                    "filename": file_name,
                    "placeholder": f"[{rel_path}]",
                    "source": "alphafold",
                    "url": url,
                    "path": rel_path,
                }
            )

    reference_structure = full_structure_files[0] if full_structure_files else None
    if reference_structure is None:
        return manifest

    mapping_cache: dict[str, list[dict[str, int | str]]] = {}
    for rel in feature_rels:
        feat_acc = str(rel.get("feature_accession", "")).strip()
        props = rel.get("properties", {}) or {}
        start = props.get("start")
        end = props.get("end")
        if not feat_acc or not isinstance(start, int) or not isinstance(end, int):
            continue

        wrote_mapped_slice = False
        for pdb_id, pdb_path in pdb_file_by_id.items():
            if pdb_id not in mapping_cache:
                mapping_cache[pdb_id] = _fetch_uniprot_pdb_mappings(accession, pdb_id)
            chain_ranges = _project_uniprot_feature_to_pdb_ranges(
                start, end, mapping_cache[pdb_id]
            )
            if not chain_ranges:
                continue

            chain_tag = "_".join(sorted({cr[0] for cr in chain_ranges}))
            file_name = (
                f"{prefix}feature_{_safe_name(feat_acc)}_{start}_{end}_"
                f"pdb_{_safe_name(pdb_id)}_chain_{_safe_name(chain_tag)}.pdb"
            )
            rel_path = f"structure/{subdir}/{file_name}" if subdir else f"structure/{file_name}"
            out_path = target_dir / file_name
            if not _slice_pdb_by_chain_ranges(pdb_path, out_path, chain_ranges):
                continue
            manifest.append(
                {
                    "filename": file_name,
                    "placeholder": f"[{rel_path}]",
                    "source": "feature_slice",
                    "method": "chain_aware_sifts_mapping",
                    "feature_accession": feat_acc,
                    "start": start,
                    "end": end,
                    "pdb_id": pdb_id,
                    "chain_ranges": [
                        {"chain_id": c, "pdb_start": s, "pdb_end": e}
                        for c, s, e in chain_ranges
                    ],
                    "reference_structure": f"structure/{pdb_path.name}",
                    "path": rel_path,
                }
            )
            wrote_mapped_slice = True
            break

        if wrote_mapped_slice:
            continue

        file_name = f"{prefix}feature_{_safe_name(feat_acc)}_{start}_{end}.pdb"
        rel_path = f"structure/{subdir}/{file_name}" if subdir else f"structure/{file_name}"
        out_path = target_dir / file_name
        if _slice_pdb_by_residue_range(reference_structure, out_path, start, end):
            manifest.append(
                {
                    "filename": file_name,
                    "placeholder": f"[{rel_path}]",
                    "source": "feature_slice",
                    "method": "naive_residue_number_fallback",
                    "feature_accession": feat_acc,
                    "start": start,
                    "end": end,
                    "reference_structure": f"structure/{reference_structure.name}",
                    "path": rel_path,
                }
            )

    return manifest


def main() -> None:
    args = parse_args()

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASS", "password")

    config = LLMConfig.from_env()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_output_dir = Path(__file__).resolve().parent.parent / "output"
    run_dir = root_output_dir / f"{args.accession}_{timestamp}_single"
    structure_dir = run_dir / "structure"
    structure_dir.mkdir(parents=True, exist_ok=True)

    with Neo4jClient(uri=neo4j_uri, username=neo4j_user, password=neo4j_pass) as neo4j:
        context = neo4j.get_protein_context(args.accession)
        protein = context.get("protein")
        if not protein:
            raise ValueError(f"Protein not found in graph: {args.accession}")

        structure_manifest = prepare_structure_files(
            accession=args.accession,
            protein=protein,
            feature_rels=context.get("feature_relationships", []),
            structure_dir=structure_dir,
            max_structures=max(1, args.max_structures),
        )

        context_path = run_dir / "context.json"
        context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")

        structure_manifest_path = run_dir / "structure_manifest.json"
        structure_manifest_path.write_text(
            json.dumps(structure_manifest, indent=2), encoding="utf-8"
        )

        with ProteinDescriptionLLMClient(neo4j=neo4j, config=config) as llm:
            text = llm.describe_protein(
                args.accession, structure_files=structure_manifest
            )

    output_path = run_dir / "description.txt"
    output_path.write_text(text, encoding="utf-8")

    print(f"Wrote output folder: {run_dir}")
    print(f"- Description: {output_path}")
    print(f"- Context JSON: {run_dir / 'context.json'}")
    print(f"- Structure manifest: {run_dir / 'structure_manifest.json'}")
    print(f"- Structure files dir: {structure_dir}")
    print()
    print(text)


if __name__ == "__main__":
    main()
