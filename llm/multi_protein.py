"""CLI for generating an interaction-network description for multiple proteins.

Usage examples:
  source .env
  python -m llm.multi_protein --accessions P04637 P01116 P38398
  python -m llm.multi_protein --seed P04637 --max-proteins 5
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

from graph.neo4j_client import Neo4jClient
from llm import LLMConfig, ProteinDescriptionLLMClient
from llm.single_protein import prepare_structure_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a concise, structure-interleaved description for interacting proteins."
    )
    parser.add_argument(
        "--accessions",
        nargs="*",
        default=[],
        help="Explicit UniProt accessions to include (max 5).",
    )
    parser.add_argument(
        "--seed",
        default="",
        help="Seed accession used to auto-select interaction partners when --accessions is omitted.",
    )
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=5,
        help="Maximum proteins in the network output (default: 5, hard cap: 5).",
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=3,
        help="Maximum full structures downloaded per protein/source (default: 3).",
    )
    return parser.parse_args()


def _resolve_accessions(neo4j: Neo4jClient, args: argparse.Namespace) -> list[str]:
    max_proteins = max(1, min(5, args.max_proteins))
    if args.accessions:
        uniq: list[str] = []
        for acc in args.accessions:
            if acc not in uniq:
                uniq.append(acc)
        return uniq[:max_proteins]

    if not args.seed:
        raise ValueError("Provide either --accessions or --seed.")

    with neo4j.session() as s:
        rows = s.run(
            """
            MATCH (p:Protein {accession: $seed})
            OPTIONAL MATCH (p)-[r:INTERACTS_WITH]-(q:Protein)
            WITH p, q, coalesce(r.score, 0.0) AS score
            ORDER BY score DESC
            RETURN p.accession AS seed, q.accession AS partner
            """,
            seed=args.seed,
        )
        ordered: list[str] = []
        for row in rows:
            seed_acc = row["seed"]
            partner = row["partner"]
            if seed_acc and seed_acc not in ordered:
                ordered.append(seed_acc)
            if partner and partner not in ordered:
                ordered.append(partner)
            if len(ordered) >= max_proteins:
                break
    return ordered[:max_proteins]


def _fetch_interaction_edges(neo4j: Neo4jClient, accessions: list[str]) -> list[dict[str, Any]]:
    if not accessions:
        return []
    with neo4j.session() as s:
        rows = s.run(
            """
            MATCH (a:Protein)-[r:INTERACTS_WITH]-(b:Protein)
            WHERE a.accession IN $accs AND b.accession IN $accs
            RETURN a.accession AS source, b.accession AS target, properties(r) AS props
            """,
            accs=accessions,
        )
        return [
            {
                "source": row["source"],
                "target": row["target"],
                "properties": row["props"],
            }
            for row in rows
        ]


def main() -> None:
    args = parse_args()
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASS", "password")
    config = LLMConfig.from_env()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_output_dir = Path(__file__).resolve().parent.parent / "output"
    run_dir = root_output_dir / f"multi_{timestamp}"
    structure_dir = run_dir / "structure"
    structure_dir.mkdir(parents=True, exist_ok=True)

    with Neo4jClient(uri=neo4j_uri, username=neo4j_user, password=neo4j_pass) as neo4j:
        accessions = _resolve_accessions(neo4j, args)
        if not accessions:
            raise ValueError("No proteins resolved for multi-protein generation.")

        contexts: dict[str, dict[str, Any]] = {}
        structure_manifest: list[dict[str, Any]] = []
        for acc in accessions:
            context = neo4j.get_protein_context(acc)
            if not context.get("protein"):
                continue
            contexts[acc] = context
            structure_manifest.extend(
                prepare_structure_files(
                    accession=acc,
                    protein=context["protein"],
                    feature_rels=context.get("feature_relationships", []),
                    structure_dir=structure_dir,
                    max_structures=max(1, args.max_structures),
                    output_subdir=acc,
                )
            )

        if args.accessions:
            missing = [acc for acc in accessions if acc not in contexts]
            if missing:
                raise ValueError(
                    "Missing requested proteins in Neo4j graph context: "
                    + ", ".join(missing)
                    + ". Run ingestion first or use --seed for auto-selection."
                )

        selected = [acc for acc in accessions if acc in contexts]
        if not selected:
            raise ValueError("None of the selected proteins were found in Neo4j context.")

        interaction_edges = _fetch_interaction_edges(neo4j, selected)

        (run_dir / "selected_accessions.json").write_text(
            json.dumps(selected, indent=2), encoding="utf-8"
        )
        (run_dir / "contexts_by_accession.json").write_text(
            json.dumps(contexts, indent=2), encoding="utf-8"
        )
        (run_dir / "interaction_edges.json").write_text(
            json.dumps(interaction_edges, indent=2), encoding="utf-8"
        )
        (run_dir / "structure_manifest.json").write_text(
            json.dumps(structure_manifest, indent=2), encoding="utf-8"
        )

        with ProteinDescriptionLLMClient(neo4j=neo4j, config=config) as llm:
            text = llm.describe_protein_network(
                accessions=selected,
                contexts=contexts,
                interaction_edges=interaction_edges,
                structure_manifest=structure_manifest,
            )

    desc_path = run_dir / "description.txt"
    desc_path.write_text(text, encoding="utf-8")

    print(f"Wrote output folder: {run_dir}")
    print(f"- Description: {desc_path}")
    print(f"- Selected accessions: {run_dir / 'selected_accessions.json'}")
    print(f"- Contexts: {run_dir / 'contexts_by_accession.json'}")
    print(f"- Interaction edges: {run_dir / 'interaction_edges.json'}")
    print(f"- Structure manifest: {run_dir / 'structure_manifest.json'}")
    print(f"- Structure files dir: {structure_dir}")
    print()
    print(text)


if __name__ == "__main__":
    main()
