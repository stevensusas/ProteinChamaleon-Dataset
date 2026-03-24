"""
ProteinChameleon Dataset — Graph Expansion Demo
================================================
Starts from one seed protein (TP53) and performs multithreaded BFS expansion
through the STRING interaction network.  Every discovered partner is fully
annotated (UniProt + PDB + AlphaFold + InterPro + GO) and its own partners
are enqueued in turn.  Expansion continues until no new partners are found
or an optional protein cap (MAX_PROTEINS) is reached.

Usage:
  # Start Neo4j first (Docker):
  #   docker run -p 7474:7474 -p 7687:7687 \\
  #     -e NEO4J_AUTH=neo4j/password neo4j:5

  python demo.py

  # Or with custom settings:
  NEO4J_URI=bolt://localhost:7687 \\
  NEO4J_USER=neo4j \\
  NEO4J_PASS=password \\
  STRING_SCORE=700 \\
  MAX_PROTEINS=50 \\
  WORKERS=6 \\
  python demo.py
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


# ── Seed protein ──────────────────────────────────────────────────────────────

SEED_PROTEIN = "P04637"  # TP53 — Tumor suppressor p53


def main() -> None:
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASS", "password")
    string_threshold = int(os.getenv("STRING_SCORE", "700"))
    max_proteins_env = os.getenv("MAX_PROTEINS", "")
    max_proteins = int(max_proteins_env) if max_proteins_env else None
    workers = int(os.getenv("WORKERS", "6"))

    logger.info("=" * 60)
    logger.info("ProteinChameleon Graph Expansion Demo (multithreaded BFS)")
    logger.info("=" * 60)
    logger.info("Seed:             %s", SEED_PROTEIN)
    logger.info("STRING threshold: %d / 1000", string_threshold)
    logger.info("Max proteins:     %s", max_proteins or "unlimited")
    logger.info("Workers:          %d", workers)
    logger.info("Neo4j:            %s (user=%s)", neo4j_uri, neo4j_user)
    logger.info("=" * 60)

    from graph.neo4j_client import Neo4jClient
    from graph.pipeline import GraphExpansionPipeline

    try:
        neo4j = Neo4jClient(uri=neo4j_uri, username=neo4j_user, password=neo4j_pass)
    except Exception as exc:
        logger.error(
            "Cannot connect to Neo4j at %s: %s\n"
            "Start Neo4j with:\n"
            "  docker run -p 7474:7474 -p 7687:7687 "
            "-e NEO4J_AUTH=neo4j/password neo4j:5",
            neo4j_uri, exc,
        )
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).resolve().parent / "output" / f"run_{timestamp}"

    with neo4j, GraphExpansionPipeline(
        neo4j=neo4j,
        string_score_threshold=string_threshold,
        max_proteins=max_proteins,
        workers=workers,
    ) as pipeline:
        stats = pipeline.run(
            seed_accessions=[SEED_PROTEIN],
            output_dir=output_dir,
        )

    logger.info("=" * 60)
    logger.info("Done. Stats: %s", stats)
    logger.info("Output dir: %s", output_dir)
    logger.info(
        "Open Neo4j Browser at http://localhost:7474 to explore the graph."
    )
    logger.info(
        "Example Cypher:\n"
        "  MATCH (p:Protein)-[r:INTERACTS_WITH]->(q:Protein) RETURN p,r,q LIMIT 25\n"
        "  MATCH (p:Protein)-[:HAS_GO_ANNOTATION]->(g:GOTerm) RETURN p.gene_name, g.name, g.aspect\n"
        "  MATCH (p:Protein)-[:HAS_FEATURE]->(f:ProteinFeature) RETURN p.gene_name, f.name, f.type\n"
        "  MATCH (p:Protein) WHERE size(p.pdb_ids) > 0 RETURN p.gene_name, p.pdb_ids, p.alphafold_entry_ids"
    )


if __name__ == "__main__":
    main()
