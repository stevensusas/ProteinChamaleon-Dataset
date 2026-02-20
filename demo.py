"""
ProteinChameleon Dataset — Graph Expansion Demo
================================================
Starts from one seed protein (TP53), fetches up to 5 STRING interaction
partners, and fills each partner on the spot (full UniProt + PDB + AlphaFold
+ InterPro + GO). No BFS.

Usage:
  # Start Neo4j first (Docker):
  #   docker run -p 7474:7474 -p 7687:7687 \\
  #     -e NEO4J_AUTH=neo4j/password neo4j:5

  python demo.py

  # Or with custom connection settings:
  NEO4J_URI=bolt://localhost:7687 \\
  NEO4J_USER=neo4j \\
  NEO4J_PASS=password \\
  python demo.py
"""

import logging
import os
import sys

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
    max_partners = int(os.getenv("MAX_PARTNERS", "5"))

    logger.info("=" * 60)
    logger.info("ProteinChameleon Graph Expansion Demo")
    logger.info("=" * 60)
    logger.info("Seed:             %s", SEED_PROTEIN)
    logger.info("STRING threshold: %d / 1000", string_threshold)
    logger.info("Max partners:     %d (filled on the spot)", max_partners)
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

    with neo4j, GraphExpansionPipeline(
        neo4j=neo4j,
        string_score_threshold=string_threshold,
        max_partners_per_protein=max_partners,
    ) as pipeline:
        stats = pipeline.run(seed_accessions=[SEED_PROTEIN])

    logger.info("=" * 60)
    logger.info("Done. Stats: %s", stats)
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
