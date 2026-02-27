"""CLI demo for generating a protein description with the LLM client.

Usage:
  source .env
  python -m llm.demo --accession P04637
"""

from __future__ import annotations

import argparse
import os

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASS", "password")

    config = LLMConfig.from_env()

    with Neo4jClient(uri=neo4j_uri, username=neo4j_user, password=neo4j_pass) as neo4j:
        with ProteinDescriptionLLMClient(neo4j=neo4j, config=config) as llm:
            text = llm.describe_protein(args.accession)

    print(text)


if __name__ == "__main__":
    main()
