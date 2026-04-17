"""
Neo4j graph client for the ProteinChameleon knowledge graph.

Node labels:    Protein, ProteinFeature
Relationships:  INTERACTS_WITH, HAS_FEATURE
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from neo4j import GraphDatabase, Session

logger = logging.getLogger(__name__)


# ── Cypher constraint / index setup ──────────────────────────────────────────

CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Protein)        REQUIRE p.accession IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (f:ProteinFeature) REQUIRE f.accession IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS FOR (p:Protein)        ON (p.gene_name)",
    "CREATE INDEX IF NOT EXISTS FOR (p:Protein)        ON (p.taxon_id)",
    "CREATE INDEX IF NOT EXISTS FOR (f:ProteinFeature) ON (f.source_database)",
]


# ── Main client ───────────────────────────────────────────────────────────────

class Neo4jClient:
    """
    Thin wrapper around the official Neo4j Python driver.

    All write operations use MERGE so the graph is safe to re-populate
    incrementally without creating duplicate nodes.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
    ) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        logger.info("Connected to Neo4j at %s", uri)

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @contextmanager
    def session(self):
        with self.driver.session() as s:
            yield s

    # ── Clear / schema ───────────────────────────────────────────────────────

    def clear_graph(self) -> None:
        """Remove all nodes and relationships. Use before a fresh pipeline run."""
        with self.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
        logger.info("Neo4j graph cleared.")

    def setup_schema(self) -> None:
        """Create uniqueness constraints and indexes. Safe to call multiple times."""
        with self.session() as s:
            for stmt in CONSTRAINTS + INDEXES:
                try:
                    s.run(stmt)
                except Exception as exc:
                    logger.warning("Schema statement skipped (%s): %s", exc, stmt[:60])
        logger.info("Neo4j schema constraints and indexes applied.")

    # ── Node upserts ─────────────────────────────────────────────────────────

    def merge_protein(self, s: Session, props: dict) -> None:
        """Upsert a Protein node keyed on accession."""
        s.run(
            """
            MERGE (p:Protein {accession: $accession})
            SET p += $props
            """,
            accession=props["accession"],
            props=props,
        )

    def merge_protein_feature(self, s: Session, props: dict) -> None:
        """Upsert a ProteinFeature node (InterPro entry) keyed on accession."""
        s.run(
            """
            MERGE (f:ProteinFeature {accession: $accession})
            SET f += $props
            """,
            accession=props["accession"],
            props=props,
        )

    # ── Relationship upserts ──────────────────────────────────────────────────

    def merge_ppi(
        self, s: Session, acc_a: str, acc_b: str, props: dict
    ) -> None:
        """
        INTERACTS_WITH: (Protein)-[:INTERACTS_WITH]->(Protein)
        Bidirectional interaction stored as a directed edge from A→B.
        """
        s.run(
            """
            MATCH (a:Protein {accession: $acc_a})
            MATCH (b:Protein {accession: $acc_b})
            MERGE (a)-[r:INTERACTS_WITH {string_pair: $pair}]->(b)
            SET r += $props
            """,
            acc_a=acc_a,
            acc_b=acc_b,
            pair=f"{min(acc_a, acc_b)}:{max(acc_a, acc_b)}",
            props=props,
        )

    def merge_has_feature(
        self,
        s: Session,
        protein_acc: str,
        feature_acc: str,
        props: dict,
    ) -> None:
        """HAS_FEATURE: (Protein)-[:HAS_FEATURE]->(ProteinFeature)"""
        s.run(
            """
            MATCH (p:Protein {accession: $acc})
            MATCH (f:ProteinFeature {accession: $feat_acc})
            MERGE (p)-[r:HAS_FEATURE {protein: $acc, feature: $feat_acc, start: $start, end: $end}]->(f)
            SET r += $props
            """,
            acc=protein_acc,
            feat_acc=feature_acc,
            start=props.get("start", 0),
            end=props.get("end", 0),
            props=props,
        )

    # ── Batch upserts (UNWIND) ────────────────────────────────────────────────

    def merge_proteins_batch(self, batch: list[dict]) -> None:
        """Upsert a batch of Protein nodes in a single transaction."""
        with self.session() as s:
            s.run(
                """
                UNWIND $batch AS props
                MERGE (p:Protein {accession: props.accession})
                SET p += props
                """,
                batch=batch,
            )

    def merge_features_batch(self, batch: list[dict]) -> None:
        """Upsert a batch of ProteinFeature nodes in a single transaction."""
        with self.session() as s:
            s.run(
                """
                UNWIND $batch AS props
                MERGE (f:ProteinFeature {accession: props.accession})
                SET f += props
                """,
                batch=batch,
            )

    def merge_has_feature_batch(self, batch: list[dict]) -> None:
        """
        Upsert a batch of HAS_FEATURE edges in a single transaction.
        Each dict: {protein_acc, feature_acc, start, end}
        Silently skips rows where Protein or ProteinFeature is missing.
        """
        with self.session() as s:
            s.run(
                """
                UNWIND $batch AS row
                MATCH (p:Protein {accession: row.protein_acc})
                MATCH (f:ProteinFeature {accession: row.feature_acc})
                MERGE (p)-[r:HAS_FEATURE {
                    protein: row.protein_acc,
                    feature: row.feature_acc,
                    start: row.start,
                    end: row.end
                }]->(f)
                """,
                batch=batch,
            )

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_protein_accessions(self) -> list[str]:
        """Return all protein accessions currently in the graph."""
        with self.session() as s:
            result = s.run("MATCH (p:Protein) RETURN p.accession AS acc")
            return [r["acc"] for r in result]

    def get_protein_context(self, accession: str) -> dict[str, Any]:
        """
        Return one protein and its connected context:
        - Protein node properties
        - ProteinFeature nodes + HAS_FEATURE relationships
        """
        with self.session() as s:
            protein_row = s.run(
                """
                MATCH (p:Protein {accession: $acc})
                RETURN properties(p) AS protein
                """,
                acc=accession,
            ).single()

            if not protein_row:
                return {"protein": None, "features": [], "feature_relationships": []}

            feature_rows = s.run(
                """
                MATCH (p:Protein {accession: $acc})-[r:HAS_FEATURE]->(f:ProteinFeature)
                RETURN properties(f) AS feature, properties(r) AS rel_props
                """,
                acc=accession,
            )
            features: list[dict[str, Any]] = []
            feature_relationships: list[dict[str, Any]] = []
            for row in feature_rows:
                feature = row["feature"]
                features.append(feature)
                feature_relationships.append(
                    {
                        "protein_accession": accession,
                        "feature_accession": feature.get("accession", ""),
                        "properties": row["rel_props"],
                    }
                )

            return {
                "protein": protein_row["protein"],
                "features": features,
                "feature_relationships": feature_relationships,
            }

    def get_counts(self) -> dict[str, int]:
        """Return node and relationship counts per label/type (no APOC needed)."""
        with self.session() as s:
            rows = s.run(
                """
                CALL {
                  MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt
                  UNION ALL
                  MATCH ()-[r]->() RETURN type(r) AS label, count(r) AS cnt
                }
                RETURN label, cnt ORDER BY cnt DESC
                """
            )
            return {r["label"]: r["cnt"] for r in rows}

    def get_graph_stats(self) -> dict:
        """Alias for get_counts() (backward compat)."""
        return self.get_counts()

    def get_protein_neighbors(
        self, accession: str, max_hops: int = 1
    ) -> list[str]:
        """Return accessions of proteins within max_hops interaction edges."""
        with self.session() as s:
            result = s.run(
                """
                MATCH (p:Protein {accession: $acc})-[:INTERACTS_WITH*1..$hops]-(n:Protein)
                RETURN DISTINCT n.accession AS acc
                """,
                acc=accession,
                hops=max_hops,
            )
            return [r["acc"] for r in result]
