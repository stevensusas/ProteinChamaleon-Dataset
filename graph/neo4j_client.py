"""
Neo4j graph client for the ProteinChameleon knowledge graph.

Node labels:    Protein, PubMedArticle, ProteinFeature, GOTerm, Pathway
Relationships:  INTERACTS_WITH, HAS_GO_ANNOTATION, HAS_FEATURE,
                EVIDENCED_BY, CITED_IN, IN_PATHWAY, ENZYMATIC_REACTION
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from neo4j import GraphDatabase, Session

logger = logging.getLogger(__name__)


# ── Cypher constraint / index setup ──────────────────────────────────────────

CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Protein)       REQUIRE p.accession IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:PubMedArticle) REQUIRE a.pmid IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (f:ProteinFeature) REQUIRE f.accession IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (g:GOTerm)        REQUIRE g.go_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (pw:Pathway)      REQUIRE pw.kegg_id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS FOR (p:Protein)       ON (p.gene_name)",
    "CREATE INDEX IF NOT EXISTS FOR (p:Protein)       ON (p.taxon_id)",
    "CREATE INDEX IF NOT EXISTS FOR (g:GOTerm)        ON (g.aspect)",
    "CREATE INDEX IF NOT EXISTS FOR (f:ProteinFeature) ON (f.source_database)",
    "CREATE INDEX IF NOT EXISTS FOR (pw:Pathway)      ON (pw.name)",
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

    def merge_pubmed_article(self, s: Session, props: dict) -> None:
        """Upsert a PubMedArticle node keyed on pmid."""
        s.run(
            """
            MERGE (a:PubMedArticle {pmid: $pmid})
            SET a += $props
            """,
            pmid=props["pmid"],
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

    def merge_go_term(self, s: Session, props: dict) -> None:
        """Upsert a GOTerm node keyed on go_id."""
        s.run(
            """
            MERGE (g:GOTerm {go_id: $go_id})
            SET g += $props
            """,
            go_id=props["go_id"],
            props=props,
        )

    def merge_pathway(self, s: Session, props: dict) -> None:
        """Upsert a KEGG Pathway node keyed on kegg_id."""
        s.run(
            """
            MERGE (pw:Pathway {kegg_id: $kegg_id})
            SET pw += $props
            """,
            kegg_id=props["kegg_id"],
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

    def merge_go_annotation(
        self, s: Session, protein_acc: str, go_id: str, props: dict
    ) -> None:
        """HAS_GO_ANNOTATION: (Protein)-[:HAS_GO_ANNOTATION]->(GOTerm)"""
        s.run(
            """
            MATCH (p:Protein {accession: $acc})
            MATCH (g:GOTerm {go_id: $go_id})
            MERGE (p)-[r:HAS_GO_ANNOTATION {go_id: $go_id, qualifier: $qualifier}]->(g)
            SET r += $props
            """,
            acc=protein_acc,
            go_id=go_id,
            qualifier=props.get("qualifier", ""),
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

    def merge_feature_evidence(
        self, s: Session, protein_acc: str, feature_acc: str, pmid: str
    ) -> None:
        """
        EVIDENCED_BY: link a protein's feature match to a PubMed article.
        Models: (Protein)-[:HAS_FEATURE]->(Feature)-[:EVIDENCED_BY]->(Article)
        We also add a direct CITED_IN from Protein → Article.
        """
        s.run(
            """
            MATCH (p:Protein {accession: $acc})
            MATCH (f:ProteinFeature {accession: $feat_acc})
            MATCH (a:PubMedArticle {pmid: $pmid})
            MERGE (p)-[:CITED_IN]->(a)
            MERGE (f)-[:EVIDENCED_BY]->(a)
            """,
            acc=protein_acc,
            feat_acc=feature_acc,
            pmid=pmid,
        )

    def merge_cited_in(self, s: Session, protein_acc: str, pmid: str) -> None:
        """CITED_IN: (Protein)-[:CITED_IN]->(PubMedArticle)"""
        s.run(
            """
            MATCH (p:Protein {accession: $acc})
            MATCH (a:PubMedArticle {pmid: $pmid})
            MERGE (p)-[:CITED_IN]->(a)
            """,
            acc=protein_acc,
            pmid=pmid,
        )

    def merge_in_pathway(
        self, s: Session, protein_acc: str, kegg_id: str, props: dict
    ) -> None:
        """IN_PATHWAY: (Protein)-[:IN_PATHWAY]->(Pathway)"""
        s.run(
            """
            MATCH (p:Protein {accession: $acc})
            MATCH (pw:Pathway {kegg_id: $kegg_id})
            MERGE (p)-[r:IN_PATHWAY]->(pw)
            SET r += $props
            """,
            acc=protein_acc,
            kegg_id=kegg_id,
            props=props,
        )

    def merge_enzymatic_relation(
        self, s: Session, acc_a: str, acc_b: str, props: dict
    ) -> None:
        """
        ENZYMATIC_REACTION: two proteins linked by a shared enzyme (EC number)
        within the same KEGG pathway.
        (Protein)-[:ENZYMATIC_REACTION]->(Protein)
        """
        s.run(
            """
            MATCH (a:Protein {accession: $acc_a})
            MATCH (b:Protein {accession: $acc_b})
            MERGE (a)-[r:ENZYMATIC_REACTION {ec_number: $ec, pathway_id: $pathway}]->(b)
            SET r += $props
            """,
            acc_a=acc_a,
            acc_b=acc_b,
            ec=props.get("ec_number", ""),
            pathway=props.get("pathway_id", ""),
            props=props,
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
        - GOTerm nodes + HAS_GO_ANNOTATION relationships
        - PubMedArticle nodes + CITED_IN / EVIDENCED_BY relationships
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
                return {
                    "protein": None,
                    "features": [],
                    "feature_relationships": [],
                    "go_terms": [],
                    "go_relationships": [],
                    "articles": [],
                    "article_relationships": {
                        "cited_in": [],
                        "feature_evidenced_by": [],
                    },
                }

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
                rel_props = row["rel_props"]
                features.append(feature)
                feature_relationships.append(
                    {
                        "protein_accession": accession,
                        "feature_accession": feature.get("accession", ""),
                        "properties": rel_props,
                    }
                )

            go_rows = s.run(
                """
                MATCH (p:Protein {accession: $acc})-[r:HAS_GO_ANNOTATION]->(g:GOTerm)
                RETURN properties(g) AS go_term, properties(r) AS rel_props
                """,
                acc=accession,
            )
            go_terms: list[dict[str, Any]] = []
            go_relationships: list[dict[str, Any]] = []
            for row in go_rows:
                go_term = row["go_term"]
                rel_props = row["rel_props"]
                go_terms.append(go_term)
                go_relationships.append(
                    {
                        "protein_accession": accession,
                        "go_id": go_term.get("go_id", ""),
                        "properties": rel_props,
                    }
                )

            cited_rows = s.run(
                """
                MATCH (p:Protein {accession: $acc})-[r:CITED_IN]->(a:PubMedArticle)
                RETURN properties(a) AS article, properties(r) AS rel_props
                """,
                acc=accession,
            )
            articles_by_pmid: dict[str, dict[str, Any]] = {}
            cited_in_relationships: list[dict[str, Any]] = []
            for row in cited_rows:
                article = row["article"]
                rel_props = row["rel_props"]
                pmid = article.get("pmid", "")
                if pmid:
                    articles_by_pmid[pmid] = article
                cited_in_relationships.append(
                    {
                        "protein_accession": accession,
                        "pmid": pmid,
                        "properties": rel_props,
                    }
                )

            evidence_rows = s.run(
                """
                MATCH (p:Protein {accession: $acc})-[:HAS_FEATURE]->(f:ProteinFeature)-[r:EVIDENCED_BY]->(a:PubMedArticle)
                RETURN properties(f) AS feature, properties(a) AS article, properties(r) AS rel_props
                """,
                acc=accession,
            )
            feature_evidence_relationships: list[dict[str, Any]] = []
            for row in evidence_rows:
                feature = row["feature"]
                article = row["article"]
                rel_props = row["rel_props"]
                pmid = article.get("pmid", "")
                if pmid:
                    articles_by_pmid[pmid] = article
                feature_evidence_relationships.append(
                    {
                        "feature_accession": feature.get("accession", ""),
                        "pmid": pmid,
                        "properties": rel_props,
                    }
                )

            return {
                "protein": protein_row["protein"],
                "features": features,
                "feature_relationships": feature_relationships,
                "go_terms": go_terms,
                "go_relationships": go_relationships,
                "articles": list(articles_by_pmid.values()),
                "article_relationships": {
                    "cited_in": cited_in_relationships,
                    "feature_evidenced_by": feature_evidence_relationships,
                },
            }

    def get_graph_stats(self) -> dict:
        """Return node and relationship counts per label/type."""
        with self.session() as s:
            nodes = dict(s.run(
                """
                CALL apoc.meta.stats() YIELD labels
                RETURN labels
                """
            ).single() or {})
            counts = s.run(
                """
                CALL {
                  MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt
                  UNION ALL
                  MATCH ()-[r]->() RETURN type(r) AS label, count(r) AS cnt
                }
                RETURN label, cnt ORDER BY cnt DESC
                """
            )
            return {r["label"]: r["cnt"] for r in counts}

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
