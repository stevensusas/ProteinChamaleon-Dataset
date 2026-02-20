"""Integration tests for the STRING PPI database API client."""

import pytest

from clients.string_db import StringMapping, StringInteraction, StringEnrichment
from .conftest import TP53_GENE


pytestmark = pytest.mark.integration

HUMAN_TAXON = 9606


class TestSTRINGMapIdentifiers:
    def test_maps_gene_name(self, string_db):
        mappings = string_db.map_identifiers([TP53_GENE], taxon=HUMAN_TAXON)
        assert isinstance(mappings, list)
        assert len(mappings) >= 1

    def test_mapping_schema(self, string_db):
        mappings = string_db.map_identifiers([TP53_GENE], taxon=HUMAN_TAXON)
        m = mappings[0]
        assert isinstance(m, StringMapping)
        assert m.stringId is not None
        assert "9606." in m.stringId  # human STRING IDs contain taxon

    def test_preferred_name_matches(self, string_db):
        mappings = string_db.map_identifiers([TP53_GENE], taxon=HUMAN_TAXON)
        assert mappings[0].preferredName == TP53_GENE

    def test_batch_mapping(self, string_db):
        genes = ["TP53", "MDM2", "BRCA1"]
        mappings = string_db.map_identifiers(genes, taxon=HUMAN_TAXON)
        assert len(mappings) == 3
        names = {m.preferredName for m in mappings}
        assert names == set(genes)


class TestSTRINGInteractionPartners:
    def test_returns_interactions(self, string_db):
        interactions = string_db.get_interaction_partners(
            [TP53_GENE], taxon=HUMAN_TAXON, required_score=700, limit=20
        )
        assert isinstance(interactions, list)
        assert len(interactions) > 0

    def test_interaction_schema(self, string_db):
        interactions = string_db.get_interaction_partners(
            [TP53_GENE], taxon=HUMAN_TAXON, required_score=700, limit=10
        )
        for i in interactions:
            assert isinstance(i, StringInteraction)
            assert 0 <= i.score <= 1000
            assert isinstance(i.confidence, float)
            assert 0.0 <= i.confidence <= 1.0

    def test_tp53_interacts_with_mdm2(self, string_db):
        """TP53–MDM2 is one of the most well-documented PPIs."""
        interactions = string_db.get_interaction_partners(
            [TP53_GENE], taxon=HUMAN_TAXON, required_score=700, limit=50
        )
        partner_names = {i.preferredName_B for i in interactions}
        assert "MDM2" in partner_names

    def test_score_threshold_respected(self, string_db):
        # STRING API returns scores in 0–1 scale; required_score 700 = 0.7
        interactions = string_db.get_interaction_partners(
            [TP53_GENE], taxon=HUMAN_TAXON, required_score=700, limit=50
        )
        assert all(i.score >= 0.7 for i in interactions)

    def test_high_confidence_flag(self, string_db):
        interactions = string_db.get_interaction_partners(
            [TP53_GENE], taxon=HUMAN_TAXON, required_score=700, limit=10
        )
        assert all(i.is_high_confidence for i in interactions)

    def test_subscores_are_numeric(self, string_db):
        # STRING API returns subscores as floats in 0–1 scale
        interactions = string_db.get_interaction_partners(
            [TP53_GENE], taxon=HUMAN_TAXON, required_score=400, limit=5
        )
        for i in interactions:
            assert isinstance(i.escore, (int, float))
            assert isinstance(i.dscore, (int, float))
            assert isinstance(i.tscore, (int, float))

    def test_preferred_names_present(self, string_db):
        interactions = string_db.get_interaction_partners(
            [TP53_GENE], taxon=HUMAN_TAXON, required_score=700, limit=5
        )
        for i in interactions:
            assert i.preferredName_A
            assert i.preferredName_B


class TestSTRINGEnrichment:
    def test_returns_enrichment(self, string_db):
        # p53 pathway genes
        genes = ["TP53", "MDM2", "CDKN1A", "ATM", "CHEK2"]
        enrichments = string_db.get_enrichment(genes, taxon=HUMAN_TAXON)
        assert isinstance(enrichments, list)
        assert len(enrichments) > 0

    def test_enrichment_schema(self, string_db):
        genes = ["TP53", "MDM2", "CDKN1A"]
        enrichments = string_db.get_enrichment(genes, taxon=HUMAN_TAXON)
        for e in enrichments:
            assert isinstance(e, StringEnrichment)
            assert e.category
            assert e.term
            assert 0.0 <= e.p_value <= 1.0
            assert 0.0 <= e.fdr <= 1.0

    def test_enrichment_has_go_categories(self, string_db):
        genes = ["TP53", "MDM2", "CDKN1A", "ATM", "CHEK2"]
        enrichments = string_db.get_enrichment(genes, taxon=HUMAN_TAXON)
        categories = {e.category for e in enrichments}
        # Should include GO process, function, or component
        go_cats = {c for c in categories if "GO" in c or "Process" in c or "Function" in c}
        assert len(go_cats) > 0


class TestSTRINGVersion:
    def test_get_version(self, string_db):
        version = string_db.get_version()
        assert isinstance(version, (list, dict))
