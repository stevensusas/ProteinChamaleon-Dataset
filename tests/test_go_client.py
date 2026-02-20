"""Integration tests for the Gene Ontology (QuickGO) API client."""

import pytest

from clients.go_client import GOAnnotation, GOTerm, GOAncestors, GOChildren
from .conftest import HBA1_ACC, TP53_ACC


pytestmark = pytest.mark.integration

# GO:0005344 = oxygen carrier activity (hemoglobin)
# GO:0004674 = protein serine/threonine kinase activity
# GO:0006915 = apoptotic process
OXYGEN_CARRIER_GO = "GO:0005344"
KINASE_GO = "GO:0004674"
APOPTOSIS_GO = "GO:0006915"


class TestGOAnnotations:
    def test_returns_list(self, go):
        annotations = go.get_annotations(HBA1_ACC)
        assert isinstance(annotations, list)

    def test_has_annotations(self, go):
        annotations = go.get_annotations(HBA1_ACC)
        assert len(annotations) > 0

    def test_annotation_schema(self, go):
        annotations = go.get_annotations(HBA1_ACC)
        for a in annotations:
            assert isinstance(a, GOAnnotation)
            assert a.geneProductId == HBA1_ACC
            assert a.goId.startswith("GO:")

    def test_go_ids_valid_format(self, go):
        annotations = go.get_annotations(HBA1_ACC)
        for a in annotations:
            parts = a.goId.split(":")
            assert len(parts) == 2
            assert parts[0] == "GO"
            assert parts[1].isdigit()
            assert len(parts[1]) == 7

    def test_go_aspects_valid(self, go):
        annotations = go.get_annotations(HBA1_ACC)
        valid_aspects = {"molecular_function", "biological_process", "cellular_component"}
        for a in annotations:
            if a.goAspect:
                assert a.goAspect in valid_aspects

    def test_filter_by_aspect_mf(self, go):
        annotations = go.get_annotations(HBA1_ACC, aspect="molecular_function")
        assert all(a.goAspect == "molecular_function" for a in annotations if a.goAspect)

    def test_filter_by_aspect_bp(self, go):
        annotations = go.get_annotations(HBA1_ACC, aspect="biological_process")
        assert all(a.goAspect == "biological_process" for a in annotations if a.goAspect)

    def test_hemoglobin_has_oxygen_carrier_annotation(self, go):
        annotations = go.get_annotations(HBA1_ACC)
        go_ids = {a.goId for a in annotations}
        assert OXYGEN_CARRIER_GO in go_ids

    def test_experimental_only_filter(self, go):
        annotations = go.get_annotations(HBA1_ACC, experimental_only=True)
        experimental_codes = {"IDA", "IMP", "IGI", "IPI", "IEP", "EXP", "HDA", "HMP", "HGI", "HEP"}
        for a in annotations:
            if a.goEvidence:
                assert a.goEvidence in experimental_codes

    def test_is_experimental_property(self, go):
        annotations = go.get_annotations(HBA1_ACC, experimental_only=True)
        assert all(a.is_experimental for a in annotations)

    def test_evidence_code_present(self, go):
        annotations = go.get_annotations(HBA1_ACC)
        codes = {a.goEvidence for a in annotations if a.goEvidence}
        assert len(codes) > 0

    def test_assigned_by_present(self, go):
        annotations = go.get_annotations(HBA1_ACC)
        for a in annotations:
            if a.assignedBy:
                assert len(a.assignedBy) > 0

    def test_pmid_extraction(self, go):
        annotations = go.get_annotations(HBA1_ACC, experimental_only=True)
        pmid_refs = [a for a in annotations if a.pmid is not None]
        # Most experimental annotations should cite a paper
        assert len(pmid_refs) > 0
        for a in pmid_refs:
            assert a.pmid.isdigit()

    def test_aspect_short_property(self, go):
        annotations = go.get_annotations(HBA1_ACC)
        for a in annotations:
            short = a.aspect_short
            assert short in {"MF", "BP", "CC", "?"}


class TestGOTermDetails:
    def test_returns_go_term(self, go):
        term = go.get_term(OXYGEN_CARRIER_GO)
        assert isinstance(term, GOTerm)

    def test_term_id_correct(self, go):
        term = go.get_term(OXYGEN_CARRIER_GO)
        assert term.id == OXYGEN_CARRIER_GO

    def test_term_name_present(self, go):
        term = go.get_term(OXYGEN_CARRIER_GO)
        assert term.name is not None
        assert "oxygen" in term.name.lower() or "carrier" in term.name.lower()

    def test_term_aspect_is_mf(self, go):
        term = go.get_term(OXYGEN_CARRIER_GO)
        assert term.aspect == "molecular_function"
        assert term.namespace == "MF"

    def test_definition_present(self, go):
        term = go.get_term(OXYGEN_CARRIER_GO)
        assert term.definition is not None
        assert len(term.definition_text) > 10

    def test_is_not_obsolete(self, go):
        term = go.get_term(OXYGEN_CARRIER_GO)
        assert term.isObsolete is False

    def test_batch_terms(self, go):
        terms = go.get_terms_batch([OXYGEN_CARRIER_GO, KINASE_GO])
        assert len(terms) == 2
        ids = {t.id for t in terms}
        assert OXYGEN_CARRIER_GO in ids
        assert KINASE_GO in ids

    def test_raises_for_invalid_term(self, go):
        with pytest.raises((ValueError, Exception)):
            go.get_term("GO:9999999")

    def test_kinase_term_is_mf(self, go):
        term = go.get_term(KINASE_GO)
        assert term.aspect == "molecular_function"

    def test_apoptosis_term_is_bp(self, go):
        term = go.get_term(APOPTOSIS_GO)
        assert term.aspect == "biological_process"


class TestGOAncestors:
    def test_returns_ancestors(self, go):
        result = go.get_ancestors(KINASE_GO, relations=["is_a", "part_of"])
        assert isinstance(result, GOAncestors)
        assert result.term_id == KINASE_GO

    def test_ancestors_are_go_ids(self, go):
        result = go.get_ancestors(KINASE_GO, relations=["is_a"])
        for ancestor in result.ancestors:
            assert ancestor.startswith("GO:")

    def test_ancestors_include_parent_kinase(self, go):
        # GO:0004674 (Ser/Thr kinase) should have GO:0004672 (protein kinase) as ancestor
        result = go.get_ancestors(KINASE_GO, relations=["is_a"])
        assert "GO:0004672" in result.ancestors


class TestGOChildren:
    def test_returns_children(self, go):
        # GO:0004672 (protein kinase) has children
        result = go.get_children("GO:0004672")
        assert isinstance(result, GOChildren)
        assert result.term_id == "GO:0004672"

    def test_children_have_relations(self, go):
        result = go.get_children("GO:0004672")
        for child in result.children:
            assert "id" in child
            assert child["id"].startswith("GO:")

    def test_ser_thr_kinase_is_child(self, go):
        result = go.get_children("GO:0004672")
        child_ids = {c["id"] for c in result.children}
        assert KINASE_GO in child_ids


class TestGOSearch:
    def test_search_returns_terms(self, go):
        terms = go.search_terms("serine threonine kinase", limit=10)
        assert isinstance(terms, list)
        assert len(terms) > 0

    def test_search_results_are_go_terms(self, go):
        terms = go.search_terms("oxygen binding", limit=5)
        for t in terms:
            assert isinstance(t, GOTerm)
            assert t.id.startswith("GO:")
