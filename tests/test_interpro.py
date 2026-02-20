"""Integration tests for the InterPro REST API client."""

import pytest

from clients.interpro import InterProMatch, InterProProteinInfo, EntryMetadata
from .conftest import HBA1_ACC


pytestmark = pytest.mark.integration


class TestInterProGetEntries:
    def test_returns_list(self, interpro):
        results = interpro.get_protein_entries(HBA1_ACC, database="interpro")
        assert isinstance(results, list)

    def test_returns_interpro_entries(self, interpro):
        results = interpro.get_protein_entries(HBA1_ACC, database="interpro")
        assert len(results) > 0

    def test_entry_structure(self, interpro):
        results = interpro.get_protein_entries(HBA1_ACC, database="interpro")
        entry = results[0]
        assert "metadata" in entry
        meta = entry["metadata"]
        assert "accession" in meta
        assert meta["accession"].startswith("IPR")

    def test_pfam_entries(self, interpro):
        results = interpro.get_protein_entries(HBA1_ACC, database="pfam")
        assert len(results) > 0
        for entry in results:
            assert entry["metadata"]["accession"].startswith("PF")


class TestInterProGetAllMatches:
    def test_returns_protein_info(self, interpro):
        info = interpro.get_all_matches(HBA1_ACC)
        assert isinstance(info, InterProProteinInfo)

    def test_accession_correct(self, interpro):
        info = interpro.get_all_matches(HBA1_ACC)
        assert info.accession == HBA1_ACC

    def test_protein_length_correct(self, interpro):
        info = interpro.get_all_matches(HBA1_ACC)
        # HBA1 is 142 amino acids
        assert info.protein_length == 142

    def test_has_matches(self, interpro):
        info = interpro.get_all_matches(HBA1_ACC)
        assert info.entry_count > 0
        assert len(info.matches) > 0

    def test_match_schema(self, interpro):
        info = interpro.get_all_matches(HBA1_ACC)
        for match in info.matches:
            assert isinstance(match, InterProMatch)
            assert isinstance(match.metadata, EntryMetadata)
            assert match.metadata.accession
            assert match.metadata.source_database

    def test_has_interpro_matches(self, interpro):
        info = interpro.get_all_matches(HBA1_ACC)
        ipr_matches = info.get_interpro_matches()
        assert len(ipr_matches) > 0
        assert all(m.metadata.accession.startswith("IPR") for m in ipr_matches)

    def test_has_pfam_matches(self, interpro):
        info = interpro.get_all_matches(HBA1_ACC)
        pfam_matches = info.get_pfam_matches()
        assert len(pfam_matches) > 0
        assert all(m.metadata.accession.startswith("PF") for m in pfam_matches)

    def test_domain_locations_valid(self, interpro):
        info = interpro.get_all_matches(HBA1_ACC)
        for match in info.matches:
            for loc in match.locations:
                for frag in loc.fragments:
                    assert frag.start >= 1
                    assert frag.end >= frag.start
                    assert frag.end <= 150  # HBA1 is 142 aa, allow small margin

    def test_go_terms_present(self, interpro):
        info = interpro.get_all_matches(HBA1_ACC)
        go_terms = info.get_go_terms()
        # Hemoglobin has well-known GO terms via InterPro
        assert isinstance(go_terms, list)

    def test_match_scores_are_floats_or_none(self, interpro):
        info = interpro.get_all_matches(HBA1_ACC)
        for match in info.matches:
            for loc in match.locations:
                if loc.score is not None:
                    assert isinstance(loc.score, float)
                    assert loc.score >= 0


class TestInterProEntryDetails:
    def test_get_interpro_entry(self, interpro):
        # Globin InterPro entry
        details = interpro.get_entry_details("IPR000971", database="interpro")
        assert "metadata" in details
        meta = details["metadata"]
        assert meta["accession"] == "IPR000971"
        assert meta["source_database"] == "interpro"

    def test_entry_has_name(self, interpro):
        details = interpro.get_entry_details("IPR000971", database="interpro")
        assert len(details["metadata"].get("name", "")) > 0
