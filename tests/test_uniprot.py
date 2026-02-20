"""Integration tests for the UniProt REST API client."""

import pytest

from clients.uniprot import UniProtEntry, UniProtClient
from .conftest import HBA1_ACC, HBA1_GENE, TP53_ACC


pytestmark = pytest.mark.integration


class TestUniProtGetEntry:
    def test_returns_correct_accession(self, uniprot):
        entry = uniprot.get_entry(HBA1_ACC)
        assert entry.primaryAccession == HBA1_ACC

    def test_schema_validates(self, uniprot):
        entry = uniprot.get_entry(HBA1_ACC)
        assert isinstance(entry, UniProtEntry)

    def test_is_reviewed(self, uniprot):
        entry = uniprot.get_entry(HBA1_ACC)
        assert entry.is_reviewed is True

    def test_organism_is_human(self, uniprot):
        entry = uniprot.get_entry(HBA1_ACC)
        assert entry.organism.taxonId == 9606
        assert "sapiens" in entry.organism.scientificName.lower()

    def test_gene_name_present(self, uniprot):
        entry = uniprot.get_entry(HBA1_ACC)
        assert entry.gene_name is not None
        assert entry.gene_name.startswith("HBA")

    def test_sequence_fields(self, uniprot):
        entry = uniprot.get_entry(HBA1_ACC)
        assert entry.sequence.length == 142
        assert len(entry.sequence.value) == 142
        assert entry.sequence.molWeight > 0
        # Sequence should be all uppercase single-letter amino acids
        assert entry.sequence.value.isupper()
        assert all(c.isalpha() for c in entry.sequence.value)

    def test_protein_name_present(self, uniprot):
        entry = uniprot.get_entry(HBA1_ACC)
        assert entry.protein_name is not None
        assert len(entry.protein_name) > 0

    def test_has_pdb_cross_refs(self, uniprot):
        entry = uniprot.get_entry(HBA1_ACC)
        pdb_ids = entry.get_pdb_ids()
        assert len(pdb_ids) > 0
        assert "4HHB" in pdb_ids

    def test_has_go_cross_refs(self, uniprot):
        entry = uniprot.get_entry(HBA1_ACC)
        go_refs = entry.get_go_terms()
        assert len(go_refs) > 0
        go_ids = [r.id for r in go_refs]
        assert all(gid.startswith("GO:") for gid in go_ids)

    def test_has_interpro_cross_refs(self, uniprot):
        entry = uniprot.get_entry(HBA1_ACC)
        ipr_ids = entry.get_interpro_ids()
        assert len(ipr_ids) > 0
        assert all(iid.startswith("IPR") for iid in ipr_ids)

    def test_has_function_comment(self, uniprot):
        entry = uniprot.get_entry(HBA1_ACC)
        func = entry.get_function_text()
        assert func is not None
        assert len(func) > 20

    def test_secondary_accessions_is_list(self, uniprot):
        entry = uniprot.get_entry(HBA1_ACC)
        assert isinstance(entry.secondaryAccessions, list)

    def test_annotation_score_present(self, uniprot):
        entry = uniprot.get_entry(HBA1_ACC)
        assert entry.annotationScore is not None
        assert entry.annotationScore > 0

    def test_keywords_present(self, uniprot):
        entry = uniprot.get_entry(HBA1_ACC)
        assert len(entry.keywords) > 0
        for kw in entry.keywords:
            assert kw.id.startswith("KW-")
            assert len(kw.name) > 0

    def test_features_present(self, uniprot):
        entry = uniprot.get_entry(HBA1_ACC)
        assert len(entry.features) > 0
        for feat in entry.features:
            assert feat.type
            assert "start" in feat.location
            assert "end" in feat.location

    def test_tp53_has_string_ref(self, uniprot):
        entry = uniprot.get_entry(TP53_ACC)
        string_id = entry.get_string_id()
        assert string_id is not None
        assert "9606." in string_id  # human STRING ID format


class TestUniProtSearch:
    def test_search_returns_entries(self, uniprot):
        results = list(uniprot.search("gene:HBA1 AND organism_id:9606", size=5))
        assert len(results) > 0
        assert all(isinstance(e, UniProtEntry) for e in results)

    def test_search_reviewed_only(self, uniprot):
        results = list(uniprot.search("gene:HBA1 AND organism_id:9606", size=5, reviewed_only=True))
        assert len(results) > 0
        assert all(e.is_reviewed for e in results)

    def test_search_finds_hba1(self, uniprot):
        results = list(uniprot.search(f"accession:{HBA1_ACC}", size=1))
        assert len(results) == 1
        assert results[0].primaryAccession == HBA1_ACC


class TestUniProtFasta:
    def test_fasta_format(self, uniprot):
        fasta = uniprot.get_fasta(HBA1_ACC)
        assert fasta.startswith(">")
        assert HBA1_ACC in fasta
        # Should contain sequence lines
        lines = fasta.strip().splitlines()
        seq_lines = [l for l in lines if not l.startswith(">")]
        assert len(seq_lines) > 0
        full_seq = "".join(seq_lines)
        assert full_seq.isupper()
        assert len(full_seq) == 142
