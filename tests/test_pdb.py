"""Integration tests for the RCSB PDB API client."""

import pytest

from clients.pdb import PDBEntry, PolymerEntity, NonPolymerEntity, PDBSearchResult
from .conftest import HBA1_ACC, HBA1_PDB


pytestmark = pytest.mark.integration


class TestPDBGetEntry:
    def test_returns_correct_id(self, pdb):
        entry = pdb.get_entry(HBA1_PDB)
        assert entry.rcsb_id == HBA1_PDB

    def test_schema_validates(self, pdb):
        entry = pdb.get_entry(HBA1_PDB)
        assert isinstance(entry, PDBEntry)

    def test_title_present(self, pdb):
        entry = pdb.get_entry(HBA1_PDB)
        assert entry.title is not None
        assert "hemoglobin" in entry.title.lower() or "HAEMOGLOBIN" in entry.title.upper()

    def test_experimental_method(self, pdb):
        entry = pdb.get_entry(HBA1_PDB)
        assert entry.experimental_method is not None
        assert "X-RAY" in entry.experimental_method.upper()

    def test_resolution_is_float(self, pdb):
        entry = pdb.get_entry(HBA1_PDB)
        assert entry.resolution is not None
        assert isinstance(entry.resolution, float)
        assert 1.0 < entry.resolution < 3.0  # 4HHB is 1.74 Å

    def test_entry_info_present(self, pdb):
        entry = pdb.get_entry(HBA1_PDB)
        assert entry.rcsb_entry_info is not None
        assert entry.rcsb_entry_info.polymer_entity_count is not None
        assert entry.rcsb_entry_info.polymer_entity_count >= 1

    def test_lowercase_id_normalized(self, pdb):
        entry = pdb.get_entry("4hhb")
        assert entry.rcsb_id == HBA1_PDB


class TestPDBPolymerEntity:
    def test_returns_polymer_entity(self, pdb):
        entity = pdb.get_polymer_entity(HBA1_PDB, 1)
        assert isinstance(entity, PolymerEntity)

    def test_entity_has_sequence(self, pdb):
        entity = pdb.get_polymer_entity(HBA1_PDB, 1)
        seq = entity.sequence
        assert seq is not None
        assert len(seq) > 0
        assert seq.isupper()

    def test_entity_has_uniprot_link(self, pdb):
        entity = pdb.get_polymer_entity(HBA1_PDB, 1)
        uniprot_ids = entity.uniprot_ids
        assert len(uniprot_ids) > 0
        # Should link back to hemoglobin alpha or beta
        assert any("P" in uid for uid in uniprot_ids)

    def test_entity_has_identifiers(self, pdb):
        entity = pdb.get_polymer_entity(HBA1_PDB, 1)
        ids = entity.rcsb_polymer_entity_container_identifiers
        assert ids is not None
        assert ids.entry_id == HBA1_PDB
        assert ids.entity_id == "1"

    def test_get_all_polymer_entities(self, pdb):
        entities = pdb.get_all_polymer_entities(HBA1_PDB)
        assert len(entities) >= 2  # hemoglobin has alpha + beta chains
        assert all(isinstance(e, PolymerEntity) for e in entities)


class TestPDBSearchByUniprot:
    def test_returns_search_result(self, pdb):
        result = pdb.search_by_uniprot(HBA1_ACC)
        assert isinstance(result, PDBSearchResult)

    def test_finds_known_structure(self, pdb):
        result = pdb.search_by_uniprot(HBA1_ACC, max_rows=400)
        assert result.total_count > 0
        assert HBA1_PDB in result.identifiers

    def test_identifiers_are_uppercase_4chars(self, pdb):
        result = pdb.search_by_uniprot(HBA1_ACC)
        for pdb_id in result.identifiers[:10]:
            assert len(pdb_id) == 4
            assert pdb_id.isupper()


class TestPDBGraphQL:
    def test_graphql_entry_query(self, pdb):
        query = """
        {
          entry(entry_id: "4HHB") {
            rcsb_id
            struct {
              title
            }
            rcsb_entry_info {
              resolution_combined
              experimental_method
            }
          }
        }
        """
        data = pdb.graphql(query)
        assert "entry" in data
        entry = data["entry"]
        assert entry["rcsb_id"] == "4HHB"
        assert entry["struct"]["title"] is not None
        assert entry["rcsb_entry_info"]["experimental_method"] is not None

    def test_graphql_multi_entry_query(self, pdb):
        query = """
        {
          entries(entry_ids: ["4HHB", "1A00"]) {
            rcsb_id
            rcsb_entry_info {
              resolution_combined
            }
          }
        }
        """
        data = pdb.graphql(query)
        assert "entries" in data
        assert len(data["entries"]) == 2


class TestPDBDownload:
    def test_download_cif_structure(self, pdb):
        content = pdb.download_structure(HBA1_PDB, fmt="cif")
        assert isinstance(content, bytes)
        assert len(content) > 1000
        # mmCIF files start with 'data_' or '#'
        text_start = content[:200].decode("utf-8", errors="ignore")
        assert "data_" in text_start or "#" in text_start

    def test_download_pdb_structure(self, pdb):
        content = pdb.download_structure(HBA1_PDB, fmt="pdb")
        assert isinstance(content, bytes)
        assert len(content) > 1000
        text_start = content[:200].decode("utf-8", errors="ignore")
        assert "ATOM" in text_start or "HEADER" in text_start

    def test_get_fasta(self, pdb):
        fasta = pdb.get_fasta(HBA1_PDB)
        assert fasta.startswith(">")
        assert HBA1_PDB in fasta
