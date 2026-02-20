"""Integration tests for the KEGG REST API client."""

import pytest

from clients.kegg import (
    KEGGGeneEntry, KEGGPathwayEntry, KEGGEnzymeEntry,
    parse_kegg_flat_file,
)
from .conftest import AKT1_ACC, AKT1_KEGG, AKT1_EC


pytestmark = pytest.mark.integration

AKT1_PATHWAY = "hsa04151"   # PI3K-Akt signaling pathway


class TestKEGGConversion:
    def test_conv_uniprot_to_kegg(self, kegg):
        kegg_id = kegg.conv_uniprot_to_kegg(AKT1_ACC, org="hsa")
        assert kegg_id is not None
        assert kegg_id == AKT1_KEGG

    def test_conv_returns_none_for_unknown(self, kegg):
        kegg_id = kegg.conv_uniprot_to_kegg("INVALID000", org="hsa")
        assert kegg_id is None

    def test_conv_batch(self, kegg):
        mapping = kegg.conv_batch([AKT1_ACC], org="hsa")
        assert AKT1_ACC in mapping
        assert mapping[AKT1_ACC] == AKT1_KEGG


class TestKEGGGeneEntry:
    def test_returns_gene_entry(self, kegg):
        entry = kegg.get_gene_entry(AKT1_KEGG)
        assert isinstance(entry, KEGGGeneEntry)

    def test_kegg_id_correct(self, kegg):
        entry = kegg.get_gene_entry(AKT1_KEGG)
        assert entry.kegg_id == AKT1_KEGG

    def test_name_present(self, kegg):
        entry = kegg.get_gene_entry(AKT1_KEGG)
        assert entry.name is not None
        assert len(entry.name) > 0

    def test_organism_code_is_hsa(self, kegg):
        entry = kegg.get_gene_entry(AKT1_KEGG)
        assert entry.organism_code == "hsa"

    def test_has_pathways(self, kegg):
        entry = kegg.get_gene_entry(AKT1_KEGG)
        assert len(entry.pathways) > 0
        for p in entry.pathways:
            assert p.kegg_id.startswith("hsa")

    def test_has_ec_numbers(self, kegg):
        entry = kegg.get_gene_entry(AKT1_KEGG)
        assert len(entry.ec_numbers) > 0
        assert AKT1_EC in entry.ec_numbers

    def test_ko_present(self, kegg):
        entry = kegg.get_gene_entry(AKT1_KEGG)
        assert entry.ko is not None
        assert entry.ko.startswith("K")

    def test_db_links_present(self, kegg):
        entry = kegg.get_gene_entry(AKT1_KEGG)
        assert isinstance(entry.db_links, dict)
        assert len(entry.db_links) > 0

    def test_high_level_method(self, kegg):
        entry = kegg.get_entry_for_uniprot(AKT1_ACC)
        assert entry is not None
        assert isinstance(entry, KEGGGeneEntry)
        assert entry.kegg_id == AKT1_KEGG


class TestKEGGLinkQueries:
    def test_get_pathways_for_gene(self, kegg):
        pathways = kegg.get_pathways_for_gene(AKT1_KEGG)
        assert isinstance(pathways, list)
        assert len(pathways) > 0

    def test_get_ec_for_gene(self, kegg):
        ec_numbers = kegg.get_ec_for_gene(AKT1_KEGG)
        assert isinstance(ec_numbers, list)
        assert len(ec_numbers) > 0
        assert AKT1_EC in ec_numbers


class TestKEGGPathwayEntry:
    def test_returns_pathway_entry(self, kegg):
        entry = kegg.get_pathway_entry(AKT1_PATHWAY)
        assert isinstance(entry, KEGGPathwayEntry)

    def test_pathway_id_correct(self, kegg):
        entry = kegg.get_pathway_entry(AKT1_PATHWAY)
        assert entry.kegg_id == AKT1_PATHWAY

    def test_name_present(self, kegg):
        entry = kegg.get_pathway_entry(AKT1_PATHWAY)
        assert entry.name is not None
        assert len(entry.name) > 0

    def test_has_genes(self, kegg):
        entry = kegg.get_pathway_entry(AKT1_PATHWAY)
        assert len(entry.genes) > 0

    def test_path_prefix_stripped(self, kegg):
        # Test that 'path:' prefix is handled
        entry = kegg.get_pathway_entry(f"path:{AKT1_PATHWAY}")
        assert entry.kegg_id == AKT1_PATHWAY


class TestKEGGEnzymeEntry:
    def test_returns_enzyme_entry(self, kegg):
        entry = kegg.get_enzyme_entry(AKT1_EC)
        assert isinstance(entry, KEGGEnzymeEntry)

    def test_ec_number_correct(self, kegg):
        entry = kegg.get_enzyme_entry(AKT1_EC)
        assert entry.ec_number == AKT1_EC

    def test_name_present(self, kegg):
        entry = kegg.get_enzyme_entry(AKT1_EC)
        assert entry.name is not None
        assert len(entry.name) > 0

    def test_reaction_present(self, kegg):
        entry = kegg.get_enzyme_entry(AKT1_EC)
        # Should mention phosphorylation
        assert entry.reaction is not None

    def test_human_genes_in_entry(self, kegg):
        entry = kegg.get_enzyme_entry(AKT1_EC)
        assert "hsa" in entry.genes
        assert len(entry.genes["hsa"]) > 0

    def test_ec_prefix_stripped(self, kegg):
        entry = kegg.get_enzyme_entry(f"ec:{AKT1_EC}")
        assert entry.ec_number == AKT1_EC


class TestKEGGFlatFileParser:
    def test_parses_sections(self):
        raw = """ENTRY       207               CDS       T01001
NAME        AKT1, AKT
DEFINITION  AKT serine/threonine kinase 1
PATHWAY     hsa04010  MAPK signaling pathway
            hsa04150  mTOR signaling pathway
DBLINKS     UniProt: P31749
            NCBI-GeneID: 207
///"""
        sections = parse_kegg_flat_file(raw)
        assert "ENTRY" in sections
        assert "NAME" in sections
        assert "PATHWAY" in sections
        assert len(sections["PATHWAY"]) == 2
        assert "DBLINKS" in sections

    def test_handles_empty_raw(self):
        sections = parse_kegg_flat_file("///")
        assert sections == {}

    def test_handles_terminator(self):
        raw = "NAME        AKT1\n///\nEXTRA       should be ignored"
        sections = parse_kegg_flat_file(raw)
        assert "NAME" in sections
        assert "EXTRA" not in sections
