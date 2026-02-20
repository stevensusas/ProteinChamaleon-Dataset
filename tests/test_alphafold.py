"""Integration tests for the AlphaFold DB API client."""

import pytest

from clients.alphafold import AlphaFoldPrediction, PAEData
from .conftest import HBA1_ACC


pytestmark = pytest.mark.integration


class TestAlphaFoldGetPrediction:
    def test_returns_list(self, alphafold):
        preds = alphafold.get_prediction(HBA1_ACC)
        assert isinstance(preds, list)
        assert len(preds) >= 1

    def test_schema_validates(self, alphafold):
        preds = alphafold.get_prediction(HBA1_ACC)
        assert all(isinstance(p, AlphaFoldPrediction) for p in preds)

    def test_entry_id_format(self, alphafold):
        preds = alphafold.get_prediction(HBA1_ACC)
        pred = preds[0]
        # Format: AF-{accession}-F{fragment}
        assert pred.entryId.startswith(f"AF-{HBA1_ACC}-F")

    def test_uniprot_accession_matches(self, alphafold):
        preds = alphafold.get_prediction(HBA1_ACC)
        assert preds[0].uniprotAccession == HBA1_ACC

    def test_latest_version_is_positive_int(self, alphafold):
        preds = alphafold.get_prediction(HBA1_ACC)
        assert isinstance(preds[0].latestVersion, int)
        assert preds[0].latestVersion >= 1

    def test_structure_urls_present(self, alphafold):
        preds = alphafold.get_prediction(HBA1_ACC)
        pred = preds[0]
        assert pred.cifUrl is not None and pred.cifUrl.startswith("https://")
        assert pred.pdbUrl is not None and pred.pdbUrl.startswith("https://")

    def test_pae_url_present(self, alphafold):
        preds = alphafold.get_prediction(HBA1_ACC)
        assert preds[0].paeDocUrl is not None

    def test_organism_is_human(self, alphafold):
        preds = alphafold.get_prediction(HBA1_ACC)
        assert preds[0].taxId == 9606
        assert "sapiens" in (preds[0].organismScientificName or "").lower()

    def test_is_reviewed_true(self, alphafold):
        preds = alphafold.get_prediction(HBA1_ACC)
        assert preds[0].isReviewed is True

    def test_sequence_present(self, alphafold):
        preds = alphafold.get_prediction(HBA1_ACC)
        seq = preds[0].uniprotSequence
        assert seq is not None
        assert len(seq) > 0
        assert seq.isupper()

    def test_fragment_number_extraction(self, alphafold):
        preds = alphafold.get_prediction(HBA1_ACC)
        # HBA1 is small enough to be a single fragment
        assert preds[0].fragment_number == 1

    def test_get_best_prediction(self, alphafold):
        pred = alphafold.get_best_prediction(HBA1_ACC)
        assert pred is not None
        assert isinstance(pred, AlphaFoldPrediction)


class TestAlphaFoldPAE:
    def test_pae_returns_matrix(self, alphafold):
        pred = alphafold.get_best_prediction(HBA1_ACC)
        assert pred is not None
        pae = alphafold.get_pae(pred)
        assert isinstance(pae, PAEData)

    def test_pae_matrix_is_square(self, alphafold):
        pred = alphafold.get_best_prediction(HBA1_ACC)
        pae = alphafold.get_pae(pred)
        n = len(pae.predicted_aligned_error)
        assert n > 0
        assert all(len(row) == n for row in pae.predicted_aligned_error)

    def test_pae_values_in_valid_range(self, alphafold):
        pred = alphafold.get_best_prediction(HBA1_ACC)
        pae = alphafold.get_pae(pred)
        # PAE values should be >= 0 and <= max_predicted_aligned_error
        max_val = pae.max_predicted_aligned_error
        assert max_val > 0
        for row in pae.predicted_aligned_error:
            for val in row:
                assert 0 <= val <= max_val + 1  # small tolerance

    def test_pae_max_value_positive(self, alphafold):
        pred = alphafold.get_best_prediction(HBA1_ACC)
        pae = alphafold.get_pae(pred)
        assert pae.max_predicted_aligned_error > 0


class TestAlphaFoldDownload:
    def test_download_cif_structure(self, alphafold):
        pred = alphafold.get_best_prediction(HBA1_ACC)
        assert pred is not None
        content = alphafold.download_structure(pred, fmt="cif")
        assert isinstance(content, bytes)
        assert len(content) > 1000
        text_start = content[:200].decode("utf-8", errors="ignore")
        assert "data_" in text_start or "ATOM" in text_start or "_atom_site" in text_start

    def test_download_pdb_structure(self, alphafold):
        pred = alphafold.get_best_prediction(HBA1_ACC)
        content = alphafold.download_structure(pred, fmt="pdb")
        assert isinstance(content, bytes)
        text = content[:500].decode("utf-8", errors="ignore")
        assert "ATOM" in text or "HEADER" in text
