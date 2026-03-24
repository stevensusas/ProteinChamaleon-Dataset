"""AlphaFold Protein Structure Database API client."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .base import BaseClient


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class AlphaFoldPrediction(BaseModel):
    entryId: str
    gene: Optional[str] = None
    uniprotAccession: str
    uniprotId: str
    uniprotDescription: Optional[str] = None
    taxId: Optional[int] = None
    organismScientificName: Optional[str] = None
    uniprotStart: Optional[int] = None
    uniprotEnd: Optional[int] = None
    uniprotSequence: Optional[str] = None
    modelCreatedDate: Optional[str] = None
    latestVersion: int
    allVersions: list[int] = Field(default_factory=list)
    isReviewed: Optional[bool] = None
    isReferenceProteome: Optional[bool] = None
    cifUrl: Optional[str] = None
    bcifUrl: Optional[str] = None
    pdbUrl: Optional[str] = None
    paeImageUrl: Optional[str] = None
    paeDocUrl: Optional[str] = None

    @property
    def fragment_number(self) -> int:
        """Extract fragment index from entryId (e.g., 'AF-P69905-F1' -> 1)."""
        for part in self.entryId.split("-"):
            if part.startswith("F") and part[1:].isdigit():
                return int(part[1:])
        return 1

    @property
    def sequence_length(self) -> Optional[int]:
        if self.uniprotStart is not None and self.uniprotEnd is not None:
            return self.uniprotEnd - self.uniprotStart + 1
        if self.uniprotSequence:
            return len(self.uniprotSequence)
        return None


class PAEData(BaseModel):
    """Predicted Aligned Error matrix data."""
    predicted_aligned_error: list[list[float]]
    max_predicted_aligned_error: float


# ── Client ────────────────────────────────────────────────────────────────────

class AlphaFoldClient(BaseClient):
    """Client for the AlphaFold Protein Structure Database API (EBI)."""

    API_URL = "https://alphafold.ebi.ac.uk/api"
    FILES_URL = "https://alphafold.ebi.ac.uk/files"

    def __init__(self, rate_limit_delay: float = 0.15, max_concurrency: int | None = None) -> None:
        super().__init__(self.API_URL, rate_limit_delay=rate_limit_delay, max_concurrency=max_concurrency)

    def get_prediction(self, uniprot_acc: str) -> list[AlphaFoldPrediction]:
        """
        Fetch AlphaFold prediction metadata for a UniProt accession.

        Returns a list — proteins >2700 aa are split into overlapping fragments
        (F1, F2, …), each a separate prediction object.
        """
        url = f"{self.API_URL}/prediction/{uniprot_acc}"
        resp = self._get(url)
        return [AlphaFoldPrediction.model_validate(p) for p in resp.json()]

    def get_pae(self, prediction: AlphaFoldPrediction) -> PAEData:
        """
        Download Predicted Aligned Error (PAE) data for a prediction.
        PAE is an N×N matrix of expected position errors in Ångströms.
        """
        url = (
            f"{self.FILES_URL}/{prediction.entryId}"
            f"-predicted_aligned_error_v{prediction.latestVersion}.json"
        )
        resp = self._get(url, headers={"Accept": "application/json"})
        data = resp.json()
        # API returns either a dict or a list with one element
        if isinstance(data, list):
            data = data[0]
        return PAEData.model_validate(data)

    def download_structure(
        self,
        prediction: AlphaFoldPrediction,
        fmt: str = "cif",
    ) -> bytes:
        """
        Download structure file for an AlphaFold prediction.
        fmt: 'cif' (mmCIF) | 'pdb' | 'bcif' (binary CIF)
        pLDDT per-residue confidence scores are stored in the B-factor column.
        """
        url = (
            f"{self.FILES_URL}/{prediction.entryId}"
            f"-model_v{prediction.latestVersion}.{fmt}"
        )
        resp = self._get(url, headers={"Accept": "*/*"})
        return resp.content

    def get_best_prediction(self, uniprot_acc: str) -> Optional[AlphaFoldPrediction]:
        """Return the first (or only) prediction fragment for a UniProt accession."""
        predictions = self.get_prediction(uniprot_acc)
        return predictions[0] if predictions else None
