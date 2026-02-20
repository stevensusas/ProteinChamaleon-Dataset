"""STRING protein-protein interaction database API client (string-db.org)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .base import BaseClient


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class StringMapping(BaseModel):
    queryIndex: int
    stringId: str
    ncbiTaxonId: Optional[int] = None
    taxonName: Optional[str] = None
    preferredName: Optional[str] = None
    annotation: Optional[str] = None


class StringInteraction(BaseModel):
    stringId_A: str
    stringId_B: str
    preferredName_A: str
    preferredName_B: str
    ncbiTaxonId: Optional[int] = None
    score: float        # Combined confidence (0.0–1.0)
    nscore: float = 0.0  # Gene neighborhood
    fscore: float = 0.0  # Gene fusion
    pscore: float = 0.0  # Phylogenetic co-occurrence
    ascore: float = 0.0  # Co-expression
    escore: float = 0.0  # Experimental evidence
    dscore: float = 0.0  # Database / curated knowledge
    tscore: float = 0.0  # Text mining

    @property
    def confidence(self) -> float:
        """Combined confidence as a fraction (0.0–1.0)."""
        return self.score

    @property
    def is_high_confidence(self) -> bool:
        return self.score >= 0.7

    @property
    def dominant_source(self) -> str:
        """Return the name of the evidence channel with the highest sub-score."""
        channels = {
            "experimental": self.escore,
            "database": self.dscore,
            "text_mining": self.tscore,
            "co_expression": self.ascore,
            "neighborhood": self.nscore,
            "fusion": self.fscore,
            "cooccurrence": self.pscore,
        }
        return max(channels, key=lambda k: channels[k])


class StringEnrichment(BaseModel):
    category: str
    term: str
    number_of_genes: int
    number_of_genes_in_background: Optional[int] = None
    ncbiTaxonId: Optional[int] = None
    inputGenes: Optional[list[str]] = None
    preferredNames: Optional[list[str]] = None
    p_value: float
    fdr: float
    description: Optional[str] = None

    @property
    def gene_list(self) -> list[str]:
        return self.preferredNames or []


# ── Client ────────────────────────────────────────────────────────────────────

class STRINGClient(BaseClient):
    """
    Client for the STRING protein-protein interaction database API.

    Always include caller_identity — STRING uses this to whitelist programmatic
    clients and avoid rate-limit blocks.
    """

    BASE_URL = "https://string-db.org/api"
    DEFAULT_CALLER = "ProteinChameleonDataset"

    def __init__(
        self,
        rate_limit_delay: float = 1.0,
        caller_identity: Optional[str] = None,
    ) -> None:
        super().__init__(self.BASE_URL, rate_limit_delay=rate_limit_delay)
        self.caller_identity = caller_identity or self.DEFAULT_CALLER

    def _params(self, **extra) -> dict:
        return {"caller_identity": self.caller_identity, **extra}

    def map_identifiers(
        self,
        identifiers: list[str],
        taxon: int = 9606,
        limit: int = 1,
    ) -> list[StringMapping]:
        """
        Map gene names or UniProt accessions to STRING IDs.
        Pass multiple identifiers; STRING resolves them in one request.
        """
        resp = self._post(
            f"{self.BASE_URL}/json/get_string_ids",
            data=self._params(
                identifiers="\r\n".join(identifiers),
                species=taxon,
                limit=limit,
                echo_query=1,
            ),
        )
        return [StringMapping.model_validate(item) for item in resp.json()]

    def get_interaction_partners(
        self,
        identifiers: list[str],
        taxon: int = 9606,
        required_score: int = 400,
        limit: int = 50,
        network_type: str = "functional",
    ) -> list[StringInteraction]:
        """
        Get top interaction partners for one or more proteins.

        required_score thresholds:
            150 = low confidence
            400 = medium confidence (default)
            700 = high confidence
            900 = highest confidence
        """
        resp = self._post(
            f"{self.BASE_URL}/json/interaction_partners",
            data=self._params(
                identifiers="\r\n".join(identifiers),
                species=taxon,
                required_score=required_score,
                limit=limit,
                network_type=network_type,
            ),
        )
        return [StringInteraction.model_validate(item) for item in resp.json()]

    def get_network(
        self,
        identifiers: list[str],
        taxon: int = 9606,
        required_score: int = 400,
    ) -> list[StringInteraction]:
        """Get all pairwise interactions within a given set of proteins."""
        resp = self._post(
            f"{self.BASE_URL}/json/network",
            data=self._params(
                identifiers="\r\n".join(identifiers),
                species=taxon,
                required_score=required_score,
            ),
        )
        return [StringInteraction.model_validate(item) for item in resp.json()]

    def get_enrichment(
        self,
        identifiers: list[str],
        taxon: int = 9606,
    ) -> list[StringEnrichment]:
        """Functional enrichment analysis for a set of proteins."""
        resp = self._post(
            f"{self.BASE_URL}/json/enrichment",
            data=self._params(
                identifiers="\r\n".join(identifiers),
                species=taxon,
            ),
        )
        return [StringEnrichment.model_validate(item) for item in resp.json()]

    def get_version(self) -> dict:
        """Return the current STRING database version information."""
        resp = self._get(f"{self.BASE_URL}/json/version")
        return resp.json()
