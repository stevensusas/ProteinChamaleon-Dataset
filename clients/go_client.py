"""Gene Ontology API client — QuickGO REST API (EBI)."""

from __future__ import annotations

from typing import Any, Generator, Optional

from pydantic import BaseModel, Field, field_validator

from .base import BaseClient


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class GOAnnotation(BaseModel):
    id: Optional[str] = None
    geneProductId: str
    geneProductDb: Optional[str] = None
    qualifier: Optional[str] = None
    goId: str
    goName: Optional[str] = None
    goAspect: Optional[str] = None         # molecular_function | biological_process | cellular_component
    evidenceCode: Optional[str] = None     # ECO code, e.g., ECO:0000269
    goEvidence: Optional[str] = None       # Legacy code: IDA, IMP, IEA, EXP, etc.
    reference: Optional[str] = None        # PMID:xxxxx or GO_REF:xxxxx
    withFrom: Optional[Any] = None
    taxonId: Optional[int] = None
    taxonName: Optional[str] = None
    assignedBy: Optional[str] = None
    symbol: Optional[str] = None
    date: Optional[str] = None
    extensions: Optional[list[dict]] = Field(default_factory=list)

    @field_validator("geneProductId", mode="before")
    @classmethod
    def strip_db_prefix(cls, v: str) -> str:
        """QuickGO returns 'UniProtKB:P69905'; normalise to bare accession."""
        return v.split(":", 1)[-1] if ":" in v else v

    # Experimental evidence legacy codes
    _EXPERIMENTAL = {"IDA", "IMP", "IGI", "IPI", "IEP", "EXP", "HDA", "HMP", "HGI", "HEP", "HTP"}

    @property
    def is_experimental(self) -> bool:
        return (self.goEvidence or "") in self._EXPERIMENTAL

    @property
    def pmid(self) -> Optional[str]:
        """Extract PMID from reference field if present."""
        if self.reference and self.reference.startswith("PMID:"):
            return self.reference.replace("PMID:", "")
        return None

    @property
    def aspect_short(self) -> str:
        return {
            "molecular_function": "MF",
            "biological_process": "BP",
            "cellular_component": "CC",
        }.get(self.goAspect or "", "?")


class GOTermDefinition(BaseModel):
    text: str
    xrefs: list[dict] = Field(default_factory=list)


class GOTermSynonym(BaseModel):
    name: str
    type: str   # exact | broad | narrow | related


class GOTerm(BaseModel):
    id: str
    name: str
    definition: Optional[GOTermDefinition] = None
    aspect: Optional[str] = None    # molecular_function | biological_process | cellular_component
    isObsolete: bool = False
    synonyms: list[GOTermSynonym] = Field(default_factory=list)
    xRefs: list[dict] = Field(default_factory=list)
    comment: Optional[str] = None
    children: list[dict] = Field(default_factory=list)

    @property
    def namespace(self) -> str:
        return {
            "molecular_function": "MF",
            "biological_process": "BP",
            "cellular_component": "CC",
        }.get(self.aspect or "", "?")

    @property
    def definition_text(self) -> str:
        return self.definition.text if self.definition else ""


class GOAncestors(BaseModel):
    term_id: str
    ancestors: list[str]


class GOChildren(BaseModel):
    term_id: str
    children: list[dict]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_term(raw: dict) -> GOTerm:
    def_raw = raw.get("definition")
    definition = None
    if def_raw:
        if isinstance(def_raw, dict):
            definition = GOTermDefinition(text=def_raw.get("text", ""), xrefs=def_raw.get("xrefs", []))
        elif isinstance(def_raw, str):
            definition = GOTermDefinition(text=def_raw)

    synonyms = [
        GOTermSynonym(name=s.get("name", ""), type=s.get("type", ""))
        for s in raw.get("synonyms", [])
        if isinstance(s, dict)
    ]

    return GOTerm(
        id=raw.get("id", ""),
        name=raw.get("name", ""),
        definition=definition,
        aspect=raw.get("aspect"),
        isObsolete=raw.get("isObsolete", False),
        synonyms=synonyms,
        xRefs=raw.get("xRefs", []),
        comment=raw.get("comment"),
        children=raw.get("children", []),
    )


# ── Client ────────────────────────────────────────────────────────────────────

class GOClient(BaseClient):
    """
    Client for the QuickGO REST API (EBI) — Gene Ontology annotations and terms.

    Docs / Swagger: https://www.ebi.ac.uk/QuickGO/services/swagger-ui.html
    Rate limit: ~600 requests/minute (~10/sec). No API key required.
    """

    BASE_URL = "https://www.ebi.ac.uk/QuickGO/services"

    def __init__(self, rate_limit_delay: float = 0.12) -> None:
        super().__init__(self.BASE_URL, rate_limit_delay=rate_limit_delay)
        self.session.headers.update({"Accept": "application/json"})

    def _paginate_annotations(self, params: dict) -> Generator[GOAnnotation, None, None]:
        """Yield all annotations across paginated QuickGO results."""
        url = f"{self.BASE_URL}/annotation/search"
        params = {**params, "limit": 200}
        page = 1
        while True:
            params["page"] = page
            resp = self._get(url, params=params)
            data = resp.json()
            results = data.get("results", [])
            for item in results:
                yield GOAnnotation.model_validate(item)
            page_info = data.get("pageInfo", {})
            if page >= page_info.get("total", 1) or not results:
                break
            page += 1

    def get_annotations(
        self,
        uniprot_acc: str,
        aspect: Optional[str] = None,
        evidence_code: Optional[str] = None,
        experimental_only: bool = False,
    ) -> list[GOAnnotation]:
        """
        Get all GO annotations for a UniProt protein.

        aspect: 'molecular_function' | 'biological_process' | 'cellular_component'
        evidence_code: ECO code filter, e.g. 'ECO:0000269' (experimental)
        experimental_only: shortcut to filter for experimental evidence only
        """
        params: dict[str, Any] = {"geneProductId": uniprot_acc}
        if evidence_code:
            params["evidenceCode"] = evidence_code
            params["evidenceCodeUsage"] = "descendants"
        elif experimental_only:
            params["evidenceCode"] = "ECO:0000269"
            params["evidenceCodeUsage"] = "descendants"
        annotations = list(self._paginate_annotations(params))
        # QuickGO's goAspect query param is unreliable; filter client-side
        if aspect:
            annotations = [a for a in annotations if a.goAspect == aspect]
        return annotations

    def get_term(self, go_id: str) -> GOTerm:
        """Fetch full metadata for a single GO term (e.g., 'GO:0004674')."""
        url = f"{self.BASE_URL}/ontology/go/terms/{go_id}"
        resp = self._get(url)
        results = resp.json().get("results", [])
        if not results:
            raise ValueError(f"GO term not found: {go_id}")
        return _build_term(results[0])

    def get_terms_batch(self, go_ids: list[str]) -> list[GOTerm]:
        """Fetch multiple GO terms in a single request (comma-separated IDs)."""
        url = f"{self.BASE_URL}/ontology/go/terms/{','.join(go_ids)}"
        resp = self._get(url)
        return [_build_term(r) for r in resp.json().get("results", [])]

    def get_ancestors(
        self,
        go_id: str,
        relations: Optional[list[str]] = None,
    ) -> GOAncestors:
        """
        Get all ancestors of a GO term by following specified relation types.
        relations: e.g., ['is_a', 'part_of', 'regulates']
        """
        url = f"{self.BASE_URL}/ontology/go/terms/{go_id}/ancestors"
        params: dict[str, Any] = {}
        if relations:
            params["relations"] = ",".join(relations)
        resp = self._get(url, params=params)
        results = resp.json().get("results", [])
        return GOAncestors(
            term_id=go_id,
            ancestors=results[0].get("ancestors", []) if results else [],
        )

    def get_children(self, go_id: str) -> GOChildren:
        """Get direct child terms of a GO term."""
        url = f"{self.BASE_URL}/ontology/go/terms/{go_id}/children"
        resp = self._get(url)
        results = resp.json().get("results", [])
        return GOChildren(
            term_id=go_id,
            children=results[0].get("children", []) if results else [],
        )

    def search_terms(self, query: str, limit: int = 25) -> list[GOTerm]:
        """Search GO terms by keyword or phrase."""
        url = f"{self.BASE_URL}/ontology/go/search"
        resp = self._get(url, params={"query": query, "limit": limit, "page": 1})
        return [_build_term(r) for r in resp.json().get("results", [])]
