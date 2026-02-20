"""UniProt REST API client (rest.uniprot.org)."""

from __future__ import annotations

import re
from typing import Any, Generator, Optional

from pydantic import BaseModel, Field

from .base import BaseClient


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class Organism(BaseModel):
    scientificName: str
    commonName: Optional[str] = None
    taxonId: int
    lineage: list[str] = Field(default_factory=list)


class Sequence(BaseModel):
    value: str
    length: int
    molWeight: int
    crc64: str = ""
    md5: str = ""


class LocalizedName(BaseModel):
    value: str
    evidences: list[dict] = Field(default_factory=list)


class RecommendedName(BaseModel):
    fullName: Optional[LocalizedName] = None
    shortNames: list[LocalizedName] = Field(default_factory=list)
    ecNumbers: list[LocalizedName] = Field(default_factory=list)


class ProteinDescription(BaseModel):
    recommendedName: Optional[RecommendedName] = None
    alternativeNames: list[dict] = Field(default_factory=list)
    submissionNames: list[dict] = Field(default_factory=list)


class Gene(BaseModel):
    geneName: Optional[dict] = None
    synonyms: list[dict] = Field(default_factory=list)
    orderedLocusNames: list[dict] = Field(default_factory=list)
    orfNames: list[dict] = Field(default_factory=list)


class Comment(BaseModel):
    commentType: str
    texts: list[dict] = Field(default_factory=list)
    subcellularLocations: list[dict] = Field(default_factory=list)
    disease: Optional[dict] = None
    reaction: Optional[dict] = None
    note: Optional[Any] = None


class Feature(BaseModel):
    type: str
    location: dict
    description: str = ""
    featureId: Optional[str] = None
    alternativeSequence: Optional[dict] = None
    evidences: list[dict] = Field(default_factory=list)


class DbReference(BaseModel):
    """
    Unified cross-reference model.
    UniProt REST returns: {database, id, properties: [{key, value}, ...]}
    We normalise properties to a plain dict {key: value}.
    """
    type: str = Field(alias="database", default="")
    id: str
    properties: dict = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    @classmethod
    def from_api(cls, raw: dict) -> "DbReference":
        props_raw = raw.get("properties", [])
        props: dict = {}
        if isinstance(props_raw, list):
            for item in props_raw:
                if isinstance(item, dict) and "key" in item:
                    props[item["key"]] = item.get("value", "")
        elif isinstance(props_raw, dict):
            props = props_raw
        return cls(
            database=raw.get("database", raw.get("type", "")),
            id=raw.get("id", ""),
            properties=props,
        )


class Keyword(BaseModel):
    id: str
    name: str
    category: Optional[str] = None


class UniProtEntry(BaseModel):
    primaryAccession: str
    secondaryAccessions: list[str] = Field(default_factory=list)
    uniProtkbId: str
    entryType: str
    annotationScore: Optional[float] = None
    organism: Organism
    sequence: Sequence
    proteinDescription: Optional[ProteinDescription] = None
    genes: list[Gene] = Field(default_factory=list)
    comments: list[Comment] = Field(default_factory=list)
    features: list[Feature] = Field(default_factory=list)
    # API field is 'uniProtKBCrossReferences'; we normalise in validator below
    dbReferences: list[DbReference] = Field(default_factory=list)
    keywords: list[Keyword] = Field(default_factory=list)

    @classmethod
    def model_validate(cls, obj: Any, **kwargs) -> "UniProtEntry":  # type: ignore[override]
        if isinstance(obj, dict):
            # Normalise cross-references from new API field name
            raw_refs = obj.pop("uniProtKBCrossReferences", obj.get("dbReferences", []))
            obj["dbReferences"] = [
                DbReference.from_api(r) if isinstance(r, dict) else r
                for r in raw_refs
            ]
        return super().model_validate(obj, **kwargs)

    # ── convenience ───────────────────────────────────────────────────────────

    @property
    def is_reviewed(self) -> bool:
        return "Swiss-Prot" in self.entryType or "reviewed" in self.entryType.lower()

    @property
    def gene_name(self) -> Optional[str]:
        if self.genes and self.genes[0].geneName:
            return self.genes[0].geneName.get("value")
        return None

    @property
    def protein_name(self) -> Optional[str]:
        if self.proteinDescription and self.proteinDescription.recommendedName:
            fn = self.proteinDescription.recommendedName.fullName
            return fn.value if fn else None
        return None

    def get_db_refs(self, db_type: str) -> list[DbReference]:
        return [r for r in self.dbReferences if r.type == db_type]

    def get_pdb_ids(self) -> list[str]:
        return [r.id for r in self.get_db_refs("PDB")]

    def get_alphafold_ids(self) -> list[str]:
        """AlphaFold DB cross-references (UniProt may use 'AlphaFold' or 'AlphaFoldDatabase')."""
        ids: list[str] = []
        for db in ("AlphaFold", "AlphaFoldDatabase"):
            ids.extend(r.id for r in self.get_db_refs(db))
        return ids

    def get_go_terms(self) -> list[DbReference]:
        return self.get_db_refs("GO")

    def get_interpro_ids(self) -> list[str]:
        return [r.id for r in self.get_db_refs("InterPro")]

    def get_string_id(self) -> Optional[str]:
        refs = self.get_db_refs("STRING")
        return refs[0].id if refs else None

    def get_kegg_ids(self) -> list[str]:
        return [r.id for r in self.get_db_refs("KEGG")]

    def get_function_text(self) -> Optional[str]:
        for c in self.comments:
            if c.commentType == "FUNCTION" and c.texts:
                return c.texts[0].get("value", "")
        return None

    def get_ec_numbers(self) -> list[str]:
        ec_numbers: list[str] = []
        if self.proteinDescription and self.proteinDescription.recommendedName:
            for ec in self.proteinDescription.recommendedName.ecNumbers:
                ec_numbers.append(ec.value)
        return ec_numbers


# ── Client ────────────────────────────────────────────────────────────────────

class UniProtClient(BaseClient):
    """Client for the UniProt REST API (rest.uniprot.org)."""

    BASE_URL = "https://rest.uniprot.org"

    def __init__(self, rate_limit_delay: float = 0.5) -> None:
        super().__init__(self.BASE_URL, rate_limit_delay=rate_limit_delay)

    def get_entry(self, accession: str) -> UniProtEntry:
        """Fetch a single UniProtKB entry by accession (e.g., 'P69905')."""
        url = f"{self.BASE_URL}/uniprotkb/{accession}.json"
        resp = self._get(url)
        return UniProtEntry.model_validate(resp.json())

    def get_isoforms(self, accession: str) -> list[str]:
        """Return isoform accession strings for an entry (e.g., ['P69905-1', 'P69905-2'])."""
        url = f"{self.BASE_URL}/uniprotkb/{accession}/isoforms"
        resp = self._get(url, headers={"Accept": "application/json"})
        data = resp.json()
        if isinstance(data, list):
            return data
        return [r.get("primaryAccession", "") for r in data.get("results", [])]

    def search(
        self,
        query: str,
        fields: Optional[list[str]] = None,
        size: int = 25,
        reviewed_only: bool = False,
    ) -> Generator[UniProtEntry, None, None]:
        """
        Search UniProtKB. Yields UniProtEntry objects, paginating automatically.

        Example queries:
            'organism_id:9606 AND reviewed:true'
            'gene:TP53 AND organism_id:9606'
        """
        if reviewed_only:
            query = f"({query}) AND reviewed:true"
        params: dict[str, Any] = {"query": query, "format": "json", "size": size}
        if fields:
            params["fields"] = ",".join(fields)

        url = f"{self.BASE_URL}/uniprotkb/search"
        while url:
            resp = self._get(url, params=params)
            data = resp.json()
            for item in data.get("results", []):
                yield UniProtEntry.model_validate(item)

            link_header = resp.headers.get("Link", "")
            if 'rel="next"' in link_header:
                match = re.search(r"<([^>]+)>", link_header)
                url = match.group(1) if match else None
                # Keep query (and format/size) so the server receives required params on next request
                if url:
                    params = {"query": query, "format": "json", "size": size}
                    if fields:
                        params["fields"] = ",".join(fields)
            else:
                url = None

    def get_fasta(self, accession: str, include_isoforms: bool = False) -> str:
        """Download FASTA sequence(s) for an accession."""
        url = f"{self.BASE_URL}/uniprotkb/{accession}.fasta"
        params = {"includeIsoform": "true"} if include_isoforms else {}
        resp = self._get(url, params=params, headers={"Accept": "text/plain"})
        return resp.text
