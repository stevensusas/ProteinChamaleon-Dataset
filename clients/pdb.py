"""RCSB PDB Data API, Search API, GraphQL API, and file download client."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .base import BaseClient


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PolymerEntitySequence(BaseModel):
    pdbx_seq_one_letter_code: Optional[str] = None
    pdbx_seq_one_letter_code_can: Optional[str] = None
    type: Optional[str] = None
    pdbx_strand_id: Optional[str] = None
    rcsb_sample_sequence_length: Optional[int] = None


class PolymerEntityIdentifiers(BaseModel):
    entry_id: str
    entity_id: str
    auth_asym_ids: list[str] = Field(default_factory=list)
    asym_ids: list[str] = Field(default_factory=list)
    uniprot_ids: list[str] = Field(default_factory=list)


class SourceOrganism(BaseModel):
    scientific_name: Optional[str] = None
    taxid: Optional[int] = None
    pdbx_gene_src_gene: Optional[str] = None


class PolymerEntity(BaseModel):
    rcsb_id: str
    entity_poly: Optional[PolymerEntitySequence] = None
    rcsb_polymer_entity_container_identifiers: Optional[PolymerEntityIdentifiers] = None
    rcsb_entity_source_organism: list[SourceOrganism] = Field(default_factory=list)
    rcsb_polymer_entity: Optional[dict] = None

    @property
    def uniprot_ids(self) -> list[str]:
        if self.rcsb_polymer_entity_container_identifiers:
            return self.rcsb_polymer_entity_container_identifiers.uniprot_ids
        return []

    @property
    def sequence(self) -> Optional[str]:
        if self.entity_poly:
            return self.entity_poly.pdbx_seq_one_letter_code_can
        return None


class NonPolymerEntity(BaseModel):
    rcsb_id: str
    chem_comp: Optional[dict] = None
    pdbx_entity_nonpoly: Optional[dict] = None

    @property
    def ligand_id(self) -> Optional[str]:
        if self.pdbx_entity_nonpoly:
            return self.pdbx_entity_nonpoly.get("comp_id")
        return None

    @property
    def ligand_name(self) -> Optional[str]:
        if self.chem_comp:
            return self.chem_comp.get("name")
        return None


class EntryInfo(BaseModel):
    experimental_method: Optional[str] = None
    resolution_combined: Optional[list[float]] = None
    polymer_entity_count: Optional[int] = None
    nonpolymer_entity_count: Optional[int] = None
    deposited_atom_count: Optional[int] = None
    polymer_composition: Optional[str] = None
    structure_determination_methodology: Optional[str] = None


class PDBEntry(BaseModel):
    rcsb_id: str
    rcsb_entry_info: Optional[EntryInfo] = None
    struct: Optional[dict] = None
    rcsb_primary_citation: Optional[dict] = None
    audit_author: list[dict] = Field(default_factory=list)
    exptl: list[dict] = Field(default_factory=list)
    refine: list[dict] = Field(default_factory=list)

    @property
    def title(self) -> Optional[str]:
        return self.struct.get("title") if self.struct else None

    @property
    def resolution(self) -> Optional[float]:
        if self.rcsb_entry_info and self.rcsb_entry_info.resolution_combined:
            return self.rcsb_entry_info.resolution_combined[0]
        return None

    @property
    def experimental_method(self) -> Optional[str]:
        if self.rcsb_entry_info:
            return self.rcsb_entry_info.experimental_method
        return None


class PDBSearchResult(BaseModel):
    total_count: int
    identifiers: list[str]


# ── Client ────────────────────────────────────────────────────────────────────

class PDBClient(BaseClient):
    """Client for the RCSB PDB APIs (Data, Search, GraphQL) and file downloads."""

    DATA_URL = "https://data.rcsb.org/rest/v1"
    SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
    GRAPHQL_URL = "https://data.rcsb.org/graphql"
    FILES_URL = "https://files.rcsb.org"
    FASTA_URL = "https://www.rcsb.org/fasta/entry"

    def __init__(self, rate_limit_delay: float = 0.15, max_concurrency: int | None = None) -> None:
        super().__init__(self.DATA_URL, rate_limit_delay=rate_limit_delay, max_concurrency=max_concurrency)

    def get_entry(self, pdb_id: str) -> PDBEntry:
        """Fetch structured metadata for a PDB entry."""
        url = f"{self.DATA_URL}/core/entry/{pdb_id.upper()}"
        resp = self._get(url)
        return PDBEntry.model_validate(resp.json())

    def get_polymer_entity(self, pdb_id: str, entity_id: int | str) -> PolymerEntity:
        """Fetch a polymer entity (protein chain) by PDB ID and entity number."""
        url = f"{self.DATA_URL}/core/polymer_entity/{pdb_id.upper()}/{entity_id}"
        resp = self._get(url)
        return PolymerEntity.model_validate(resp.json())

    def get_nonpolymer_entity(self, pdb_id: str, entity_id: int | str) -> NonPolymerEntity:
        """Fetch a non-polymer entity (ligand) by PDB ID and entity number."""
        url = f"{self.DATA_URL}/core/nonpolymer_entity/{pdb_id.upper()}/{entity_id}"
        resp = self._get(url)
        return NonPolymerEntity.model_validate(resp.json())

    def get_all_polymer_entities(self, pdb_id: str) -> list[PolymerEntity]:
        """Fetch all polymer entities for a PDB entry."""
        entry = self.get_entry(pdb_id)
        count = (
            entry.rcsb_entry_info.polymer_entity_count
            if entry.rcsb_entry_info
            else 0
        ) or 0
        entities = []
        for i in range(1, count + 1):
            try:
                entities.append(self.get_polymer_entity(pdb_id, i))
            except Exception:
                break
        return entities

    def search_by_uniprot(self, uniprot_acc: str, max_rows: int = 100) -> PDBSearchResult:
        """Find all PDB entries linked to a UniProt accession (Search API v2)."""
        # v2 does not support text search on uniprot_ids; use reference_sequence_identifiers
        payload: dict[str, Any] = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
                            "operator": "exact_match",
                            "value": "UniProt",
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                            "operator": "exact_match",
                            "value": uniprot_acc.upper(),
                        },
                    },
                ],
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": max_rows},
                "sort": [{"sort_by": "score", "direction": "desc"}],
            },
        }
        resp = self._post(
            self.SEARCH_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        data = resp.json()
        return PDBSearchResult(
            total_count=data.get("total_count", 0),
            identifiers=[r["identifier"] for r in data.get("result_set", [])],
        )

    def graphql(self, query: str, variables: Optional[dict] = None) -> dict:
        """Execute a GraphQL query against the RCSB PDB GraphQL endpoint."""
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        resp = self._post(
            self.GRAPHQL_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        data = resp.json()
        if "errors" in data:
            raise ValueError(f"GraphQL errors: {data['errors']}")
        return data.get("data", {})

    def download_structure(self, pdb_id: str, fmt: str = "cif") -> bytes:
        """
        Download a structure file.
        fmt: 'cif' (mmCIF, recommended) | 'pdb' (legacy)
        """
        url = f"{self.FILES_URL}/download/{pdb_id.upper()}.{fmt}"
        resp = self._get(url, headers={"Accept": "*/*"})
        return resp.content

    def get_fasta(self, pdb_id: str) -> str:
        """Download FASTA sequences for all chains in a PDB entry."""
        url = f"{self.FASTA_URL}/{pdb_id.upper()}"
        resp = self._get(url, headers={"Accept": "text/plain"})
        return resp.text
