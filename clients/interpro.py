"""InterPro REST API client (www.ebi.ac.uk/interpro/api)."""

from __future__ import annotations

from typing import Any, Generator, Optional

from pydantic import BaseModel, Field

from .base import BaseClient


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class DomainFragment(BaseModel):
    start: int
    end: int
    dc_status: Optional[str] = Field(None, alias="dc-status")

    model_config = {"populate_by_name": True}

    @property
    def length(self) -> int:
        return self.end - self.start + 1


class EntryProteinLocation(BaseModel):
    fragments: list[DomainFragment] = Field(default_factory=list)
    model: Optional[str] = None
    score: Optional[float] = None
    sequence_feature: Optional[str] = None


class EntryMetadata(BaseModel):
    accession: str
    name: str
    source_database: str
    type: Optional[str] = None
    integrated: Optional[str] = None       # parent InterPro accession (if member DB entry)
    member_databases: Optional[dict] = None
    go_terms: list[dict] = Field(default_factory=list)


class InterProMatch(BaseModel):
    metadata: EntryMetadata
    protein_length: Optional[int] = None
    locations: list[EntryProteinLocation] = Field(default_factory=list)

    @property
    def is_interpro(self) -> bool:
        return self.metadata.source_database == "interpro"

    @property
    def entry_type(self) -> Optional[str]:
        return self.metadata.type

    @property
    def all_positions(self) -> list[tuple[int, int]]:
        """Return all (start, end) fragment positions across all locations."""
        positions = []
        for loc in self.locations:
            for frag in loc.fragments:
                positions.append((frag.start, frag.end))
        return positions


class InterProProteinInfo(BaseModel):
    accession: str
    source_database: str = "uniprot"
    protein_length: Optional[int] = None
    entry_count: int = 0
    matches: list[InterProMatch] = Field(default_factory=list)

    def get_matches_by_db(self, database: str) -> list[InterProMatch]:
        return [m for m in self.matches if m.metadata.source_database == database]

    def get_interpro_matches(self) -> list[InterProMatch]:
        return self.get_matches_by_db("interpro")

    def get_pfam_matches(self) -> list[InterProMatch]:
        return self.get_matches_by_db("pfam")

    def get_go_terms(self) -> list[dict]:
        """Collect all GO terms from all InterPro matches."""
        go_terms: dict[str, dict] = {}
        for m in self.matches:
            for go in m.metadata.go_terms:
                go_id = go.get("identifier", "")
                if go_id:
                    go_terms[go_id] = go
        return list(go_terms.values())


# ── Client ────────────────────────────────────────────────────────────────────

MEMBER_DATABASES = [
    "pfam", "smart", "prosite", "prints", "panther",
    "cathgene3d", "superfamily", "hamap", "tigrfam", "pirsf", "sfld", "ncbifam",
]


class InterProClient(BaseClient):
    """Client for the InterPro REST API."""

    BASE_URL = "https://www.ebi.ac.uk/interpro/api"

    def __init__(self, rate_limit_delay: float = 0.15, max_concurrency: int | None = None) -> None:
        super().__init__(self.BASE_URL, rate_limit_delay=rate_limit_delay, max_concurrency=max_concurrency)

    def _paginate(self, url: str, params: Optional[dict] = None) -> Generator[dict, None, None]:
        """Iterate through all pages of a paginated InterPro response."""
        params = {**(params or {}), "format": "json", "page_size": 200}
        current_url: Optional[str] = url
        while current_url:
            resp = self._get(current_url, params=params)
            data = resp.json()
            yield from data.get("results", [])
            current_url = data.get("next")
            params = {}  # cursor is embedded in next URL

    def get_protein_entries(
        self,
        uniprot_acc: str,
        database: str = "interpro",
    ) -> list[dict]:
        """
        Fetch all entries matching a protein from one database (or 'all').
        database: 'interpro' | 'all' | 'pfam' | 'smart' | 'prosite' | …
        """
        url = f"{self.BASE_URL}/entry/{database}/protein/uniprot/{uniprot_acc}/"
        return list(self._paginate(url))

    # Databases to query for get_all_matches (entry/all often returns empty or different structure)
    PROTEIN_ENTRY_DATABASES = ["interpro", "pfam", "smart", "prosite"]

    def get_all_matches(self, uniprot_acc: str) -> InterProProteinInfo:
        """
        Get comprehensive domain/motif annotation for a protein.
        Fetches from InterPro, Pfam, SMART, PROSITE and merges (avoids unreliable entry/all).
        """
        raw_entries: list[dict] = []
        seen: set[str] = set()
        for db in self.PROTEIN_ENTRY_DATABASES:
            try:
                for entry in self.get_protein_entries(uniprot_acc, database=db):
                    acc = (entry.get("metadata") or {}).get("accession", "")
                    if acc and acc not in seen:
                        seen.add(acc)
                        raw_entries.append(entry)
            except Exception:
                continue
        matches: list[InterProMatch] = []
        protein_length: Optional[int] = None

        for entry in raw_entries:
            meta_raw = entry.get("metadata", {})
            meta = EntryMetadata(
                accession=meta_raw.get("accession", ""),
                name=meta_raw.get("name", ""),
                source_database=meta_raw.get("source_database", ""),
                type=meta_raw.get("type"),
                integrated=meta_raw.get("integrated"),
                member_databases=meta_raw.get("member_databases"),
                go_terms=meta_raw.get("go_terms") or [],
            )

            proteins_raw = entry.get("proteins", [])
            if not isinstance(proteins_raw, list):
                proteins_raw = [proteins_raw] if isinstance(proteins_raw, dict) else []

            # Use first protein match that has locations; fallback to entry-level locations
            raw_locations: list[dict] = []
            protein_length_from_entry: Optional[int] = None
            for p in proteins_raw:
                if not isinstance(p, dict):
                    continue
                # Accept nested structure (e.g. {"accession": "...", "entry_protein_locations": [...]})
                protein_length_from_entry = p.get("protein_length", protein_length_from_entry)
                locs = p.get("entry_protein_locations", [])
                if locs:
                    raw_locations = locs
                    if protein_length_from_entry is not None:
                        protein_length = protein_length_from_entry
                    break
            if not raw_locations:
                raw_locations = entry.get("entry_protein_locations", [])

            if protein_length_from_entry is not None:
                protein_length = protein_length_from_entry

            locations = []
            for loc in raw_locations:
                frags_raw = loc.get("fragments", [])
                # Some API responses put start/end on the location itself
                if not frags_raw and isinstance(loc, dict):
                    start, end = loc.get("start"), loc.get("end")
                    if start is None and "begin" in loc:
                        start = loc["begin"]
                    if start is not None and end is not None:
                        frags_raw = [{"start": start, "end": end}]
                frags = []
                for f in frags_raw:
                    if not isinstance(f, dict):
                        continue
                    f = dict(f)
                    if "start" not in f and "begin" in f:
                        f["start"] = f["begin"]
                    try:
                        frags.append(DomainFragment.model_validate(f))
                    except Exception:
                        continue
                locations.append(
                    EntryProteinLocation(
                        fragments=frags,
                        model=loc.get("model"),
                        score=loc.get("score"),
                        sequence_feature=loc.get("sequence_feature"),
                    )
                )
            if locations or meta.accession:
                matches.append(
                    InterProMatch(
                        metadata=meta,
                        protein_length=protein_length,
                        locations=locations,
                    )
                )

        return InterProProteinInfo(
            accession=uniprot_acc,
            protein_length=protein_length,
            entry_count=len(matches),
            matches=matches,
        )

    def get_entry_details(
        self, accession: str, database: str = "interpro"
    ) -> dict:
        """Fetch full metadata for a specific InterPro or member-DB entry."""
        url = f"{self.BASE_URL}/entry/{database}/{accession}/"
        resp = self._get(url, params={"format": "json"})
        return resp.json()

    def get_protein_structures(self, uniprot_acc: str) -> list[dict]:
        """Fetch PDB structures linked to a protein via InterPro cross-references."""
        url = f"{self.BASE_URL}/structure/pdb/protein/uniprot/{uniprot_acc}/"
        return list(self._paginate(url))
