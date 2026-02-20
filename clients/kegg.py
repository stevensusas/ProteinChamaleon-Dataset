"""KEGG REST API client (rest.kegg.jp)."""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

from .base import BaseClient


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class KEGGPathwayRef(BaseModel):
    kegg_id: str    # e.g., 'hsa04010' or 'path:hsa04010'
    name: str       # e.g., 'MAPK signaling pathway'


class KEGGModuleRef(BaseModel):
    kegg_id: str
    name: str


class KEGGGeneEntry(BaseModel):
    kegg_id: str
    name: Optional[str] = None
    definition: Optional[str] = None
    ko: Optional[str] = None          # KO accession, e.g., 'K04456'
    ko_definition: Optional[str] = None
    organism: Optional[str] = None    # Scientific name
    organism_code: Optional[str] = None  # 3-letter code, e.g., 'hsa'
    pathways: list[KEGGPathwayRef] = Field(default_factory=list)
    modules: list[KEGGModuleRef] = Field(default_factory=list)
    ec_numbers: list[str] = Field(default_factory=list)
    diseases: list[str] = Field(default_factory=list)
    db_links: dict[str, list[str]] = Field(default_factory=dict)
    pdb_ids: list[str] = Field(default_factory=list)

    @property
    def pathway_ids(self) -> list[str]:
        return [p.kegg_id for p in self.pathways]

    @property
    def uniprot_acc(self) -> Optional[str]:
        ids = self.db_links.get("UniProt", self.db_links.get("Uniprot", []))
        return ids[0] if ids else None


class KEGGPathwayEntry(BaseModel):
    kegg_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    pathway_class: Optional[str] = None
    genes: list[str] = Field(default_factory=list)   # KEGG gene IDs in this pathway
    compounds: list[str] = Field(default_factory=list)
    ko_pathway: Optional[str] = None
    db_links: dict[str, list[str]] = Field(default_factory=dict)


class KEGGEnzymeEntry(BaseModel):
    ec_number: str
    name: Optional[str] = None
    ec_class: Optional[str] = None
    sysname: Optional[str] = None
    reaction: Optional[str] = None
    substrates: list[str] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)
    genes: dict[str, list[str]] = Field(default_factory=dict)  # org_code -> [gene_ids]
    db_links: dict[str, list[str]] = Field(default_factory=dict)


# ── Flat-file parser ──────────────────────────────────────────────────────────

def parse_kegg_flat_file(raw: str) -> dict[str, list[str]]:
    """
    Parse a KEGG flat-file entry into section_name -> [content_lines].

    KEGG flat files:
    - Section names appear left-justified (first ~12 chars)
    - Continuation lines are indented
    - Each entry is terminated by '///'
    """
    sections: dict[str, list[str]] = {}
    current: Optional[str] = None

    for line in raw.splitlines():
        if line.startswith("///"):
            break
        if line and not line[0].isspace():
            parts = line.split(None, 1)
            current = parts[0]
            sections.setdefault(current, [])
            if len(parts) > 1:
                sections[current].append(parts[1].strip())
        elif current and line.strip():
            sections[current].append(line.strip())

    return sections


def _parse_db_links(sections: dict) -> dict[str, list[str]]:
    db_links: dict[str, list[str]] = {}
    for line in sections.get("DBLINKS", []):
        if ":" in line:
            db, ids_str = line.split(":", 1)
            db_links[db.strip()] = ids_str.split()
    return db_links


def _extract_ec_numbers(text: str) -> list[str]:
    return re.findall(r"EC:([\d\.\-]+)", text) + re.findall(r"\[EC:([\d\.\-]+)\]", text)


def _parse_gene_entry(sections: dict, kegg_id: str) -> KEGGGeneEntry:
    pathways = []
    for line in sections.get("PATHWAY", []):
        parts = line.split(None, 1)
        if len(parts) == 2:
            pathways.append(KEGGPathwayRef(kegg_id=parts[0], name=parts[1]))

    modules = []
    for line in sections.get("MODULE", []):
        parts = line.split(None, 1)
        if len(parts) == 2:
            modules.append(KEGGModuleRef(kegg_id=parts[0], name=parts[1]))

    # EC numbers from ORTHOLOGY and DEFINITION
    ec_set: list[str] = []
    for ec in _extract_ec_numbers(" ".join(sections.get("ORTHOLOGY", []))):
        if ec not in ec_set:
            ec_set.append(ec)
    for ec in _extract_ec_numbers(" ".join(sections.get("DEFINITION", []))):
        if ec not in ec_set:
            ec_set.append(ec)

    # Diseases
    diseases = [line.split(None, 1)[1] if " " in line else line for line in sections.get("DISEASE", [])]

    # PDB IDs from STRUCTURE section
    pdb_ids: list[str] = []
    for line in sections.get("STRUCTURE", []):
        if line.startswith("PDB:"):
            pdb_ids.extend(line.replace("PDB:", "").split())

    # KO
    ko_lines = sections.get("ORTHOLOGY", [])
    ko = ko_lines[0].split()[0] if ko_lines else None
    ko_def = " ".join(ko_lines[0].split()[1:]) if ko_lines else None

    # Organism
    org_lines = sections.get("ORGANISM", [])
    org_code = org_name = None
    if org_lines:
        parts = org_lines[0].split(None, 1)
        org_code = parts[0]
        org_name = parts[1] if len(parts) > 1 else None

    return KEGGGeneEntry(
        kegg_id=kegg_id,
        name=sections.get("NAME", [""])[0] if sections.get("NAME") else None,
        definition=" ".join(sections.get("DEFINITION", [])),
        ko=ko,
        ko_definition=ko_def,
        organism=org_name,
        organism_code=org_code,
        pathways=pathways,
        modules=modules,
        ec_numbers=ec_set,
        diseases=diseases,
        db_links=_parse_db_links(sections),
        pdb_ids=pdb_ids,
    )


def _parse_pathway_entry(sections: dict, kegg_id: str) -> KEGGPathwayEntry:
    genes = [line.split(None, 1)[0] for line in sections.get("GENE", []) if line.split()]
    compounds = [line.split(None, 1)[0] for line in sections.get("COMPOUND", []) if line.split()]
    return KEGGPathwayEntry(
        kegg_id=kegg_id,
        name=sections.get("NAME", [""])[0] if sections.get("NAME") else None,
        description=" ".join(sections.get("DESCRIPTION", [])),
        pathway_class=" ".join(sections.get("CLASS", [])),
        genes=genes,
        compounds=compounds,
        ko_pathway=sections.get("KO_PATHWAY", [None])[0],
        db_links=_parse_db_links(sections),
    )


def _parse_enzyme_entry(sections: dict, ec_number: str) -> KEGGEnzymeEntry:
    genes: dict[str, list[str]] = {}
    for line in sections.get("GENES", []):
        if ":" in line:
            org, ids_str = line.split(":", 1)
            gene_ids = re.findall(r"(\w+)(?:\([^)]*\))?", ids_str)
            genes[org.strip().lower()] = [g for g in gene_ids if g]

    substrates = [line.split("[")[0].strip() for line in sections.get("SUBSTRATE", []) if line.strip()]
    products = [line.split("[")[0].strip() for line in sections.get("PRODUCT", []) if line.strip()]

    return KEGGEnzymeEntry(
        ec_number=ec_number,
        name=sections.get("NAME", [""])[0] if sections.get("NAME") else None,
        ec_class=" ".join(sections.get("CLASS", [])),
        sysname=sections.get("SYSNAME", [""])[0] if sections.get("SYSNAME") else None,
        reaction=" ".join(sections.get("REACTION", [])),
        substrates=substrates,
        products=products,
        genes=genes,
        db_links=_parse_db_links(sections),
    )


# ── Client ────────────────────────────────────────────────────────────────────

class KEGGClient(BaseClient):
    """
    Client for the KEGG REST API (rest.kegg.jp).

    All responses are plain text (tab-separated or KEGG flat-file format),
    not JSON. This client parses those text formats into typed Pydantic models.
    """

    BASE_URL = "https://rest.kegg.jp"

    def __init__(self, rate_limit_delay: float = 0.4) -> None:
        super().__init__(self.BASE_URL, rate_limit_delay=rate_limit_delay)
        # KEGG returns plain text
        self.session.headers.update({"Accept": "text/plain"})

    def _text(self, path: str, **kwargs) -> str:
        url = f"{self.BASE_URL}/{path}"
        return self._get(url, **kwargs).text

    # ── ID conversion ─────────────────────────────────────────────────────────

    def conv_uniprot_to_kegg(self, uniprot_acc: str, org: str = "hsa") -> Optional[str]:
        """
        Convert a UniProt accession to a KEGG gene ID.
        Returns e.g. 'hsa:207' or None if not found.
        """
        text = self._text(f"conv/{org}/uniprot:{uniprot_acc}")
        for line in text.strip().splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                return parts[1].strip()
        return None

    def conv_batch(
        self, uniprot_accs: list[str], org: str = "hsa"
    ) -> dict[str, str]:
        """
        Convert up to 10 UniProt accessions to KEGG gene IDs in one request.
        Returns {uniprot_acc: kegg_id}.
        """
        ids = "+".join(f"uniprot:{acc}" for acc in uniprot_accs)
        text = self._text(f"conv/{org}/{ids}")
        result: dict[str, str] = {}
        for line in text.strip().splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                up_id = parts[0].replace("up:", "").strip()
                result[up_id] = parts[1].strip()
        return result

    # ── link queries ──────────────────────────────────────────────────────────

    def get_pathways_for_gene(self, kegg_gene_id: str) -> list[KEGGPathwayRef]:
        """Return all pathway refs for a KEGG gene (e.g., 'hsa:207')."""
        text = self._text(f"link/pathway/{kegg_gene_id}")
        refs: list[KEGGPathwayRef] = []
        for line in text.strip().splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                refs.append(KEGGPathwayRef(kegg_id=parts[1].strip(), name=parts[1].strip()))
        return refs

    def get_ec_for_gene(self, kegg_gene_id: str) -> list[str]:
        """Return EC numbers linked to a KEGG gene."""
        text = self._text(f"link/enzyme/{kegg_gene_id}")
        return [
            line.split("\t")[1].strip().replace("ec:", "")
            for line in text.strip().splitlines()
            if "\t" in line
        ]

    # ── full entry fetch ──────────────────────────────────────────────────────

    def get_gene_entry(self, kegg_gene_id: str) -> KEGGGeneEntry:
        """Fetch and parse a KEGG gene flat-file entry (e.g., 'hsa:207')."""
        raw = self._text(f"get/{kegg_gene_id}")
        sections = parse_kegg_flat_file(raw)
        return _parse_gene_entry(sections, kegg_gene_id)

    def get_pathway_entry(self, pathway_id: str) -> KEGGPathwayEntry:
        """Fetch and parse a KEGG pathway entry (e.g., 'hsa04010')."""
        pathway_id = pathway_id.replace("path:", "")
        raw = self._text(f"get/{pathway_id}")
        sections = parse_kegg_flat_file(raw)
        return _parse_pathway_entry(sections, pathway_id)

    def get_enzyme_entry(self, ec_number: str) -> KEGGEnzymeEntry:
        """Fetch and parse a KEGG enzyme entry (e.g., '2.7.11.1')."""
        ec_number = ec_number.replace("ec:", "")
        raw = self._text(f"get/ec:{ec_number}")
        sections = parse_kegg_flat_file(raw)
        return _parse_enzyme_entry(sections, ec_number)

    def get_raw(self, entry_id: str) -> str:
        """Return the raw KEGG flat-file text for any entry."""
        return self._text(f"get/{entry_id}")

    def find(self, database: str, query: str) -> list[tuple[str, str]]:
        """Search a KEGG database. Returns [(kegg_id, description)]."""
        text = self._text(f"find/{database}/{query}")
        return [
            (parts[0], parts[1])
            for line in text.strip().splitlines()
            if "\t" in line
            for parts in [line.split("\t", 1)]
            if len(parts) == 2
        ]

    def list_pathways(self, org: str = "hsa") -> list[tuple[str, str]]:
        """List all pathways for an organism. Returns [(path_id, description)]."""
        text = self._text(f"list/pathway/{org}")
        return [
            (parts[0], parts[1])
            for line in text.strip().splitlines()
            if "\t" in line
            for parts in [line.split("\t", 1)]
            if len(parts) == 2
        ]

    # ── high-level ────────────────────────────────────────────────────────────

    def get_entry_for_uniprot(
        self, uniprot_acc: str, org: str = "hsa"
    ) -> Optional[KEGGGeneEntry]:
        """Convert a UniProt accession and return the full KEGG gene entry."""
        kegg_id = self.conv_uniprot_to_kegg(uniprot_acc, org=org)
        return self.get_gene_entry(kegg_id) if kegg_id else None
