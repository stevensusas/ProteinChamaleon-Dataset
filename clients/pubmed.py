"""NCBI Entrez E-utilities / PubMed API client."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

from pydantic import BaseModel, Field

from .base import BaseClient


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PubMedAuthor(BaseModel):
    last_name: Optional[str] = None
    fore_name: Optional[str] = None
    initials: Optional[str] = None
    affiliation: Optional[str] = None

    @property
    def full_name(self) -> str:
        parts = [p for p in (self.last_name, self.fore_name) if p]
        return " ".join(parts)


class PubMedArticle(BaseModel):
    pmid: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    journal: Optional[str] = None
    journal_abbrev: Optional[str] = None
    year: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    pmc: Optional[str] = None
    authors: list[PubMedAuthor] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)

    @property
    def author_names(self) -> list[str]:
        return [a.full_name for a in self.authors]

    @property
    def citation(self) -> str:
        names = self.author_names[:3]
        suffix = " et al." if len(self.authors) > 3 else ""
        journal = self.journal_abbrev or self.journal or ""
        return f"{', '.join(names)}{suffix}. {self.title}. {journal}. {self.year or ''}."

    @property
    def pubmed_url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


# ── XML helpers ───────────────────────────────────────────────────────────────

def _el_text(el: Optional[ET.Element], xpath: str) -> Optional[str]:
    if el is None:
        return None
    node = el.find(xpath)
    return node.text if node is not None else None


def _parse_article(el: ET.Element) -> PubMedArticle:
    pmid = _el_text(el, ".//PMID") or ""
    title = _el_text(el, ".//ArticleTitle") or ""

    # Structured abstracts may have multiple <AbstractText Label="...">
    abstract_parts = []
    for ab in el.findall(".//AbstractText"):
        label = ab.get("Label", "")
        content = ab.text or ""
        abstract_parts.append(f"{label}: {content}" if label else content)
    abstract = " ".join(abstract_parts)

    journal = _el_text(el, ".//Journal/Title")
    journal_abbrev = _el_text(el, ".//ISOAbbreviation")
    year = _el_text(el, ".//PubDate/Year") or _el_text(el, ".//PubDate/MedlineDate")
    volume = _el_text(el, ".//JournalIssue/Volume")
    issue = _el_text(el, ".//JournalIssue/Issue")
    pages = _el_text(el, ".//Pagination/MedlinePgn")

    doi = pmc = None
    for aid in el.findall(".//ArticleId"):
        id_type = aid.get("IdType", "")
        if id_type == "doi":
            doi = aid.text
        elif id_type == "pmc":
            pmc = aid.text

    authors = []
    for auth_el in el.findall(".//Author"):
        affil_el = auth_el.find(".//AffiliationInfo/Affiliation")
        authors.append(PubMedAuthor(
            last_name=_el_text(auth_el, "LastName"),
            fore_name=_el_text(auth_el, "ForeName"),
            initials=_el_text(auth_el, "Initials"),
            affiliation=affil_el.text if affil_el is not None else None,
        ))

    mesh_terms = [n.text for n in el.findall(".//MeshHeading/DescriptorName") if n.text]
    pub_types = [n.text for n in el.findall(".//PublicationType") if n.text]

    return PubMedArticle(
        pmid=pmid,
        title=title,
        abstract=abstract,
        journal=journal,
        journal_abbrev=journal_abbrev,
        year=year,
        volume=volume,
        issue=issue,
        pages=pages,
        doi=doi,
        pmc=pmc,
        authors=authors,
        mesh_terms=mesh_terms,
        publication_types=pub_types,
    )


def _parse_efetch_xml(xml_content: bytes) -> list[PubMedArticle]:
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return []
    articles = []
    for article_el in root.findall(".//PubmedArticle"):
        try:
            articles.append(_parse_article(article_el))
        except Exception:
            continue
    return articles


def _parse_pmc_full_text(xml_content: bytes) -> str:
    """Extract concatenated paragraph text from a PMC full-text XML payload."""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return ""

    paragraphs: list[str] = []
    for el in root.iter():
        # Handle both namespaced and non-namespaced tags.
        if el.tag == "p" or str(el.tag).endswith("}p"):
            text = "".join(el.itertext()).strip()
            if text:
                paragraphs.append(" ".join(text.split()))

    return "\n\n".join(paragraphs)


# ── Client ────────────────────────────────────────────────────────────────────

class PubMedClient(BaseClient):
    """
    Client for NCBI Entrez E-utilities (PubMed).

    Register a free API key at https://www.ncbi.nlm.nih.gov/account/ to
    raise the rate limit from 3 → 10 requests/second.
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(
        self,
        api_key: Optional[str] = None,
        email: Optional[str] = "research@example.com",
        tool: str = "ProteinChameleonDataset",
        max_concurrency: int | None = None,
    ) -> None:
        # Respect NCBI rate limits: 3 req/s without key, 10 req/s with key
        delay = 0.11 if api_key else 0.35
        super().__init__(self.BASE_URL, rate_limit_delay=delay, max_concurrency=max_concurrency)
        self.api_key = api_key
        self.email = email
        self.tool = tool

    def _base_params(self) -> dict:
        params = {"tool": self.tool, "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def search(
        self,
        query: str,
        max_results: int = 100,
        sort: str = "relevance",
        date_range: Optional[tuple[str, str]] = None,
    ) -> tuple[list[str], Optional[str], Optional[str]]:
        """
        Search PubMed.

        Returns (pmid_list, webenv, query_key).
        Use webenv + query_key with fetch() to retrieve large result sets
        without re-sending the full PMID list.

        Example queries:
            'TP53[Gene Name] AND "Homo sapiens"[Organism]'
            'insulin AND diabetes[MeSH Terms]'
        """
        params = {
            **self._base_params(),
            "db": "pubmed",
            "term": query,
            "retmax": min(max_results, 10_000),
            "retmode": "json",
            "sort": sort,
            "usehistory": "y",
        }
        if date_range:
            params["datetype"] = "pdat"
            params["mindate"] = date_range[0]
            params["maxdate"] = date_range[1]

        resp = self._get(f"{self.BASE_URL}/esearch.fcgi", params=params)
        result = resp.json().get("esearchresult", {})
        return (
            result.get("idlist", []),
            result.get("webenv"),
            result.get("querykey"),
        )

    def fetch(
        self,
        pmids: Optional[list[str]] = None,
        webenv: Optional[str] = None,
        query_key: Optional[str] = None,
        batch_size: int = 200,
    ) -> list[PubMedArticle]:
        """
        Fetch full article records (title, abstract, authors, DOI, etc.).
        Supply either pmids or (webenv, query_key) from a prior search().
        """
        if not pmids and not (webenv and query_key):
            return []

        url = f"{self.BASE_URL}/efetch.fcgi"
        articles: list[PubMedArticle] = []

        if pmids:
            for i in range(0, len(pmids), batch_size):
                batch = pmids[i : i + batch_size]
                params = {
                    **self._base_params(),
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "rettype": "abstract",
                    "retmode": "xml",
                }
                resp = self._get(url, params=params)
                articles.extend(_parse_efetch_xml(resp.content))
        else:
            params = {
                **self._base_params(),
                "db": "pubmed",
                "webenv": webenv,
                "query_key": query_key,
                "rettype": "abstract",
                "retmode": "xml",
                "retmax": batch_size,
                "retstart": 0,
            }
            resp = self._get(url, params=params)
            articles.extend(_parse_efetch_xml(resp.content))

        return articles

    def search_and_fetch(
        self,
        gene_name: str,
        organism: str = "Homo sapiens",
        max_results: int = 20,
    ) -> list[PubMedArticle]:
        """Convenience: search by gene + organism and return full article records."""
        query = f'{gene_name}[Gene Name] AND "{organism}"[Organism]'
        pmids, webenv, query_key = self.search(query, max_results=max_results)
        return self.fetch(pmids=pmids[:max_results]) if pmids else []

    def fetch_by_pmids(self, pmids: list[str]) -> list[PubMedArticle]:
        """Fetch articles directly by a list of PMIDs."""
        return self.fetch(pmids=pmids)

    def fetch_pmc_full_text(self, pmc: str) -> Optional[str]:
        """
        Fetch full text from PMC (when available) as plain concatenated paragraphs.
        Returns None if not available or retrieval/parsing fails.
        """
        if not pmc:
            return None
        pmc_id = pmc if pmc.upper().startswith("PMC") else f"PMC{pmc}"
        params = {
            **self._base_params(),
            "db": "pmc",
            "id": pmc_id,
            "rettype": "full",
            "retmode": "xml",
        }
        try:
            resp = self._get(f"{self.BASE_URL}/efetch.fcgi", params=params)
            text = _parse_pmc_full_text(resp.content)
            return text or None
        except Exception:
            return None
