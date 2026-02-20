"""Integration tests for the NCBI PubMed / E-utilities API client."""

import pytest

from clients.pubmed import PubMedArticle, PubMedAuthor
from .conftest import HBA1_GENE, TP53_GENE


pytestmark = pytest.mark.integration


class TestPubMedSearch:
    def test_search_returns_pmids(self, pubmed):
        pmids, webenv, query_key = pubmed.search(
            f'{HBA1_GENE}[Gene Name] AND "Homo sapiens"[Organism]',
            max_results=10,
        )
        assert isinstance(pmids, list)
        assert len(pmids) > 0
        assert all(pmid.isdigit() for pmid in pmids)

    def test_search_returns_webenv(self, pubmed):
        _, webenv, query_key = pubmed.search(
            f'{HBA1_GENE}[Gene Name] AND "Homo sapiens"[Organism]',
            max_results=5,
        )
        assert webenv is not None
        assert query_key is not None

    def test_search_with_date_range(self, pubmed):
        pmids, _, _ = pubmed.search(
            f'{TP53_GENE}[Gene Name] AND "Homo sapiens"[Organism]',
            max_results=5,
            date_range=("2020/01/01", "2025/12/31"),
        )
        assert isinstance(pmids, list)

    def test_tp53_has_many_papers(self, pubmed):
        pmids, _, _ = pubmed.search(
            f'{TP53_GENE}[Gene Name] AND "Homo sapiens"[Organism]',
            max_results=100,
        )
        assert len(pmids) >= 5  # TP53 is well-published; lower bound allows for rate limiting


class TestPubMedFetch:
    def test_fetch_by_pmids(self, pubmed):
        pmids, _, _ = pubmed.search(
            f'{HBA1_GENE}[Gene Name] AND "Homo sapiens"[Organism]',
            max_results=3,
        )
        articles = pubmed.fetch(pmids=pmids[:3])
        assert isinstance(articles, list)
        assert len(articles) >= 1
        assert len(articles) <= 3  # May get fewer if search returned fewer or fetch was limited

    def test_article_schema_validates(self, pubmed):
        pmids, _, _ = pubmed.search(
            f'{HBA1_GENE}[Gene Name]', max_results=3
        )
        articles = pubmed.fetch(pmids=pmids[:3])
        for a in articles:
            assert isinstance(a, PubMedArticle)

    def test_pmid_field_present(self, pubmed):
        pmids, _, _ = pubmed.search(f'{HBA1_GENE}[Gene Name]', max_results=3)
        articles = pubmed.fetch(pmids=pmids[:3])
        for a in articles:
            assert a.pmid
            assert a.pmid.isdigit()
            assert a.pmid in pmids

    def test_title_present(self, pubmed):
        pmids, _, _ = pubmed.search(f'{HBA1_GENE}[Gene Name]', max_results=3)
        articles = pubmed.fetch(pmids=pmids[:3])
        for a in articles:
            assert a.title is not None
            assert len(a.title) > 5

    def test_journal_present(self, pubmed):
        pmids, _, _ = pubmed.search(f'{TP53_GENE}[Gene Name]', max_results=5)
        articles = pubmed.fetch(pmids=pmids[:5])
        journals = [a.journal for a in articles if a.journal]
        assert len(journals) > 0

    def test_year_is_digit_string(self, pubmed):
        pmids, _, _ = pubmed.search(f'{TP53_GENE}[Gene Name]', max_results=5)
        articles = pubmed.fetch(pmids=pmids[:5])
        for a in articles:
            if a.year and a.year.isdigit():
                year_int = int(a.year)
                assert 1950 <= year_int <= 2030

    def test_authors_list(self, pubmed):
        pmids, _, _ = pubmed.search(f'{TP53_GENE}[Gene Name]', max_results=5)
        articles = pubmed.fetch(pmids=pmids[:5])
        for a in articles:
            assert isinstance(a.authors, list)
            for author in a.authors:
                assert isinstance(author, PubMedAuthor)

    def test_doi_format_when_present(self, pubmed):
        pmids, _, _ = pubmed.search(f'{TP53_GENE}[Gene Name]', max_results=10)
        articles = pubmed.fetch(pmids=pmids[:10])
        for a in articles:
            if a.doi:
                assert "10." in a.doi  # DOIs always start with 10.

    def test_pubmed_url_format(self, pubmed):
        pmids, _, _ = pubmed.search(f'{HBA1_GENE}[Gene Name]', max_results=2)
        articles = pubmed.fetch(pmids=pmids[:2])
        for a in articles:
            assert a.pubmed_url == f"https://pubmed.ncbi.nlm.nih.gov/{a.pmid}/"

    def test_author_full_name(self, pubmed):
        pmids, _, _ = pubmed.search(f'{TP53_GENE}[Gene Name]', max_results=3)
        articles = pubmed.fetch(pmids=pmids[:3])
        for a in articles:
            for author in a.authors:
                name = author.full_name
                assert isinstance(name, str)

    def test_mesh_terms_list(self, pubmed):
        pmids, _, _ = pubmed.search(
            '"Homo sapiens"[Organism] AND hemoglobin[Title/Abstract]',
            max_results=5,
        )
        articles = pubmed.fetch(pmids=pmids[:5])
        for a in articles:
            assert isinstance(a.mesh_terms, list)


class TestPubMedSearchAndFetch:
    def test_search_and_fetch_combined(self, pubmed):
        articles = pubmed.search_and_fetch(
            gene_name=HBA1_GENE, organism="Homo sapiens", max_results=5
        )
        assert isinstance(articles, list)
        assert len(articles) > 0
        assert all(isinstance(a, PubMedArticle) for a in articles)

    def test_fetch_by_pmids_direct(self, pubmed):
        # Known PMID for a hemoglobin paper
        articles = pubmed.fetch_by_pmids(["6726807"])
        assert len(articles) == 1
        assert articles[0].pmid == "6726807"
