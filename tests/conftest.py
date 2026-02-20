"""Shared pytest fixtures for all API client integration tests."""

import pytest

from clients.uniprot import UniProtClient
from clients.pdb import PDBClient
from clients.alphafold import AlphaFoldClient
from clients.interpro import InterProClient
from clients.string_db import STRINGClient
from clients.pubmed import PubMedClient
from clients.kegg import KEGGClient
from clients.go_client import GOClient


# ── Well-known test proteins ───────────────────────────────────────────────────
# These are stable, canonical proteins used across all tests.

# Hemoglobin subunit alpha (Human) — small, well-studied, reviewed Swiss-Prot
HBA1_ACC = "P69905"
HBA1_PDB = "4HHB"      # Deoxy-hemoglobin, 1.74 Å resolution
HBA1_GENE = "HBA1"

# Tumor suppressor p53 (Human) — very well-annotated, many interactions
TP53_ACC = "P04637"
TP53_GENE = "TP53"

# AKT serine/threonine kinase 1 (Human) — good enzyme + pathway coverage
AKT1_ACC = "P31749"
AKT1_KEGG = "hsa:207"
AKT1_EC = "2.7.11.1"


# ── Client fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def uniprot():
    with UniProtClient() as c:
        yield c


@pytest.fixture(scope="session")
def pdb():
    with PDBClient() as c:
        yield c


@pytest.fixture(scope="session")
def alphafold():
    with AlphaFoldClient() as c:
        yield c


@pytest.fixture(scope="session")
def interpro():
    with InterProClient() as c:
        yield c


@pytest.fixture(scope="session")
def string_db():
    with STRINGClient() as c:
        yield c


@pytest.fixture(scope="session")
def pubmed():
    with PubMedClient() as c:
        yield c


@pytest.fixture(scope="session")
def kegg():
    with KEGGClient() as c:
        yield c


@pytest.fixture(scope="session")
def go():
    with GOClient() as c:
        yield c
