# ProteinChameleon Demo — Run & Visualize in Neo4j

## 1. Start Neo4j

**If Docker isn’t running:** start Docker Desktop (or the Docker daemon) first, then run the commands below.

**Option A — Docker Compose (recommended)**

```bash
docker compose up -d
```

Wait until Neo4j is up (e.g. 10–20 seconds), then continue.

**Option B — Plain Docker**

```bash
docker run -d --name neo4j-proteinchameleon \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

## 2. Run the demo

Activate the conda env and run the graph expansion:

```bash
conda activate proteinchameleon
cd /Users/stevensu/ProteinChamaleonDataset
python demo.py
```

This will:

- Connect to Neo4j at `bolt://localhost:7687`
- Create schema (constraints/indexes)
- Start from 5 seed proteins (TP53, AKT1, EGFR, BRAF, BRCA1)
- Fetch data from UniProt, InterPro, GO, STRING, PubMed, KEGG
- Write **Protein**, **GOTerm**, **Pathway**, **PubMedArticle**, **ProteinFeature** nodes and their relationships

Tune (optional):

- `MAX_DEPTH=1` (default) — seeds + one hop of interaction partners
- `STRING_SCORE=700` — interaction confidence threshold (0–1000)
- `MAX_PARTNERS=10` — max STRING partners per protein
- `MAX_PAPERS=5` — max PubMed papers per protein

Example:

```bash
MAX_DEPTH=1 MAX_PARTNERS=15 python demo.py
```

## 3. Visualize in Neo4j Browser

1. Open **http://localhost:7474** in your browser.
2. Log in: **Connect URL** `neo4j://localhost:7687`, **Username** `neo4j`, **Password** `password`.
3. Run Cypher in the query bar to explore the graph.

**Example queries**

- **Protein–protein interactions (first 25):**
  ```cypher
  MATCH (p:Protein)-[r:INTERACTS_WITH]->(q:Protein) RETURN p,r,q LIMIT 25
  ```

- **Proteins and their GO terms:**
  ```cypher
  MATCH (p:Protein)-[:HAS_GO_ANNOTATION]->(g:GOTerm) RETURN p.gene_name, g.name, g.aspect
  ```

- **Proteins and InterPro/features:**
  ```cypher
  MATCH (p:Protein)-[:HAS_FEATURE]->(f:ProteinFeature) RETURN p.gene_name, f.name, f.type
  ```

- **Proteins in KEGG pathways:**
  ```cypher
  MATCH (p:Protein)-[:IN_PATHWAY]->(pw:Pathway) RETURN p.gene_name, pw.name
  ```

- **Proteins and cited PubMed articles:**
  ```cypher
  MATCH (p:Protein)-[:CITED_IN]->(a:PubMedArticle) RETURN p.gene_name, a.title LIMIT 20
  ```

- **Full graph (small sample):**
  ```cypher
  MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 100
  ```

Use the graph visualization in Neo4j Browser to zoom, pan, and click nodes/relationships for details.
