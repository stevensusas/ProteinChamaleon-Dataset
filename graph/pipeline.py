"""
Graph expansion pipeline for the ProteinChameleon knowledge graph.

Starting from a set of seed proteins, the pipeline:
  1. Fetches full annotations for each protein from all 8 APIs
  2. Writes nodes and edges to Neo4j
  3. Discovers interaction partners (STRING) and enqueues them for BFS expansion

The graph grows outward from seeds until no new partners are found above the
score threshold, or an optional protein cap is reached.

When workers > 1 the BFS is executed by a thread pool so multiple proteins
are annotated and written concurrently.

Node types written:   Protein, ProteinFeature, GOTerm
Edge types written:   INTERACTS_WITH, HAS_GO_ANNOTATION, HAS_FEATURE
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from clients.uniprot import UniProtClient
from clients.interpro import InterProClient
from clients.string_db import STRINGClient
from clients.go_client import GOClient
from clients.alphafold import AlphaFoldClient
from clients.pdb import PDBClient
from clients.pubmed import PubMedClient

from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# Per-API concurrency caps (semaphore values).
_API_CONCURRENCY = {
    "uniprot": 3,
    "interpro": 2,
    "string": 1,
    "go": 2,
    "alphafold": 2,
    "pdb": 3,
    "pubmed": 3,
}


class GraphExpansionPipeline:
    """
    Multithreaded BFS graph expansion from seed proteins.

    For each protein (seed or discovered partner):
      - Fetch UniProt → write Protein node
      - Fetch PDB → add pdb_ids to Protein
      - Fetch AlphaFold → add alphafold structure links
      - Fetch InterPro → write ProteinFeature nodes + HAS_FEATURE edges
      - Fetch GO (QuickGO) → write GOTerm nodes + HAS_GO_ANNOTATION edges
      - Fetch STRING → write INTERACTS_WITH edges, enqueue unseen partners

    Expansion continues until the BFS queue is exhausted or max_proteins is hit.
    """

    _PMID_RE = re.compile(r"PMID[:\s]?(\d+)", re.IGNORECASE)
    _PMC_FULL_TEXT_MAX_CHARS = 100_000

    def __init__(
        self,
        neo4j: Neo4jClient,
        string_score_threshold: int = 700,
        max_proteins: Optional[int] = None,
        workers: int = 6,
    ) -> None:
        self.neo4j = neo4j
        self.string_score_threshold = string_score_threshold
        self.max_proteins = max_proteins
        self.workers = max(1, workers)

        # Initialise API clients with per-API concurrency semaphores
        self.uniprot = UniProtClient(max_concurrency=_API_CONCURRENCY["uniprot"])
        self.interpro = InterProClient(max_concurrency=_API_CONCURRENCY["interpro"])
        self.string_db = STRINGClient(max_concurrency=_API_CONCURRENCY["string"])
        self.go = GOClient(max_concurrency=_API_CONCURRENCY["go"])
        self.alphafold = AlphaFoldClient(max_concurrency=_API_CONCURRENCY["alphafold"])
        self.pdb = PDBClient(max_concurrency=_API_CONCURRENCY["pdb"])
        self.pubmed = PubMedClient(max_concurrency=_API_CONCURRENCY["pubmed"])

        self._pubmed_cache: dict[str, dict] = {}
        self._pubmed_lock = threading.Lock()

    def close(self) -> None:
        for client in (
            self.uniprot, self.interpro, self.string_db,
            self.go, self.alphafold, self.pdb, self.pubmed,
        ):
            client.close()

    def __enter__(self) -> "GraphExpansionPipeline":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        seed_accessions: list[str],
        clear: bool = True,
        output_dir: Optional[str | Path] = None,
    ) -> dict[str, int]:
        """
        Multithreaded BFS expansion from seed proteins.

        Workers pull accessions from a shared queue, process them (API calls +
        Neo4j writes), and push newly discovered partners back onto the queue.
        Continues until the queue is drained or max_proteins is reached.

        Args:
            seed_accessions: Starting UniProt accessions.
            clear: Wipe the graph before this run (default True).
            output_dir: If given, save growth_log.json and growth_chart.png here.
        """
        if clear:
            self.neo4j.clear_graph()
        self.neo4j.setup_schema()

        bfs_queue: queue.Queue[str] = queue.Queue()
        visited: set[str] = set()
        visited_lock = threading.Lock()
        processed_count = 0
        processed_lock = threading.Lock()
        cap_reached = threading.Event()

        t0 = time.monotonic()
        snapshot_file = None
        snapshot_file_lock = threading.Lock()

        for acc in seed_accessions:
            if acc not in visited:
                visited.add(acc)
                bfs_queue.put(acc)

        def _worker() -> None:
            nonlocal processed_count
            while not cap_reached.is_set():
                try:
                    accession = bfs_queue.get(timeout=3)
                except queue.Empty:
                    return

                with processed_lock:
                    if self.max_proteins is not None and processed_count >= self.max_proteins:
                        bfs_queue.task_done()
                        cap_reached.set()
                        return
                    processed_count += 1
                    seq = processed_count

                logger.info(
                    "Processing %s (#%d, ~%d queued)",
                    accession, seq, bfs_queue.qsize(),
                )

                try:
                    new_partners = self._process_protein(accession)
                except Exception:
                    logger.exception("Unhandled error processing %s", accession)
                    new_partners = []

                with visited_lock:
                    for partner_acc in new_partners:
                        if partner_acc not in visited:
                            visited.add(partner_acc)
                            bfs_queue.put(partner_acc)

                try:
                    counts = self.neo4j.get_counts()
                    snap = {"seq": seq, "accession": accession, "elapsed_s": round(time.monotonic() - t0, 1)}
                    snap.update(counts)
                    if snapshot_file is not None:
                        with snapshot_file_lock:
                            snapshot_file.write(json.dumps(snap) + "\n")
                            snapshot_file.flush()
                except Exception:
                    pass

                bfs_queue.task_done()

        try:
            if output_dir is not None:
                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                snapshot_file = open(out_path / "growth_log.jsonl", "a", encoding="utf-8")

            with ThreadPoolExecutor(
                max_workers=self.workers, thread_name_prefix="bfs"
            ) as pool:
                futures = [pool.submit(_worker) for _ in range(self.workers)]
                try:
                    bfs_queue.join()
                except KeyboardInterrupt:
                    logger.info("Interrupted — stopping workers and saving progress...")
                finally:
                    cap_reached.set()
                for f in futures:
                    try:
                        f.result()
                    except Exception:
                        pass
        finally:
            if snapshot_file is not None and not snapshot_file.closed:
                snapshot_file.close()
            if output_dir is not None:
                self._save_growth_chart(Path(output_dir))

        logger.info(
            "Pipeline complete. Processed %d protein(s), %d discovered total.",
            processed_count, len(visited),
        )

        return {"proteins_processed": processed_count, "proteins_discovered": len(visited)}

    # ── Growth chart ─────────────────────────────────────────────────────────

    @staticmethod
    def _save_growth_chart(output_dir: Path) -> None:
        """Read incremental growth_log.jsonl and generate growth_chart.png."""
        jsonl_path = output_dir / "growth_log.jsonl"
        if not jsonl_path.exists():
            logger.warning("No growth_log.jsonl found in %s", output_dir)
            return

        snapshots: list[dict[str, Any]] = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    snapshots.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not snapshots:
            return

        snapshots.sort(key=lambda s: s["seq"])

        json_path = output_dir / "growth_log.json"
        json_path.write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
        logger.info("Saved growth log (%d snapshots) to %s", len(snapshots), json_path)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed — skipping growth chart.")
            return

        node_labels = ["Protein", "ProteinFeature", "GOTerm", "PubMedArticle"]
        edge_labels = ["INTERACTS_WITH", "HAS_FEATURE", "HAS_GO_ANNOTATION", "CITED_IN"]

        x = [s["elapsed_s"] for s in snapshots]

        fig, (ax_nodes, ax_edges) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        for label in node_labels:
            y = [s.get(label, 0) for s in snapshots]
            ax_nodes.plot(x, y, marker="o", markersize=3, label=label)
        ax_nodes.set_ylabel("Node count")
        ax_nodes.set_title("Graph Growth Over Time")
        ax_nodes.legend(loc="upper left")
        ax_nodes.grid(True, alpha=0.3)

        for label in edge_labels:
            y = [s.get(label, 0) for s in snapshots]
            ax_edges.plot(x, y, marker="o", markersize=3, label=label)
        ax_edges.set_ylabel("Edge count")
        ax_edges.set_xlabel("Elapsed time (seconds)")
        ax_edges.legend(loc="upper left")
        ax_edges.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = output_dir / "growth_chart.png"
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        logger.info("Saved growth chart to %s", chart_path)

    @classmethod
    def generate_chart(cls, output_dir: str | Path) -> None:
        """Standalone: regenerate chart from an existing growth_log.jsonl."""
        cls._save_growth_chart(Path(output_dir))

    # ── Per-protein processing ────────────────────────────────────────────────

    def _extract_pmids(self, value: Any) -> set[str]:
        """Recursively extract PMID values from strings / nested dicts / lists."""
        pmids: set[str] = set()

        def walk(v: Any) -> None:
            if v is None:
                return
            if isinstance(v, str):
                pmids.update(self._PMID_RE.findall(v))
                return
            if isinstance(v, dict):
                for child in v.values():
                    walk(child)
                return
            if isinstance(v, (list, tuple, set)):
                for child in v:
                    walk(child)

        walk(value)
        return pmids

    def _ensure_pubmed_articles(self, s, pmids: set[str]) -> None:
        """Ensure PubMedArticle nodes exist for PMID evidence."""
        with self._pubmed_lock:
            missing = [pmid for pmid in pmids if pmid not in self._pubmed_cache]

        if missing:
            try:
                articles = self.pubmed.fetch_by_pmids(missing)
                fetched = {a.pmid: a for a in articles}
                with self._pubmed_lock:
                    for pmid in missing:
                        if pmid in self._pubmed_cache:
                            continue
                        article = fetched.get(pmid)
                        if article:
                            full_text = ""
                            if article.pmc:
                                pmc_text = self.pubmed.fetch_pmc_full_text(article.pmc)
                                if pmc_text:
                                    full_text = pmc_text[: self._PMC_FULL_TEXT_MAX_CHARS]
                            self._pubmed_cache[pmid] = {
                                "pmid": article.pmid,
                                "title": article.title or "",
                                "abstract": article.abstract or "",
                                "journal": article.journal or "",
                                "journal_abbrev": article.journal_abbrev or "",
                                "year": article.year or "",
                                "volume": article.volume or "",
                                "issue": article.issue or "",
                                "pages": article.pages or "",
                                "doi": article.doi or "",
                                "pmc": article.pmc or "",
                                "pubmed_url": article.pubmed_url,
                                "authors": article.author_names,
                                "mesh_terms": article.mesh_terms,
                                "publication_types": article.publication_types,
                                "full_text": full_text,
                            }
                        else:
                            self._pubmed_cache[pmid] = {"pmid": pmid}
            except Exception as exc:
                logger.warning("PubMed fetch failed for %d PMID(s): %s", len(missing), exc)
                with self._pubmed_lock:
                    for pmid in missing:
                        if pmid not in self._pubmed_cache:
                            self._pubmed_cache[pmid] = {"pmid": pmid}

        with self._pubmed_lock:
            cached = {pmid: self._pubmed_cache[pmid] for pmid in pmids if pmid in self._pubmed_cache}
        for pmid, props in cached.items():
            self.neo4j.merge_pubmed_article(s, props)

    def _link_go_evidence(self, s, protein_acc: str, reference: Optional[str]) -> None:
        """Attach PubMed evidence for GO annotations via Protein -> CITED_IN."""
        pmids = self._extract_pmids(reference)
        if not pmids:
            return
        self._ensure_pubmed_articles(s, pmids)
        for pmid in pmids:
            self.neo4j.merge_cited_in(s, protein_acc, pmid)

    def _link_feature_evidence(
        self, s, protein_acc: str, feature_acc: str, evidence_blob: Any
    ) -> None:
        """Attach PubMed evidence for protein feature matches."""
        pmids = self._extract_pmids(evidence_blob)
        if not pmids:
            return
        self._ensure_pubmed_articles(s, pmids)
        for pmid in pmids:
            self.neo4j.merge_feature_evidence(s, protein_acc, feature_acc, pmid)

    def _write_protein_node(self, accession: str, entry) -> None:
        """Build full protein props (UniProt + PDB + AlphaFold), merge node, write InterPro and GO."""
        protein_props = {
            "accession": entry.primaryAccession,
            "name": entry.protein_name or "",
            "gene_name": entry.gene_name or "",
            "organism": entry.organism.scientificName,
            "taxon_id": entry.organism.taxonId,
            "sequence": entry.sequence.value,
            "length": entry.sequence.length,
            "mol_weight": entry.sequence.molWeight,
            "is_reviewed": entry.is_reviewed,
            "function_text": entry.get_function_text() or "",
            "ec_numbers": entry.get_ec_numbers(),
            "string_id": entry.get_string_id() or "",
        }

        # PDB structures (from PDB Search API)
        try:
            pdb_result = self.pdb.search_by_uniprot(accession, max_rows=100)
            pdb_ids = pdb_result.identifiers
            protein_props["pdb_ids"] = pdb_ids
            protein_props["pdb_entry_urls"] = [
                f"https://www.rcsb.org/structure/{pdb_id}" for pdb_id in pdb_ids
            ]
            protein_props["pdb_cif_urls"] = [
                f"https://files.rcsb.org/download/{pdb_id}.cif" for pdb_id in pdb_ids
            ]
            protein_props["pdb_pdb_urls"] = [
                f"https://files.rcsb.org/download/{pdb_id}.pdb" for pdb_id in pdb_ids
            ]
        except Exception as exc:
            logger.warning("PDB search failed for %s: %s", accession, exc)
            protein_props["pdb_ids"] = []
            protein_props["pdb_entry_urls"] = []
            protein_props["pdb_cif_urls"] = []
            protein_props["pdb_pdb_urls"] = []

        # AlphaFold structure (from AlphaFold API)
        try:
            preds = self.alphafold.get_prediction(accession)
            protein_props["alphafold_entry_ids"] = [p.entryId for p in preds]
            protein_props["alphafold_cif_urls"] = [p.cifUrl or "" for p in preds]
            protein_props["alphafold_pdb_urls"] = [p.pdbUrl or "" for p in preds]
            best = self.alphafold.get_best_prediction(accession)
            if best:
                protein_props["alphafold_cif_url"] = best.cifUrl or ""
                protein_props["alphafold_pdb_url"] = best.pdbUrl or ""
            else:
                protein_props["alphafold_cif_url"] = ""
                protein_props["alphafold_pdb_url"] = ""
        except Exception as exc:
            logger.warning("AlphaFold fetch failed for %s: %s", accession, exc)
            protein_props["alphafold_entry_ids"] = []
            protein_props["alphafold_cif_urls"] = []
            protein_props["alphafold_pdb_urls"] = []
            protein_props["alphafold_cif_url"] = ""
            protein_props["alphafold_pdb_url"] = ""

        with self.neo4j.session() as s:
            self.neo4j.merge_protein(s, protein_props)

        self._write_interpro(accession)
        self._write_go_annotations(accession)

    def _process_protein(self, accession: str) -> list[str]:
        """
        Fetch all data for one protein, write to Neo4j, discover STRING
        partners, and return their accessions for BFS enqueuing.
        """
        try:
            entry = self.uniprot.get_entry(accession)
        except Exception as exc:
            logger.warning("UniProt fetch failed for %s: %s", accession, exc)
            return []

        self._write_protein_node(accession, entry)
        return self._write_ppi(accession, entry)

    # ── InterPro ──────────────────────────────────────────────────────────────

    def _write_interpro(self, accession: str) -> None:
        try:
            info = self.interpro.get_all_matches(accession)
        except Exception as exc:
            logger.warning("InterPro fetch failed for %s: %s", accession, exc)
            return

        if not info.matches:
            logger.info("InterPro: no feature matches for %s", accession)
            return

        logger.info("InterPro: %s -> %d feature(s)", accession, len(info.matches))

        with self.neo4j.session() as s:
            for match in info.matches:
                meta = match.metadata
                feature_props = {
                    "accession": meta.accession,
                    "name": meta.name,
                    "source_database": meta.source_database,
                    "type": meta.type or "",
                    "integrated": meta.integrated or "",
                }
                self.neo4j.merge_protein_feature(s, feature_props)

                for loc in match.locations:
                    for frag in loc.fragments:
                        edge_props = {
                            "start": frag.start,
                            "end": frag.end,
                            "score": loc.score,
                            "model": loc.model or "",
                        }
                        self.neo4j.merge_has_feature(
                            s, accession, meta.accession, edge_props
                        )
                # InterPro sometimes contains PMID references in nested metadata payloads.
                self._link_feature_evidence(
                    s, accession, meta.accession, match.model_dump()
                )

    # ── GO annotations ────────────────────────────────────────────────────────

    def _write_go_annotations(self, accession: str) -> None:
        try:
            annotations = self.go.get_annotations(accession)
        except Exception as exc:
            logger.warning("GO fetch failed for %s: %s", accession, exc)
            return

        # Collect unique GO IDs to fetch term details
        go_ids = list({a.goId for a in annotations})
        go_terms: dict[str, dict] = {}

        # Batch fetch term details (up to 10 at a time)
        batch_size = 10
        for i in range(0, len(go_ids), batch_size):
            batch = go_ids[i : i + batch_size]
            try:
                terms = self.go.get_terms_batch(batch)
                for t in terms:
                    go_terms[t.id] = {
                        "go_id": t.id,
                        "name": t.name,
                        "aspect": t.aspect or "",
                        "definition": t.definition_text,
                        "is_obsolete": t.isObsolete,
                    }
            except Exception as exc:
                logger.warning("GO term batch fetch failed: %s", exc)

        with self.neo4j.session() as s:
            for ann in annotations:
                # Ensure GOTerm node exists
                term_props = go_terms.get(ann.goId, {
                    "go_id": ann.goId,
                    "name": ann.goName or "",
                    "aspect": ann.goAspect or "",
                    "definition": "",
                    "is_obsolete": False,
                })
                self.neo4j.merge_go_term(s, term_props)

                edge_props = {
                    "qualifier": ann.qualifier or "",
                    "evidence_code": ann.goEvidence or "",
                    "eco_code": ann.evidenceCode or "",
                    "reference": ann.reference or "",
                    "assigned_by": ann.assignedBy or "",
                    "is_experimental": ann.is_experimental,
                }
                self.neo4j.merge_go_annotation(s, accession, ann.goId, edge_props)
                self._link_go_evidence(s, accession, ann.reference)

    # ── STRING PPI ────────────────────────────────────────────────────────────

    def _write_ppi(self, accession: str, entry) -> list[str]:
        """
        Fetch all STRING interaction partners above the score threshold,
        write placeholder Protein nodes and INTERACTS_WITH edges, and return
        the partner accessions so the BFS caller can enqueue them.
        """
        string_id = entry.get_string_id()
        if not string_id:
            logger.debug("No STRING ID for %s, skipping PPI", accession)
            return []

        try:
            interactions = self.string_db.get_interaction_partners(
                [string_id],
                taxon=entry.organism.taxonId,
                required_score=self.string_score_threshold,
            )
        except Exception as exc:
            logger.warning("STRING fetch failed for %s: %s", accession, exc)
            return []

        discovered: list[str] = []
        with self.neo4j.session() as s:
            for iact in interactions:
                partner_acc = self._resolve_string_to_uniprot(
                    iact.stringId_B, iact.preferredName_B, entry.organism.taxonId
                )
                if not partner_acc:
                    continue

                self.neo4j.merge_protein(s, {"accession": partner_acc})

                edge_props = {
                    "score": iact.score,
                    "confidence": iact.confidence,
                    "escore": iact.escore,
                    "dscore": iact.dscore,
                    "tscore": iact.tscore,
                    "nscore": iact.nscore,
                    "ascore": iact.ascore,
                    "fscore": iact.fscore,
                    "dominant_source": iact.dominant_source,
                }
                self.neo4j.merge_ppi(s, accession, partner_acc, edge_props)
                discovered.append(partner_acc)

        logger.info("STRING: %s -> %d partner(s) discovered", accession, len(discovered))
        return discovered

    def _resolve_string_to_uniprot(
        self, string_id: str, gene_name: str, taxon_id: int
    ) -> Optional[str]:
        """
        Try to resolve a STRING ID to a UniProt accession via UniProt search.
        Falls back to gene name search if cross-reference lookup fails.
        """
        try:
            # Search UniProt for this gene in this organism
            results = list(self.uniprot.search(
                f"gene:{gene_name} AND organism_id:{taxon_id} AND reviewed:true",
                size=1,
            ))
            if results:
                return results[0].primaryAccession
        except Exception:
            pass

        try:
            # Fallback: search unreviewed too
            results = list(self.uniprot.search(
                f"gene:{gene_name} AND organism_id:{taxon_id}",
                size=1,
            ))
            if results:
                return results[0].primaryAccession
        except Exception:
            pass

        return None
