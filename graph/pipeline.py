"""
Graph expansion pipeline for the ProteinChameleon knowledge graph.

Starting from a set of seed proteins, the pipeline:
  1. Fetches full annotations for each protein from all 8 APIs
  2. Writes nodes and edges to Neo4j
  3. Discovers interaction partners (STRING) and recurse (BFS, depth-limited)

Node types written:   Protein, ProteinFeature, GOTerm
Edge types written:   INTERACTS_WITH, HAS_GO_ANNOTATION, HAS_FEATURE
"""

from __future__ import annotations

import logging
from typing import Optional

from clients.uniprot import UniProtClient
from clients.interpro import InterProClient
from clients.string_db import STRINGClient
from clients.go_client import GOClient
from clients.alphafold import AlphaFoldClient
from clients.pdb import PDBClient

from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphExpansionPipeline:
    """
    Process seed protein(s) and up to N interaction partners (filled on the spot).

    For each seed:
      - Fetch UniProt → write Protein node (core fields only)
      - Fetch PDB → add pdb_ids to Protein (from PDB Search API)
      - Fetch AlphaFold → add alphafold_entry_ids, alphafold_cif_url, alphafold_pdb_url to Protein
      - Fetch InterPro → write ProteinFeature nodes + HAS_FEATURE edges
      - Fetch GO (QuickGO) → write GOTerm nodes + HAS_GO_ANNOTATION edges
      - Fetch STRING → write INTERACTS_WITH edges, enqueue partners
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        string_score_threshold: int = 700,
        max_partners_per_protein: int = 5,
    ) -> None:
        self.neo4j = neo4j
        self.string_score_threshold = string_score_threshold
        self.max_partners = max_partners_per_protein

        # Initialise API clients
        self.uniprot = UniProtClient()
        self.interpro = InterProClient()
        self.string_db = STRINGClient()
        self.go = GOClient()
        self.alphafold = AlphaFoldClient()
        self.pdb = PDBClient()

    def close(self) -> None:
        for client in (
            self.uniprot, self.interpro, self.string_db,
            self.go, self.alphafold, self.pdb,
        ):
            client.close()

    def __enter__(self) -> "GraphExpansionPipeline":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, seed_accessions: list[str]) -> dict[str, int]:
        """
        Process each seed protein and up to max_partners_per_protein interaction
        partners. Partners are filled on the spot (no BFS).
        Clears the graph before each run.
        """
        self.neo4j.clear_graph()
        self.neo4j.setup_schema()

        for i, accession in enumerate(seed_accessions):
            logger.info("Processing seed %s (%d/%d)", accession, i + 1, len(seed_accessions))
            self._process_protein(accession)

        logger.info("Pipeline complete. Processed %d seed(s).", len(seed_accessions))
        return {"proteins_processed": len(seed_accessions)}

    # ── Per-protein processing ────────────────────────────────────────────────

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
            protein_props["pdb_ids"] = pdb_result.identifiers
        except Exception as exc:
            logger.warning("PDB search failed for %s: %s", accession, exc)
            protein_props["pdb_ids"] = []

        # AlphaFold structure (from AlphaFold API)
        try:
            preds = self.alphafold.get_prediction(accession)
            protein_props["alphafold_entry_ids"] = [p.entryId for p in preds]
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
            protein_props["alphafold_cif_url"] = ""
            protein_props["alphafold_pdb_url"] = ""

        with self.neo4j.session() as s:
            self.neo4j.merge_protein(s, protein_props)

        self._write_interpro(accession)
        self._write_go_annotations(accession)

    def _fill_protein(self, accession: str) -> None:
        """Fetch UniProt entry and write full protein node (for partners)."""
        try:
            entry = self.uniprot.get_entry(accession)
        except Exception as exc:
            logger.warning("UniProt fetch failed for %s: %s", accession, exc)
            return
        self._write_protein_node(accession, entry)

    def _process_protein(self, accession: str) -> None:
        """Fetch all data for one protein, write to Neo4j, then fetch and fill up to max_partners interaction partners on the spot."""
        try:
            entry = self.uniprot.get_entry(accession)
        except Exception as exc:
            logger.warning("UniProt fetch failed for %s: %s", accession, exc)
            return

        self._write_protein_node(accession, entry)
        self._write_ppi(accession, entry)

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

    # ── STRING PPI ────────────────────────────────────────────────────────────

    def _write_ppi(self, accession: str, entry) -> None:
        """
        Fetch up to max_partners interaction partners, fill each on the spot (full node),
        then write INTERACTS_WITH edges.
        """
        string_id = entry.get_string_id()
        if not string_id:
            logger.debug("No STRING ID for %s, skipping PPI", accession)
            return

        try:
            interactions = self.string_db.get_interaction_partners(
                [string_id],
                taxon=entry.organism.taxonId,
                required_score=self.string_score_threshold,
                limit=self.max_partners,
            )
        except Exception as exc:
            logger.warning("STRING fetch failed for %s: %s", accession, exc)
            return

        with self.neo4j.session() as s:
            for iact in interactions:
                partner_name = iact.preferredName_B
                partner_string_id = iact.stringId_B

                partner_acc = self._resolve_string_to_uniprot(
                    partner_string_id, partner_name, entry.organism.taxonId
                )

                if partner_acc:
                    self._fill_protein(partner_acc)

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
