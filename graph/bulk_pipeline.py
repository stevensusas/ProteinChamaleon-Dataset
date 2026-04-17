"""
Bulk ingestion pipeline — outputs two linked CSV files.

Phase 1 — proteins.csv
  Streams all ~570k reviewed proteins from UniProt via cursor-paginated REST API.
  One row per protein. PDB and AlphaFold IDs are extracted from UniProt
  cross-references at no extra API cost.

Phase 2 — features.csv
  Downloads two flat files from the InterPro FTP (cached to `data_dir`):
    entry.list          (~2 MB)  IPR accession → type + name
    protein2ipr.dat.gz  (~4 GB)  UniProt accession → InterPro matches
  Streams protein2ipr.dat.gz line-by-line, emitting one row per match
  only for proteins present in proteins.csv.

The two files are linked by protein_acc == accession:
  pd.merge(proteins, features, left_on="accession", right_on="protein_acc")

Runtime estimates:
  Phase 1:  ~20-30 min  (1140 HTTP requests, streaming writes)
  Phase 2:  ~60-90 min  (4 GB download + stream parse)
"""

from __future__ import annotations

import csv
import gzip
import logging
import time
from pathlib import Path
from typing import Optional

import httpx

from clients.uniprot import UniProtClient, UniProtEntry

logger = logging.getLogger(__name__)

_ENTRY_LIST_URL   = "https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/entry.list"
_PROTEIN2IPR_URL  = "https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/protein2ipr.dat.gz"

# protein2ipr.dat column indices (0-based, tab-separated, 6 columns)
_COL_UNIPROT = 0
_COL_IPR     = 1
_COL_NAME    = 2
# col 3 = member-DB method accession (skipped)
_COL_START   = 4
_COL_END     = 5

PROTEIN_COLUMNS = [
    "accession", "name", "gene_name", "organism", "taxon_id",
    "length", "mol_weight", "is_reviewed",
    "function_text", "ec_numbers", "pdb_ids", "alphafold_entry_ids",
    "sequence",
]

FEATURE_COLUMNS = [
    "protein_acc", "ipr_acc", "ipr_name", "ipr_type", "start", "end",
]


def _entry_to_row(entry: UniProtEntry) -> dict:
    return {
        "accession":           entry.primaryAccession,
        "name":                entry.protein_name or "",
        "gene_name":           entry.gene_name or "",
        "organism":            entry.organism.scientificName,
        "taxon_id":            entry.organism.taxonId,
        "length":              entry.sequence.length,
        "mol_weight":          entry.sequence.molWeight,
        "is_reviewed":         entry.is_reviewed,
        "function_text":       entry.get_function_text() or "",
        # lists serialised as "|"-separated strings
        "ec_numbers":          "|".join(entry.get_ec_numbers()),
        "pdb_ids":             "|".join(entry.get_pdb_ids()),
        "alphafold_entry_ids": "|".join(entry.get_alphafold_ids()),
        "sequence":            entry.sequence.value,
    }


class BulkIngestionPipeline:
    def __init__(
        self,
        output_dir: str | Path = "output",
        data_dir:   str | Path = "data",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.data_dir   = Path(data_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.proteins_csv = self.output_dir / "proteins.csv"
        self.features_csv = self.output_dir / "features.csv"

    # ── Phase 1: UniProt streaming ────────────────────────────────────────────

    def run_phase1(self, limit: Optional[int] = None) -> int:
        """
        Stream all reviewed UniProt proteins and write proteins.csv.
        Returns the number of proteins written.
        """
        logger.info("Phase 1: streaming reviewed UniProt proteins → %s", self.proteins_csv)
        logger.info("  limit: %s", limit or "none (all ~570k reviewed proteins)")

        client = UniProtClient(rate_limit_delay=0.1)
        total  = 0
        t0     = time.monotonic()

        try:
            with open(self.proteins_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=PROTEIN_COLUMNS)
                writer.writeheader()

                for entry in client.search("reviewed:true", size=500):
                    writer.writerow(_entry_to_row(entry))
                    total += 1

                    if total % 10_000 == 0:
                        elapsed = time.monotonic() - t0
                        logger.info(
                            "  %d proteins written  (%.0f/s, %.1f min elapsed)",
                            total, total / elapsed, elapsed / 60,
                        )

                    if limit and total >= limit:
                        break
        finally:
            client.close()

        elapsed = time.monotonic() - t0
        logger.info(
            "Phase 1 complete: %d proteins in %.1f min (%.0f/s)",
            total, elapsed / 60, total / max(elapsed, 1),
        )
        return total

    # ── Phase 2: InterPro flat-file ───────────────────────────────────────────

    def run_phase2(self) -> tuple[int, int]:
        """
        Download InterPro flat files (if not cached) and write features.csv.
        Returns (unique_ipr_entries_seen, total_rows_written).
        """
        if not self.proteins_csv.exists():
            raise FileNotFoundError(
                f"{self.proteins_csv} not found — run phase 1 first."
            )

        entry_list_path  = self.data_dir / "entry.list"
        protein2ipr_path = self.data_dir / "protein2ipr.dat.gz"

        self._download_if_missing(entry_list_path,  _ENTRY_LIST_URL,  "entry.list")
        self._download_if_missing(protein2ipr_path, _PROTEIN2IPR_URL, "protein2ipr.dat.gz")

        logger.info("Loading IPR entry metadata from entry.list...")
        ipr_meta = self._load_entry_list(entry_list_path)

        logger.info("Loading ingested protein accessions from %s...", self.proteins_csv)
        ingested = self._load_accessions(self.proteins_csv)
        logger.info("  %d proteins to match against", len(ingested))

        logger.info("Phase 2: streaming protein2ipr.dat.gz → %s", self.features_csv)
        rows, unique_ipr = self._stream_features(protein2ipr_path, ipr_meta, ingested)

        logger.info(
            "Phase 2 complete: %d rows written, %d unique IPR entries seen",
            rows, unique_ipr,
        )
        return unique_ipr, rows

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _download_if_missing(self, dest: Path, url: str, label: str) -> None:
        if dest.exists():
            size_mb = dest.stat().st_size / 1_048_576
            logger.info("Cache hit: %s (%.1f MB) — skipping download", label, size_mb)
            return

        logger.info("Downloading %s ...", label)
        tmp = dest.with_suffix(".tmp")
        t0  = time.monotonic()
        downloaded = 0

        with httpx.stream("GET", url, follow_redirects=True, timeout=300) as resp:
            resp.raise_for_status()
            total_bytes = int(resp.headers.get("content-length", 0))
            with open(tmp, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (100 << 20) == 0:
                        pct = f"{100 * downloaded / total_bytes:.0f}%" if total_bytes else f"{downloaded >> 20} MB"
                        logger.info("  %s: %s", label, pct)

        tmp.rename(dest)
        logger.info("Downloaded %s in %.1f min", label, (time.monotonic() - t0) / 60)

    def _load_entry_list(self, path: Path) -> dict[str, dict]:
        """Parse entry.list → {ipr_acc: {name, type}}."""
        meta: dict[str, dict] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                meta[parts[0]] = {"name": parts[2], "type": parts[1]}
        logger.info("  %d IPR entries loaded", len(meta))
        return meta

    def _load_accessions(self, csv_path: Path) -> set[str]:
        """Read the accession column from proteins.csv into a set."""
        accessions: set[str] = set()
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                accessions.add(row["accession"])
        return accessions

    def _stream_features(
        self,
        gz_path:  Path,
        ipr_meta: dict[str, dict],
        ingested: set[str],
    ) -> tuple[int, int]:
        """
        Stream protein2ipr.dat.gz and write features.csv.
        Returns (rows_written, unique_ipr_accessions_seen).
        """
        rows       = 0
        skipped    = 0
        seen_ipr:  set[str] = set()
        t0 = time.monotonic()

        with open(self.features_csv, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=FEATURE_COLUMNS)
            writer.writeheader()

            with gzip.open(gz_path, "rt", encoding="utf-8") as gz_f:
                for line_num, line in enumerate(gz_f, 1):
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) <= _COL_END:
                        skipped += 1
                        continue

                    protein_acc = parts[_COL_UNIPROT]
                    ipr_acc     = parts[_COL_IPR]

                    if protein_acc not in ingested or ipr_acc not in ipr_meta:
                        skipped += 1
                        continue

                    try:
                        start = int(parts[_COL_START])
                        end   = int(parts[_COL_END])
                    except ValueError:
                        skipped += 1
                        continue

                    entry = ipr_meta[ipr_acc]
                    writer.writerow({
                        "protein_acc": protein_acc,
                        "ipr_acc":     ipr_acc,
                        "ipr_name":    entry["name"],
                        "ipr_type":    entry["type"],
                        "start":       start,
                        "end":         end,
                    })
                    rows += 1
                    seen_ipr.add(ipr_acc)

                    if line_num % 5_000_000 == 0:
                        elapsed = time.monotonic() - t0
                        logger.info(
                            "  %dM lines parsed, %d rows written (%.1f min)",
                            line_num // 1_000_000, rows, elapsed / 60,
                        )

        logger.info("  %d lines skipped (protein not in graph or IPR unknown)", skipped)
        return rows, len(seen_ipr)
