"""
ProteinChameleon Bulk Ingestion
================================
Outputs two linked CSV files:

  output/proteins.csv   — one row per reviewed UniProt protein (~570k rows)
  output/features.csv   — one row per InterPro match, linked by protein_acc

Join them in pandas:
  pd.merge(proteins, features, left_on="accession", right_on="protein_acc")

Usage:
  python bulk_ingest.py                    # run both phases
  python bulk_ingest.py --phase 1          # proteins.csv only
  python bulk_ingest.py --phase 2          # features.csv only (requires phase 1)
  python bulk_ingest.py --limit 10000      # cap proteins (for testing)
  python bulk_ingest.py --output-dir out/  # custom output directory
"""

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bulk_ingest")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk ingest UniProt proteins + InterPro features to CSV"
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap protein count in phase 1 — useful for testing",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for output CSVs (default: output/)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory for cached InterPro downloads (default: data/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_dir   = Path(args.data_dir)

    logger.info("=" * 60)
    logger.info("ProteinChameleon Bulk Ingestion")
    logger.info("=" * 60)
    logger.info("Phase:      %s", args.phase)
    logger.info("Limit:      %s", args.limit or "none (all ~570k reviewed proteins)")
    logger.info("Output dir: %s", output_dir.resolve())
    logger.info("Data dir:   %s  (InterPro download cache)", data_dir.resolve())
    logger.info("=" * 60)

    from graph.bulk_pipeline import BulkIngestionPipeline

    pipeline = BulkIngestionPipeline(output_dir=output_dir, data_dir=data_dir)
    t0 = time.monotonic()

    if args.phase in ("1", "all"):
        proteins = pipeline.run_phase1(limit=args.limit)
        logger.info("Phase 1 done: %d proteins → %s", proteins, pipeline.proteins_csv)

    if args.phase in ("2", "all"):
        unique_ipr, rows = pipeline.run_phase2()
        logger.info(
            "Phase 2 done: %d feature rows, %d unique IPR entries → %s",
            rows, unique_ipr, pipeline.features_csv,
        )

    elapsed = (time.monotonic() - t0) / 60
    logger.info("=" * 60)
    logger.info("Total wall time: %.1f min", elapsed)
    logger.info("")
    logger.info("Output files:")
    logger.info("  %s", pipeline.proteins_csv)
    logger.info("  %s", pipeline.features_csv)
    logger.info("")
    logger.info("Load in pandas:")
    logger.info('  proteins = pd.read_csv("%s")', pipeline.proteins_csv)
    logger.info('  features = pd.read_csv("%s")', pipeline.features_csv)
    logger.info('  df = pd.merge(proteins, features, left_on="accession", right_on="protein_acc")')


if __name__ == "__main__":
    main()
