"""
Run ONE Assignment 1 datamart layer for ONE snapshot month.

    python3 utils/run_layer.py --layer bronze        --snapshotdate 2023-01-01
    python3 utils/run_layer.py --layer silver        --snapshotdate 2023-01-01
    python3 utils/run_layer.py --layer gold_features --snapshotdate 2023-01-01
    python3 utils/run_layer.py --layer gold_labels   --snapshotdate 2023-01-01

Two adaptations of the A1 code, both deliberate:

1. The A1 layers take a LIST of snapshot dates and loop. We hand them a list of
   one, built with A1's own generate_snapshot_dates so the date objects match.

2. A1 calls ensure_clean_dir() on each output folder, which deletes the whole
   table before writing. That is right for a single batch run over all months;
   it is wrong for a monthly pipeline, where run N would wipe runs 1..N-1.
   We swap it for a create-if-missing so each run owns only its own partition
   file. Partition names carry the date, so re-running a month overwrites just
   that month - idempotent, which is what backfill needs.
"""
import argparse
import os
import sys

# this file lives in utils/, but imports the utils package - put the repo root on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import config
from utils import bronze_processing, silver_processing, gold_feature_store, gold_label_store
from utils.spark_utils import generate_snapshot_dates, get_spark


def _keep_dir(path):
    """Non-destructive stand-in for A1's ensure_clean_dir."""
    os.makedirs(path, exist_ok=True)


# each module did `from utils.spark_utils import ensure_clean_dir`, so the name
# is bound in that module's namespace - patch each one
for _module in (bronze_processing, silver_processing, gold_feature_store, gold_label_store):
    _module.ensure_clean_dir = _keep_dir


def main(layer, snapshotdate):
    print(f"\n--- {layer} | {snapshotdate} ---\n")
    dates = generate_snapshot_dates(snapshotdate, snapshotdate)
    if not dates:
        raise SystemExit(f"generate_snapshot_dates returned nothing for {snapshotdate}")
    print("processing:", dates)

    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        if layer == "bronze":
            os.makedirs(config.BRONZE_DIR, exist_ok=True)
            bronze_processing.run_bronze_layer(spark, dates)
        elif layer == "silver":
            os.makedirs(config.SILVER_DIR, exist_ok=True)
            silver_processing.run_silver_layer(spark, dates)
        elif layer == "gold_features":
            os.makedirs(config.GOLD_DIR, exist_ok=True)
            gold_feature_store.build_feature_store(spark, dates)
        elif layer == "gold_labels":
            os.makedirs(config.GOLD_DIR, exist_ok=True)
            gold_label_store.build_label_store(spark, dates)
        else:
            raise SystemExit(f"unknown layer: {layer}")
    finally:
        spark.stop()

    print(f"\n--- {layer} | {snapshotdate} done ---\n")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", required=True,
                    choices=["bronze", "silver", "gold_features", "gold_labels"])
    ap.add_argument("--snapshotdate", required=True, help="YYYY-MM-DD")
    a = ap.parse_args()
    sys.exit(main(a.layer, a.snapshotdate))
