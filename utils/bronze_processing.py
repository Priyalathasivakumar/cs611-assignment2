"""
Bronze layer — raw integration.

Bronze lands the source extracts as-is and partitions them by month.

Deliberately NOT done here:
  * no type casting  — every column stays a string, so a malformed value
                       like '52312.68_' is captured rather than silently
                       nulled by a reader expecting a number
  * no cleaning      — bronze is the audit trail
  * no deduplication — row counts must reconcile against source
"""

import os
from pyspark.sql import functions as F

from utils import config
from utils.spark_utils import ensure_clean_dir, partition_name


def _read_raw_as_strings(spark, source_path: str):
    """Read a CSV with every column as a string.

    inferSchema is OFF on purpose: it reads the file twice AND coerces dirty
    values to NULL before we ever get to see them.
    """
    return (spark.read
            .option("header", True)
            .option("inferSchema", False)
            .csv(source_path))


def build_bronze_table(spark, table_name: str, snapshot_dates) -> int:
    """Ingest one raw source into a partitioned bronze table."""
    source_path = config.RAW_SOURCES[table_name]
    output_dir  = config.BRONZE_TABLES[table_name]
    ensure_clean_dir(output_dir)

    raw_df = _read_raw_as_strings(spark, source_path)
    raw_df.cache()            # reused once per month — keep it in memory

    total_rows = 0
    partitions_written = 0

    # --- one partition per month ------------------------------------------
    for snapshot_date in snapshot_dates:
        stamp = snapshot_date.strftime("%Y-%m-%d")
        partition_df = raw_df.filter(F.col("snapshot_date") == stamp)

        row_count = partition_df.count()
        if row_count == 0:
            # Not every source covers every month (clickstream stops at
            # 2024-12). Skipping empty months keeps the datamart honest.
            continue

        out_path = os.path.join(
            output_dir, partition_name(f"bronze_{table_name}", snapshot_date, "csv"))
        partition_df.toPandas().to_csv(out_path, index=False)

        total_rows += row_count
        partitions_written += 1

    source_rows = raw_df.count()
    raw_df.unpersist()

    print(f"  [bronze] {table_name:22s} {partitions_written:3d} partitions | "
          f"{total_rows:>7,} rows (source: {source_rows:,})")

    # --- reconciliation ---------------------------------------------------
    # If these disagree, rows carried a snapshot_date outside our window.
    # Better to surface it here than discover it in gold.
    # Only meaningful on a full-window batch: in the monthly Airflow DAG each
    # run processes one month, so it would always "disagree" with the whole file.
    if len(snapshot_dates) > 1 and total_rows != source_rows:
        print(f"      WARNING: {source_rows - total_rows:,} rows fell outside "
              f"{config.START_DATE}..{config.END_DATE}")

    return total_rows


def run_bronze_layer(spark, snapshot_dates) -> None:
    """Build every bronze table."""
    print("\n[1/4] BRONZE LAYER - raw integration")
    os.makedirs(config.BRONZE_DIR, exist_ok=True)

    for table_name in config.RAW_SOURCES:
        build_bronze_table(spark, table_name, snapshot_dates)