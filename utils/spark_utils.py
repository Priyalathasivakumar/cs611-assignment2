"""Shared helpers: Spark session and snapshot-date handling."""

import os
import shutil
from datetime import date, datetime

from pyspark.sql import SparkSession


def get_spark(app_name: str = "cs611_assignment1") -> SparkSession:
    """Create a local Spark session tuned for this workload.

    shuffle.partitions defaults to 200. On data this small that means every
    groupBy writes 200 near-empty files and the job spends all its time on
    scheduling overhead. 8 is plenty here.
    """
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")                        # use all available cores
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def generate_snapshot_dates(start_date: str, end_date: str):
    """Every month-start date in [start, end] inclusive.

    These are the partition keys the pipeline loops over — one per monthly
    batch run.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end   = datetime.strptime(end_date,   "%Y-%m-%d").date()

    dates = []
    year, month = start.year, start.month
    while date(year, month, 1) <= end:
        dates.append(date(year, month, 1))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return dates


def partition_name(prefix: str, snapshot_date: date, extension: str) -> str:
    """e.g. bronze_lms_loan_daily_2023_01_01.csv"""
    return f"{prefix}_{snapshot_date.strftime('%Y_%m_%d')}.{extension}"


def ensure_clean_dir(path: str) -> None:
    """Create a directory, clearing it first so reruns stay idempotent.

    Without this, changing a rule and re-running leaves stale partitions
    behind — and the table silently mixes two definitions.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)