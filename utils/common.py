"""Shared helpers for train / inference / monitoring. Run from repo root (/opt/airflow)."""
import glob
import json
import os
from datetime import datetime

import pandas as pd
import pyspark
from pyspark.sql.functions import col

# ---- paths (relative to repo root) --------------------------------------
FEATURE_STORE_DIR = "datamart/gold/feature_store/"
LABEL_STORE_DIR = "datamart/gold/label_store/"
PREDICTIONS_DIR = "datamart/gold/model_predictions/"
MONITORING_DIR = "datamart/gold/model_monitoring/"
MODEL_BANK_DIR = "model_bank/"
CHAMPION_FILE = MODEL_BANK_DIR + "champion.json"

ID_COLS = {"Customer_ID", "snapshot_date", "loan_id", "label", "label_def",
           "loan_start_date", "mob", "dpd"}

# A label is observed MOB_THRESHOLD months after the application (A1 config:
# 30dpd_6mob). Everything that touches labels respects this lag, so a backfill
# over historical data gives the same answers a live run would have given.
LABEL_LAG_MONTHS = 6


def get_spark(app="a2"):
    spark = pyspark.sql.SparkSession.builder.appName(app).master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def read_store(spark, folder, start=None, end=None):
    """Read every parquet partition in a gold folder, optionally filter by snapshot_date."""
    files = sorted(glob.glob(os.path.join(folder, "*")))
    if not files:
        return None
    sdf = spark.read.parquet(*files)
    if start is not None:
        sdf = sdf.filter(col("snapshot_date") >= start)
    if end is not None:
        sdf = sdf.filter(col("snapshot_date") <= end)
    return sdf


def feature_columns(pdf):
    """All numeric non-id columns are features."""
    cols, seen = [], set()
    for c in pdf.columns:
        if c in ID_COLS or c in seen:
            continue
        seen.add(c)
        series = pdf[c]
        if isinstance(series, pd.DataFrame):      # duplicate column name - skip
            continue
        if series.dtype.kind in "biuf":
            cols.append(c)
    return cols


def join_features_labels(features_pdf, labels_pdf):
    """Point-in-time join of the A1 gold stores.

    The feature store is keyed by APPLICATION month (snapshot_date).
    The label store is keyed by OBSERVATION month (application + MOB_THRESHOLD)
    and carries loan_start_date = the application month.

    So features line up with labels on feature.snapshot_date == label.loan_start_date.
    Joining both on snapshot_date matches nothing - the two dates are 6 months apart
    by design, which is exactly the label lag the monitoring has to cope with.

    Only Customer_ID, loan_start_date and label are taken from the label store:
    mob and dpd are post-origination facts and would leak the target.
    """
    f = features_pdf.copy()
    f["snapshot_date"] = pd.to_datetime(f["snapshot_date"])
    # the label store is the single source of truth for the target: drop any
    # label columns the feature side carries, or the merge yields two and
    # y comes out 2-D
    f = f.drop(columns=[c for c in ("label", "label_def") if c in f.columns])

    # the feature store has a loan_start_date of its own, so give the label side
    # a name that cannot collide
    l = labels_pdf[["Customer_ID", "loan_start_date", "label"]].copy()
    l = l.rename(columns={"loan_start_date": "_label_appl_date"})
    l["_label_appl_date"] = pd.to_datetime(l["_label_appl_date"])
    l = l.drop_duplicates(subset=["Customer_ID", "_label_appl_date"])

    out = f.merge(l, left_on=["Customer_ID", "snapshot_date"],
                  right_on=["Customer_ID", "_label_appl_date"], how="inner")
    return out.drop(columns=["_label_appl_date"])


def partition_name(prefix, date_str, ext=".parquet"):
    return f"{prefix}_{date_str.replace('-', '_')}{ext}"


def latest_model_path(model_name, snapshot_date_str):
    """Most recent artefact for model_name trained on or before snapshot_date."""
    snap = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    candidates = []
    for p in glob.glob(os.path.join(MODEL_BANK_DIR, f"{model_name}_*.pkl")):
        stamp = os.path.basename(p)[len(model_name) + 1:-4]          # YYYY_MM_DD
        try:
            d = datetime.strptime(stamp, "%Y_%m_%d")
        except ValueError:
            continue
        if d <= snap:
            candidates.append((d, p))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def read_champion():
    if os.path.exists(CHAMPION_FILE):
        with open(CHAMPION_FILE) as f:
            return json.load(f)
    return None


def write_champion(info):
    os.makedirs(MODEL_BANK_DIR, exist_ok=True)
    with open(CHAMPION_FILE, "w") as f:
        json.dump(info, f, indent=2, default=str)
