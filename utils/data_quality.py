"""
Pipeline validation.

Cheap insurance. Each check asserts something the pipeline promises, and
main.py exits non-zero if any fail — so a scheduler treats a bad run as a
failure rather than a success that printed warnings.
"""

import os
from pyspark.sql import functions as F

from utils import config

# Loan-ledger columns describing post-origination performance. If any of these
# ever appear in the feature store, the model is being handed the answer.
FORBIDDEN_FEATURE_COLS = {
    "paid_amt", "overdue_amt", "balance", "due_amt", "dpd",
    "installments_missed", "first_missed_date", "installment_num", "mob", "label",
}


def _read_gold(spark, path: str):
    return spark.read.parquet(os.path.join(path, "*.parquet"))


def _report(name: str, passed: bool, detail: str) -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name:38s} {detail}")
    return passed


def validate(spark) -> bool:
    print("\n[4/4] VALIDATION")

    features = _read_gold(spark, config.GOLD_FEATURE_STORE)
    labels   = _read_gold(spark, config.GOLD_LABEL_STORE)
    results  = []

    # --- grain ------------------------------------------------------------
    feature_rows  = features.count()
    unique_loans  = features.select("loan_id").distinct().count()
    results.append(_report("feature store is one row per loan",
                           feature_rows == unique_loans,
                           f"{feature_rows:,} rows / {unique_loans:,} loan_ids"))

    label_rows     = labels.count()
    unique_labeled = labels.select("loan_id").distinct().count()
    results.append(_report("label store is one row per loan",
                           label_rows == unique_labeled,
                           f"{label_rows:,} rows / {unique_labeled:,} loan_ids"))

    # --- leakage ----------------------------------------------------------
    leaked = FORBIDDEN_FEATURE_COLS.intersection(set(features.columns))
    results.append(_report("no post-origination columns leaked", len(leaked) == 0,
                           "none found" if not leaked else f"LEAKED: {sorted(leaked)}"))

    # every feature row must be dated at its own loan's origination
    misdated = features.filter(F.col("snapshot_date") != F.col("loan_start_date")).count()
    results.append(_report("features dated at application", misdated == 0,
                           f"{misdated:,} rows dated away from loan_start_date"))

    # labels must be observed strictly AFTER the application they describe
    joined = labels.alias("l").join(
        features.select("loan_id", F.col("snapshot_date").alias("feature_date")),
        on="loan_id", how="inner")
    early = joined.filter(F.col("l.snapshot_date") <= F.col("feature_date")).count()
    results.append(_report("labels observed after features", early == 0,
                           f"{early:,} labels not strictly in the future"))

    # --- joinability ------------------------------------------------------
    matched = joined.count()
    results.append(_report("feature/label join coverage",
                           matched >= 0.99 * min(feature_rows, label_rows),
                           f"{matched:,} loans match on loan_id"))

    # --- label sanity -----------------------------------------------------
    default_rate = labels.agg(F.avg("label")).collect()[0][0]
    results.append(_report("label is non-degenerate",
                           0.01 < default_rate < 0.99,
                           f"default rate {default_rate:.2%}"))

    # --- key integrity ----------------------------------------------------
    null_keys = features.filter(
        F.col("loan_id").isNull() | F.col("Customer_ID").isNull()).count()
    results.append(_report("no null keys in feature store", null_keys == 0,
                           f"{null_keys:,} null keys"))

    # --- PII --------------------------------------------------------------
    pii = {"Name", "SSN", "name", "ssn"}.intersection(set(features.columns))
    results.append(_report("no direct identifiers in features", len(pii) == 0,
                           "none found" if not pii else f"PII PRESENT: {sorted(pii)}"))

    print(f"\n  {sum(results)}/{len(results)} checks passed")
    return all(results)