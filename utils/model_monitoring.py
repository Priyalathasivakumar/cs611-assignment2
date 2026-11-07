"""
Monitor --modelname on run month M, using only what the bank holds on that date:

  STABILITY  for month M        - PSI of this month's scores vs the training
                                  baseline. Needs no labels, so it is available
                                  the moment inference finishes.
  PERFORMANCE for month M-6     - AUC / GINI / KS / precision / recall for the
                                  applications scored six months ago, whose
                                  labels have just matured (label store rows
                                  observed at M carry loan_start_date = M-6).

So a month's PSI appears immediately and its AUC appears six months later,
which is exactly the lag a live credit model lives with.

Writes
  datamart/gold/model_monitoring/<model>/stability/stability_<M>.parquet
  datamart/gold/model_monitoring/<model>/performance/performance_<M-6>.parquet

    python3 utils/model_monitoring.py --snapshotdate 2024-10-01 --modelname xgb
"""
import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

import common

THRESHOLD = 0.5


def psi(expected_counts, actual_scores, bins=10, eps=1e-4):
    exp = np.asarray(expected_counts, dtype=float)
    act = np.histogram(actual_scores, bins=bins, range=(0, 1))[0].astype(float)
    exp = exp / max(exp.sum(), 1) + eps
    act = act / max(act.sum(), 1) + eps
    return float(np.sum((act - exp) * np.log(act / exp)))


def load_predictions(spark, modelname, month):
    fp = os.path.join(common.PREDICTIONS_DIR, modelname,
                      common.partition_name(f"{modelname}_predictions", month.strftime("%Y-%m-%d")))
    if not os.path.exists(fp):
        return None
    return spark.read.parquet(fp).toPandas()


def load_artefact(version):
    with open(os.path.join(common.MODEL_BANK_DIR, version + ".pkl"), "rb") as f:
        return pickle.load(f)


def main(snapshotdate, modelname):
    print("\n---starting monitoring job---\n")
    spark = common.get_spark("monitoring")
    M = datetime.strptime(snapshotdate, "%Y-%m-%d")
    out_root = os.path.join(common.MONITORING_DIR, modelname)

    # ---- 1. stability for month M ----
    preds_M = load_predictions(spark, modelname, M)
    if preds_M is None:
        print("no predictions for", snapshotdate, "- model not live yet, nothing to monitor")
    else:
        art = load_artefact(preds_M["model_version"].iloc[0])
        row = {
            "snapshot_date": M.date(), "model_name": modelname,
            "model_version": art["model_version"], "n_scored": int(len(preds_M)),
            "mean_score": float(preds_M["model_predictions"].mean()),
            "psi_score": psi(art["psi_baseline"], preds_M["model_predictions"].values),
        }
        d = os.path.join(out_root, "stability"); os.makedirs(d, exist_ok=True)
        fp = os.path.join(d, common.partition_name("stability", snapshotdate))
        pd.DataFrame([row]).to_parquet(fp, index=False)
        print("stability:", row)

    # ---- 2. performance for month A = M - lag, whose labels matured at M ----
    A = M - relativedelta(months=common.LABEL_LAG_MONTHS)
    preds_A = load_predictions(spark, modelname, A)
    labels_M = common.read_store(spark, common.LABEL_STORE_DIR, M, M)   # observed this month
    if preds_A is None:
        print(f"no predictions for {A:%Y-%m} - nothing to evaluate yet")
    elif labels_M is None or labels_M.count() == 0:
        print(f"no labels observed at {M:%Y-%m} - cannot evaluate {A:%Y-%m}")
    else:
        art = load_artefact(preds_A["model_version"].iloc[0])
        j = common.join_features_labels(preds_A, labels_M.toPandas())
        row = {
            "snapshot_date": A.date(), "labels_observed_at": M.date(),
            "model_name": modelname, "model_version": art["model_version"],
            "n_scored": int(len(preds_A)), "n_labelled": int(len(j)),
            "label_rate": None, "auc": None, "gini": None, "ks": None,
            "precision": None, "recall": None,
            "auc_oot_baseline": float(art["results"]["auc_oot"]),
        }
        if len(j) > 0 and j["label"].nunique() == 2:
            y, p = j["label"].values, j["model_predictions"].values
            fpr, tpr, _ = roc_curve(y, p)
            row.update({
                "label_rate": float(y.mean()), "auc": float(roc_auc_score(y, p)),
                "ks": float(np.max(tpr - fpr)),
                "precision": float(precision_score(y, p >= THRESHOLD, zero_division=0)),
                "recall": float(recall_score(y, p >= THRESHOLD, zero_division=0)),
            })
            row["gini"] = 2 * row["auc"] - 1
        d = os.path.join(out_root, "performance"); os.makedirs(d, exist_ok=True)
        fp = os.path.join(d, common.partition_name("performance", A.strftime("%Y-%m-%d")))
        pd.DataFrame([row]).to_parquet(fp, index=False)
        print("performance:", row)

    spark.stop()
    print("\n---completed monitoring job---\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshotdate", required=True)
    ap.add_argument("--modelname", required=True)
    a = ap.parse_args()
    main(a.snapshotdate, a.modelname)
