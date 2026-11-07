"""
Score one month's feature store with the latest artefact of --modelname trained on or before that month.
Writes datamart/gold/model_predictions/<model>/<model>_predictions_YYYY_MM_DD.parquet

    python3 utils/model_inference.py --snapshotdate "2024-10-01" --modelname xgb
"""
import argparse
import os
import pickle
from datetime import datetime

import pandas as pd

import common


def main(snapshotdate, modelname):
    print("\n---starting inference job---\n")
    model_path = common.latest_model_path(modelname, snapshotdate)
    if model_path is None:
        print(f"no {modelname} artefact trained on/before {snapshotdate} - nothing to score, exiting cleanly")
        return
    with open(model_path, "rb") as f:
        art = pickle.load(f)
    print("loaded", model_path)

    spark = common.get_spark("inference")
    snap = datetime.strptime(snapshotdate, "%Y-%m-%d")
    feats = common.read_store(spark, common.FEATURE_STORE_DIR, snap, snap)
    if feats is None or feats.count() == 0:
        print("no features for", snapshotdate, "- exiting cleanly")
        spark.stop()
        return
    pdf = feats.toPandas()

    X = pdf[art["feature_cols"]]
    X = art["preprocessing_transformers"]["preprocessor"].transform(X)
    p = art["model"].predict_proba(X)[:, 1]

    out = pdf[["Customer_ID", "snapshot_date"]].copy()
    out["model_name"] = art["model_name"]
    out["model_version"] = art["model_version"]
    out["model_predictions"] = p

    out_dir = os.path.join(common.PREDICTIONS_DIR, modelname)
    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, common.partition_name(f"{modelname}_predictions", snapshotdate))
    spark.createDataFrame(out).write.mode("overwrite").parquet(fp)
    print(f"scored {len(out)} rows -> {fp}")

    spark.stop()
    print("\n---completed inference job---\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshotdate", required=True)
    ap.add_argument("--modelname", required=True)
    a = ap.parse_args()
    main(a.snapshotdate, a.modelname)
