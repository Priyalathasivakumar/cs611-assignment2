"""
Train LogReg + XGBoost on a 12-month window, evaluate on the 2 months before snapshotdate (OOT),
save every model to model_bank/ and point champion.json at the best OOT AUC.

    python3 utils/model_train.py --snapshotdate "2024-09-01"
"""
import argparse
import json
import os
import pickle
import pprint
from datetime import datetime, timedelta

import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import common

TRAIN_MONTHS = 12
OOT_MONTHS = 2
TRAIN_TEST_RATIO = 0.8
SEED = 88


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def build_models():
    """Return {name: (estimator, param_dist or None)}."""
    return {
        "logreg": (LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED), None),
        "xgb": (
            xgb.XGBClassifier(eval_metric="logloss", random_state=SEED),
            {
                "n_estimators": [25, 50, 100],
                "max_depth": [2, 3, 4],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "min_child_weight": [1, 3, 5],
                "reg_lambda": [1, 1.5, 2],
            },
        ),
    }


def main(snapshotdate):
    print("\n---starting training job---\n")
    spark = common.get_spark("train")

    # ---- date windows ----
    # Lab 5 puts the OOT window in the 2 months before the train date. We shift
    # everything back by LABEL_LAG_MONTHS: on train date T the newest label the
    # bank actually holds is for applications from T-6, so the windows end there.
    T = datetime.strptime(snapshotdate, "%Y-%m-%d")
    last_labelled_month = T - relativedelta(months=common.LABEL_LAG_MONTHS)
    cfg = {"model_train_date_str": snapshotdate, "model_train_date": T,
           "label_lag_months": common.LABEL_LAG_MONTHS}
    cfg["oot_end_date"] = last_labelled_month + relativedelta(months=1) - timedelta(days=1)
    cfg["oot_start_date"] = last_labelled_month - relativedelta(months=OOT_MONTHS - 1)
    cfg["train_test_end_date"] = cfg["oot_start_date"] - timedelta(days=1)
    cfg["train_test_start_date"] = cfg["oot_start_date"] - relativedelta(months=TRAIN_MONTHS)
    pprint.pprint(cfg)

    # ---- load gold stores ----
    # features: filtered by APPLICATION month.
    # labels: only those OBSERVED on or before the train date (their snapshot_date
    # is the observation month), then joined on loan_start_date. This is what makes
    # a backfilled training run identical to what a live run would have seen.
    feats = common.read_store(spark, common.FEATURE_STORE_DIR,
                              cfg["train_test_start_date"], cfg["oot_end_date"])
    labels = common.read_store(spark, common.LABEL_STORE_DIR, end=T)
    if labels is None or feats is None:
        raise SystemExit("label store or feature store is empty - run the datamart tasks first")

    data = common.join_features_labels(feats.toPandas(), labels.toPandas())
    print("joined rows:", len(data))
    if len(data) == 0:
        raise SystemExit("no labelled applications in this window - pick a later train date")
    # the data may start after train_test_start_date; record what was really used
    cfg["actual_first_month"] = str(data["snapshot_date"].min().date())
    cfg["actual_last_month"] = str(data["snapshot_date"].max().date())
    print("application months used:", cfg["actual_first_month"], "->", cfg["actual_last_month"])

    feature_cols = common.feature_columns(data)
    print(f"{len(feature_cols)} feature columns")

    oot = data[data["snapshot_date"] >= cfg["oot_start_date"]]
    tt = data[data["snapshot_date"] <= cfg["train_test_end_date"]]
    X_train, X_test, y_train, y_test = train_test_split(
        tt[feature_cols], tt["label"], test_size=1 - TRAIN_TEST_RATIO,
        random_state=SEED, shuffle=True, stratify=tt["label"],
    )
    X_oot, y_oot = oot[feature_cols], oot["label"]
    print("train/test/oot:", len(X_train), len(X_test), len(X_oot), "| default rate:", round(y_train.mean(), 3))

    # ---- preprocessing fitted on TRAIN ONLY (no test/oot contamination) ----
    # the A1 feature store leaves genuine gaps as NaN and flags them with
    # *_is_missing indicators, so impute the median and keep the flags as features
    preprocessor = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]).fit(X_train)
    Xtr = preprocessor.transform(X_train)
    Xte = preprocessor.transform(X_test)
    Xoot = preprocessor.transform(X_oot)

    # ---- fit each model ----
    os.makedirs(common.MODEL_BANK_DIR, exist_ok=True)
    results = {}
    for name, (est, params) in build_models().items():
        print(f"\n=== {name} ===")
        if params:
            search = RandomizedSearchCV(est, params, n_iter=30, cv=3, scoring="roc_auc",
                                        random_state=SEED, n_jobs=-1, verbose=0)
            search.fit(Xtr, y_train)
            model, hp = search.best_estimator_, search.best_params_
        else:
            model, hp = est.fit(Xtr, y_train), est.get_params()

        p_tr, p_te, p_oot = (model.predict_proba(X)[:, 1] for X in (Xtr, Xte, Xoot))
        metrics = {
            "auc_train": roc_auc_score(y_train, p_tr),
            "auc_test": roc_auc_score(y_test, p_te),
            "auc_oot": roc_auc_score(y_oot, p_oot),
            "ks_oot": ks_stat(y_oot, p_oot),
        }
        metrics["gini_oot"] = 2 * metrics["auc_oot"] - 1
        pprint.pprint(metrics)

        version = f"{name}_{snapshotdate.replace('-', '_')}"
        artefact = {
            "model": model, "model_name": name, "model_version": version,
            "preprocessing_transformers": {"preprocessor": preprocessor},
            "feature_cols": feature_cols, "data_dates": cfg,
            "data_stats": {"n_train": len(X_train), "n_test": len(X_test), "n_oot": len(X_oot),
                           "default_rate_train": float(y_train.mean())},
            "results": metrics, "hp_params": hp,
            # score distribution baseline for PSI in monitoring
            "psi_baseline": np.histogram(p_tr, bins=10, range=(0, 1))[0].tolist(),
        }
        with open(os.path.join(common.MODEL_BANK_DIR, version + ".pkl"), "wb") as f:
            pickle.dump(artefact, f)
        with open(os.path.join(common.MODEL_BANK_DIR, version + ".json"), "w") as f:
            json.dump({k: v for k, v in artefact.items() if k not in ("model", "preprocessing_transformers")},
                      f, indent=2, default=str)
        results[name] = metrics
        print("saved", version)

    # ---- champion = best OOT AUC ----
    champ = max(results, key=lambda n: results[n]["auc_oot"])
    common.write_champion({
        "model_name": champ,
        "model_version": f"{champ}_{snapshotdate.replace('-', '_')}",
        "train_date": snapshotdate,
        "auc_oot": results[champ]["auc_oot"],
        "all_results": results,
    })
    print(f"\nchampion: {champ}  (OOT AUC {results[champ]['auc_oot']:.4f})")

    spark.stop()
    print("\n---completed training job---\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshotdate", required=True, help="YYYY-MM-DD")
    main(ap.parse_args().snapshotdate)
