# Assignment 2 — Loan Default ML Pipeline (starter)

## Tonight's checklist
1. Copy your A1 `data/` CSVs into `data/`, your A1 pipeline scripts into `utils/`, and your populated A1 `datamart/` in.
2. Rename / point the four datamart tasks in `dags/loan_default_pipeline.py` at your real A1 scripts
   (`bronze_tables.py`, `silver_tables.py`, `gold_feature_store.py`, `gold_label_store.py` are placeholders).
   Each script must accept `--snapshotdate YYYY-MM-DD` and process that one month only.
3. Check `utils/common.py` paths match your A1 folder names (`datamart/gold/feature_store/`, `datamart/gold/label_store/`).
4. `echo "AIRFLOW_UID=$(id -u)" > .env` (Linux/Mac; skip on Windows)
5. `docker-compose build && docker-compose up` → http://localhost:8080 (admin / admin)
6. Trigger `loan_default_pipeline` once from the UI, or backfill everything:
   `docker-compose exec airflow-scheduler airflow dags backfill loan_default_pipeline -s 2023-01-01 -e 2024-12-01`

## What each piece does
| file | job |
|---|---|
| `dags/loan_default_pipeline.py` | monthly DAG: datamart → (train on `TRAIN_DATES`) → inference → monitoring → charts |
| `utils/common.py` | shared paths, spark session, store readers, champion.json helpers |
| `utils/model_train.py` | LogReg + XGBoost, 12m train / 2m OOT, saves both to `model_bank/`, writes `champion.json` |
| `utils/model_inference.py` | scores one month with the latest artefact of a model → `gold/model_predictions/` |
| `utils/model_monitoring.py` | AUC/GINI/KS/precision/recall where labels exist + PSI → `gold/model_monitoring/` |
| `utils/model_visualise.py` | performance + stability PNGs for the deck |

## Assumptions baked in (change if yours differ)
- Feature store partitions live in `datamart/gold/feature_store/` with `Customer_ID`, `snapshot_date` + numeric feature columns.
- Label store partitions live in `datamart/gold/label_store/` with `Customer_ID`, `snapshot_date`, `label` (Lab 2 format).
- Features and labels join on `Customer_ID` + `snapshot_date` (same as Lab 5).
- Training month is `2024-09-01` (`TRAIN_DATES` in the DAG). Inference/monitoring skip cleanly for months before any model exists.
