"""
Assignment 2 - end-to-end monthly ML pipeline.

datamart (feature + label store)  ->  [train on TRAIN_DATES only]  ->  inference  ->  monitoring  ->  visualise

Every task gets the logical date via {{ ds }} so a backfill replays one month per run:
    airflow dags backfill loan_default_pipeline -s 2023-01-01 -e 2024-12-01
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule

# ---- config --------------------------------------------------------------
# Labels mature 6 months after application. On train date T the bank only holds
# labels for applications up to T-6, and every script respects that, so the
# backfill reproduces what a live run would have seen (no hindsight).
#   T = 2024-04-01  ->  labelled applications up to 2023-10
#                       train 2023-01..2023-08, OOT 2023-09..2023-10
#                       live from 2024-04; PSI for 2024-04..12 immediately,
#                       AUC for 2024-04..06 once their labels mature (2024-10..12)
TRAIN_DATES = ["2024-04-01"]          # months on which the training task actually runs
MODELS = ["logreg", "xgb"]            # model names used by train / inference / monitoring scripts

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

PY = "cd /opt/airflow && python3"     # all scripts run from repo root so datamart/ and model_bank/ resolve


def _should_train(**ctx):
    return "train_models" if ctx["ds"] in TRAIN_DATES else "skip_training"


with DAG(
    dag_id="loan_default_pipeline",
    default_args=default_args,
    description="Monthly loan-default ML pipeline: datamart -> train -> inference -> monitoring",
    schedule_interval="0 0 1 * *",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
    max_active_runs=1,
) as dag:

    # ---- 1. datamart (Assignment 1 code, one month per run) ------------
    bronze = BashOperator(
        task_id="bronze_tables",
        bash_command=f'{PY} utils/run_layer.py --layer bronze --snapshotdate "{{{{ ds }}}}"',
    )
    silver = BashOperator(
        task_id="silver_tables",
        bash_command=f'{PY} utils/run_layer.py --layer silver --snapshotdate "{{{{ ds }}}}"',
    )
    gold_features = BashOperator(
        task_id="gold_feature_store",
        bash_command=f'{PY} utils/run_layer.py --layer gold_features --snapshotdate "{{{{ ds }}}}"',
    )
    gold_labels = BashOperator(
        task_id="gold_label_store",
        bash_command=f'{PY} utils/run_layer.py --layer gold_labels --snapshotdate "{{{{ ds }}}}"',
    )
    datamart_done = EmptyOperator(task_id="datamart_completed")

    bronze >> silver >> [gold_features, gold_labels] >> datamart_done

    # ---- 2. training (only on TRAIN_DATES) -----------------------------
    train_branch = BranchPythonOperator(task_id="check_train_month", python_callable=_should_train)
    skip_training = EmptyOperator(task_id="skip_training")

    train_models = BashOperator(
        task_id="train_models",
        bash_command=f'{PY} utils/model_train.py --snapshotdate "{{{{ ds }}}}"',
    )
    training_done = EmptyOperator(task_id="training_completed", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    datamart_done >> train_branch >> [train_models, skip_training] >> training_done

    # ---- 3. inference --------------------------------------------------
    inference_tasks = [
        BashOperator(
            task_id=f"inference_{m}",
            bash_command=f'{PY} utils/model_inference.py --snapshotdate "{{{{ ds }}}}" --modelname {m}',
        )
        for m in MODELS
    ]
    inference_done = EmptyOperator(task_id="inference_completed")
    training_done >> inference_tasks >> inference_done

    # ---- 4. monitoring -------------------------------------------------
    monitor_tasks = [
        BashOperator(
            task_id=f"monitor_{m}",
            bash_command=f'{PY} utils/model_monitoring.py --snapshotdate "{{{{ ds }}}}" --modelname {m}',
        )
        for m in MODELS
    ]
    monitoring_done = EmptyOperator(task_id="monitoring_completed")
    inference_done >> monitor_tasks >> monitoring_done

    # ---- 5. charts -----------------------------------------------------
    visualise = BashOperator(
        task_id="visualise_monitoring",
        bash_command=f"{PY} utils/model_visualise.py",
    )
    monitoring_done >> visualise
