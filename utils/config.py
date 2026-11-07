
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
DATAMART_DIR = os.path.join(PROJECT_ROOT, "datamart")

BRONZE_DIR = os.path.join(DATAMART_DIR, "bronze")
SILVER_DIR = os.path.join(DATAMART_DIR, "silver")
GOLD_DIR   = os.path.join(DATAMART_DIR, "gold")

# The four source extracts, keyed by the table name we'll use throughout
RAW_SOURCES = {
    "lms_loan_daily":      os.path.join(DATA_DIR, "lms_loan_daily.csv"),
    "feature_clickstream": os.path.join(DATA_DIR, "feature_clickstream.csv"),
    "feature_attributes":  os.path.join(DATA_DIR, "features_attributes.csv"),
    "feature_financials":  os.path.join(DATA_DIR, "features_financials.csv"),
}

# One output folder per table, per layer
BRONZE_TABLES = {name: os.path.join(BRONZE_DIR, name) for name in RAW_SOURCES}
SILVER_TABLES = {name: os.path.join(SILVER_DIR, name) for name in RAW_SOURCES}

GOLD_LABEL_STORE   = os.path.join(GOLD_DIR, "label_store")
GOLD_FEATURE_STORE = os.path.join(GOLD_DIR, "feature_store")

# ---------------------------------------------------------------------------
# Batch window
# ---------------------------------------------------------------------------
# The pipeline runs as a monthly batch. Each snapshot_date is one month-start
# partition — standing in for the monthly schedule a real orchestrator
# (Airflow) would trigger.
START_DATE = "2023-01-01"
END_DATE   = "2025-12-01"

# ---------------------------------------------------------------------------
# Label definition  (justified by EDA: 30 DPD converges with raw arrears at MOB 6)
# ---------------------------------------------------------------------------
MOB_THRESHOLD = 6
DPD_THRESHOLD = 30
LABEL_DEFINITION = f"{DPD_THRESHOLD}dpd_{MOB_THRESHOLD}mob"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
# Clickstream is aggregated over a trailing window ending STRICTLY BEFORE the
# application month. EDA showed 65% of clickstream rows post-date the
# application — using them would be predicting the past from the future.
CLICKSTREAM_WINDOW_MONTHS = 6
CLICKSTREAM_FEATURE_COLS = [f"fe_{i}" for i in range(1, 21)]