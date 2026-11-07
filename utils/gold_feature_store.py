"""
Gold layer — feature store.

Grain: ONE ROW PER LOAN APPLICATION, keyed by loan_id, as at loan_start_date.

LEAKAGE CONTROLS
----------------
1. Target leakage — nothing from the loan ledger after origination. paid_amt,
   overdue_amt, balance and dpd all describe how the loan performed, which is
   the answer. Only loan_amt and tenure are taken: both fixed at origination.

2. Temporal leakage — clickstream aggregated over the 6 months STRICTLY
   BEFORE the application month. Behaviour in the application month itself may
   post-date the application and reflect the loan rather than predict it.

3. Train-test contamination — no scaling, imputation or encoding here. Each
   learns a statistic from data; fitting one across the whole store leaks test
   information into training. Gold ships clean typed values and honest NULLs.

Missingness is carried explicitly: where a value was corrupt at source and
resolved to NULL in silver, a *_is_missing flag travels alongside it, so a
model can learn from missingness rather than having it papered over.
"""

import os
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from utils import config
from utils.spark_utils import ensure_clean_dir, partition_name

# Columns where corruption was common enough that "missing" is informative
MISSING_INDICATOR_COLS = [
    "age", "annual_income", "num_bank_accounts", "num_credit_card",
    "interest_rate", "num_existing_loans", "num_delayed_payments",
    "changed_credit_limit", "num_credit_inquiries", "outstanding_debt",
    "amount_invested_monthly", "monthly_balance", "credit_history_months",
    "total_emi_per_month",
]


def _safe_divide(numerator, denominator):
    """Divide, returning NULL instead of raising or producing infinity."""
    return F.when(
        (F.col(denominator).isNotNull()) & (F.col(denominator) != 0)
        & (F.col(numerator).isNotNull()),
        F.col(numerator) / F.col(denominator),
    ).otherwise(F.lit(None).cast(T.DoubleType()))


def _read_silver_partition(spark, table_name: str, snapshot_date):
    path = os.path.join(
        config.SILVER_TABLES[table_name],
        partition_name(f"silver_{table_name}", snapshot_date, "parquet"))
    if not os.path.exists(path):
        return None
    return spark.read.parquet(path)


def _read_clickstream_window(spark, snapshot_date):
    """Read clickstream for the months BEFORE an application.

    A loan applied for on 2024-06-01 sees 2023-12-01 through 2024-05-01 —
    never 2024-06-01 itself. That exclusion is the whole point.
    """
    frames = []
    year, month = snapshot_date.year, snapshot_date.month

    for step in range(1, config.CLICKSTREAM_WINDOW_MONTHS + 1):
        m, y = month - step, year
        while m <= 0:
            m += 12
            y -= 1
        frame = _read_silver_partition(
            spark, "feature_clickstream", snapshot_date.replace(year=y, month=m, day=1))
        if frame is not None:
            frames.append(frame)

    if not frames:
        return None

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.unionByName(frame)
    return combined


def _aggregate_clickstream(clickstream_df):
    """Reduce a customer's window to one row.

    Two statistics per signal:
      avg_fe_i  - the customer's typical level over the window
      last_fe_i - the most recent observation before applying
    Together they capture baseline behaviour AND recent shift; an average
    alone would flatten the shift out.
    """
    aggs = [F.avg(c).alias(f"avg_{c}") for c in config.CLICKSTREAM_FEATURE_COLS]
    aggs.append(F.countDistinct("snapshot_date").alias("clickstream_months"))
    averages = clickstream_df.groupBy("Customer_ID").agg(*aggs)

    # most recent row per customer inside the window
    recency = Window.partitionBy("Customer_ID").orderBy(F.col("snapshot_date").desc())
    latest = (clickstream_df
              .withColumn("_rn", F.row_number().over(recency))
              .filter(F.col("_rn") == 1)
              .select("Customer_ID",
                      *[F.col(c).alias(f"last_{c}")
                        for c in config.CLICKSTREAM_FEATURE_COLS]))

    return averages.join(latest, on="Customer_ID", how="left")


def _add_derived_features(df):
    """Affordability and leverage ratios.

    Raw income and raw debt say little alone; their ratio is what an
    underwriter actually reasons about. Ratios also travel better across time
    than absolute currency amounts, which drift with inflation.
    """
    df = df.withColumn("debt_to_income_ratio", _safe_divide("outstanding_debt", "annual_income"))
    df = df.withColumn("emi_to_salary_ratio", _safe_divide("total_emi_per_month", "monthly_inhand_salary"))
    df = df.withColumn("loan_to_income_ratio", _safe_divide("loan_amt", "annual_income"))
    df = df.withColumn("investment_rate", _safe_divide("amount_invested_monthly", "monthly_inhand_salary"))
    df = df.withColumn("balance_to_salary_ratio", _safe_divide("monthly_balance", "monthly_inhand_salary"))
    df = df.withColumn("credit_lines_total",
                       (F.coalesce(F.col("num_bank_accounts"), F.lit(0))
                        + F.coalesce(F.col("num_credit_card"), F.lit(0))).cast(T.IntegerType()))
    df = df.withColumn("delinquency_intensity",
                       _safe_divide("num_delayed_payments", "credit_history_months"))
    return df


def _add_missing_indicators(df):
    for column in MISSING_INDICATOR_COLS:
        df = df.withColumn(f"{column}_is_missing",
                           F.when(F.col(column).isNull(), F.lit(1))
                            .otherwise(F.lit(0)).cast(T.IntegerType()))
    return df


def build_feature_store(spark, snapshot_dates) -> int:
    ensure_clean_dir(config.GOLD_FEATURE_STORE)

    total_rows = 0
    partitions_written = 0

    for snapshot_date in snapshot_dates:
        # --- applications originated this month --------------------------
        loans = _read_silver_partition(spark, "lms_loan_daily", snapshot_date)
        if loans is None:
            continue

        # installment_num == 0 is the disbursement row: one per new loan.
        # ONLY loan_amt and tenure are taken — everything else in the ledger
        # is post-origination and would leak the answer.
        applications = loans.filter(F.col("installment_num") == 0).select(
            "loan_id", "Customer_ID", "loan_start_date", "tenure", "loan_amt")
        if applications.rdd.isEmpty():
            continue

        # --- applicant attributes and financials -------------------------
        # EDA confirmed these are 100% point-in-time (snapshot_date ==
        # loan_start_date), so a straight join is safe — no date filter needed.
        attributes = _read_silver_partition(spark, "feature_attributes", snapshot_date)
        financials = _read_silver_partition(spark, "feature_financials", snapshot_date)

        features = applications
        if attributes is not None:
            features = features.join(attributes.drop("snapshot_date"),
                                     on="Customer_ID", how="left")
        if financials is not None:
            features = features.join(financials.drop("snapshot_date"),
                                     on="Customer_ID", how="left")

        # --- trailing behavioural window ---------------------------------
        clickstream = _read_clickstream_window(spark, snapshot_date)
        if clickstream is not None:
            features = features.join(_aggregate_clickstream(clickstream),
                                     on="Customer_ID", how="left")
        else:
            # Earliest months have no PRIOR clickstream at all. The columns
            # still have to exist so the schema stays stable across partitions.
            for c in config.CLICKSTREAM_FEATURE_COLS:
                features = features.withColumn(f"avg_{c}", F.lit(None).cast(T.DoubleType()))
                features = features.withColumn(f"last_{c}", F.lit(None).cast(T.IntegerType()))
            features = features.withColumn("clickstream_months",
                                           F.lit(None).cast(T.LongType()))

        features = features.withColumn(
            "clickstream_months",
            F.coalesce(F.col("clickstream_months"), F.lit(0)).cast(T.IntegerType()))
        # 28% of applicants have no clickstream. Expose that to the model
        # rather than hiding it behind a silent zero.
        features = features.withColumn(
            "has_clickstream",
            F.when(F.col("clickstream_months") > 0, F.lit(1))
             .otherwise(F.lit(0)).cast(T.IntegerType()))

        # --- derived features and quality flags --------------------------
        features = _add_derived_features(features)
        features = _add_missing_indicators(features)

        features = features.withColumn(
            "snapshot_date",
            F.lit(snapshot_date.strftime("%Y-%m-%d")).cast(T.DateType())
        ).withColumn("feature_store_version", F.lit("v1"))

        features.write.mode("overwrite").parquet(os.path.join(
            config.GOLD_FEATURE_STORE,
            partition_name("gold_feature_store", snapshot_date, "parquet")))

        total_rows += features.count()
        partitions_written += 1

    print(f"  [gold]   feature_store          {partitions_written:3d} partitions | "
          f"{total_rows:>7,} rows")
    return total_rows