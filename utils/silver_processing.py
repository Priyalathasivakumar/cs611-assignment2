"""
Silver layer — cleaned, typed, schema-enforced.

Cleaning philosophy: NULL over guess.
When a value is unusable (age 8,678; 1,756 bank accounts) we set it to NULL
rather than clipping to a plausible bound. Clipping fabricates data and hides
the quality problem. NULL is honest, and gold adds a *_is_missing flag so the
model can learn from missingness instead of having it papered over.
"""

import os
from pyspark.sql import functions as F
from pyspark.sql import types as T

from utils import config
from utils.spark_utils import ensure_clean_dir, partition_name


# ---------------------------------------------------------------------------
# Reusable cleaning expressions
# ---------------------------------------------------------------------------
def _clean_numeric(column: str, cast_to=T.DoubleType()):
    """Strip formatting noise from a numeric-as-string column.

    ORDER MATTERS. The sentinel check runs BEFORE underscores are stripped —
    otherwise '__10000__' would survive as a real 10000, inventing an
    investment amount for 558 customers who have none.
    """
    col = F.col(column)
    col = F.when(col.rlike(r"^__.*__$"), None).otherwise(col)   # sentinel -> NULL
    col = F.regexp_replace(col, r"_", "")                        # artefact -> strip
    col = F.trim(col)
    col = F.when(col == "", None).otherwise(col)
    return col.cast(cast_to)


def _bounded(col, low, high):
    """NULL out values outside a plausible range."""
    return F.when((col >= low) & (col <= high), col).otherwise(F.lit(None))


def _read_bronze_partition(spark, table_name: str, snapshot_date):
    """Read one bronze CSV partition, or None if that month has no data."""
    path = os.path.join(
        config.BRONZE_TABLES[table_name],
        partition_name(f"bronze_{table_name}", snapshot_date, "csv"))
    if not os.path.exists(path):
        return None
    return spark.read.option("header", True).option("inferSchema", False).csv(path)


# ---------------------------------------------------------------------------
# lms_loan_daily — type it, then derive the delinquency measures
# ---------------------------------------------------------------------------
def clean_loan_daily(df):
    out = df.select(
        F.col("loan_id").cast(T.StringType()),
        F.col("Customer_ID").cast(T.StringType()),
        F.to_date("loan_start_date").alias("loan_start_date"),
        F.col("tenure").cast(T.IntegerType()).alias("tenure"),
        F.col("installment_num").cast(T.IntegerType()).alias("installment_num"),
        F.col("loan_amt").cast(T.DoubleType()).alias("loan_amt"),
        F.col("due_amt").cast(T.DoubleType()).alias("due_amt"),
        F.col("paid_amt").cast(T.DoubleType()).alias("paid_amt"),
        F.col("overdue_amt").cast(T.DoubleType()).alias("overdue_amt"),
        F.col("balance").cast(T.DoubleType()).alias("balance"),
        F.to_date("snapshot_date").alias("snapshot_date"),
    )

    # Month on book: instalment 0 is disbursement, so MOB == instalment number
    out = out.withColumn("mob", F.col("installment_num"))

    # Overdue balance expressed in whole instalments.
    # Guarded against due_amt = 0 (the disbursement row) -> divide by zero.
    out = out.withColumn(
        "installments_missed",
        F.when(F.col("due_amt") > 0,
               F.ceil(F.col("overdue_amt") / F.col("due_amt")))
         .otherwise(F.lit(0)).cast(T.IntegerType()))

    # Walk back one month per missed instalment to find when arrears began
    out = out.withColumn(
        "first_missed_date",
        F.when(F.col("installments_missed") > 0,
               F.add_months(F.col("snapshot_date"), -F.col("installments_missed")))
         .otherwise(F.lit(None).cast(T.DateType())))

    out = out.withColumn(
        "dpd",
        F.when(F.col("overdue_amt") > 0,
               F.datediff(F.col("snapshot_date"), F.col("first_missed_date")))
         .otherwise(F.lit(0)).cast(T.IntegerType()))

    return out


# ---------------------------------------------------------------------------
# feature_attributes — clean demographics, drop direct identifiers
# ---------------------------------------------------------------------------
def clean_attributes(df):
    """Name and SSN are DROPPED here.

    They're personal identifiers with no predictive value, and a feature store
    is a widely shared asset — the fewer places PII lives, the better. SSN
    survives only as a data-quality flag: 703 records carry junk, and how a
    record was captured may itself be predictive.
    """
    age = _clean_numeric("Age", T.IntegerType())

    return df.select(
        F.col("Customer_ID").cast(T.StringType()),
        _bounded(age, 18, 100).alias("age"),
        F.when(F.col("SSN").rlike(r"^\d{3}-\d{2}-\d{4}$"), F.lit(1))
         .otherwise(F.lit(0)).cast(T.IntegerType()).alias("has_valid_ssn"),
        # '_______' is the source placeholder for unknown occupation
        F.when((F.col("Occupation").isNull()) | (F.col("Occupation").rlike(r"^_+$")),
               F.lit("Unknown"))
         .otherwise(F.col("Occupation")).alias("occupation"),
        F.to_date("snapshot_date").alias("snapshot_date"),
    )


# ---------------------------------------------------------------------------
# feature_financials — the dirtiest source
# ---------------------------------------------------------------------------
def clean_financials(df):
    out = df.select(
        F.col("Customer_ID").cast(T.StringType()),
        _clean_numeric("Annual_Income").alias("annual_income"),
        _clean_numeric("Monthly_Inhand_Salary").alias("monthly_inhand_salary"),
        _bounded(_clean_numeric("Num_Bank_Accounts", T.IntegerType()), 0, 20)
            .alias("num_bank_accounts"),
        _bounded(_clean_numeric("Num_Credit_Card", T.IntegerType()), 0, 15)
            .alias("num_credit_card"),
        _bounded(_clean_numeric("Interest_Rate", T.IntegerType()), 1, 50)
            .alias("interest_rate"),
        _bounded(_clean_numeric("Num_of_Loan", T.IntegerType()), 0, 15)
            .alias("num_existing_loans"),
        # negatives are legitimate here — the customer paid early
        _clean_numeric("Delay_from_due_date", T.IntegerType()).alias("delay_from_due_date"),
        _bounded(_clean_numeric("Num_of_Delayed_Payment", T.IntegerType()), 0, 50)
            .alias("num_delayed_payments"),
        _clean_numeric("Changed_Credit_Limit").alias("changed_credit_limit"),
        _bounded(_clean_numeric("Num_Credit_Inquiries", T.IntegerType()), 0, 50)
            .alias("num_credit_inquiries"),
        # '_' is the placeholder for an unclassified credit mix
        F.when((F.col("Credit_Mix").isNull()) | (F.col("Credit_Mix").rlike(r"^_+$")),
               F.lit("Unknown")).otherwise(F.col("Credit_Mix")).alias("credit_mix"),
        _clean_numeric("Outstanding_Debt").alias("outstanding_debt"),
        _clean_numeric("Credit_Utilization_Ratio").alias("credit_utilization_ratio"),
        # 'NM' (not mentioned) is a third state, not a missing Yes/No
        F.when(F.col("Payment_of_Min_Amount") == "NM", F.lit("Unknown"))
         .otherwise(F.col("Payment_of_Min_Amount")).alias("payment_of_min_amount"),
        _bounded(_clean_numeric("Total_EMI_per_month"), 0, 25000).alias("total_emi_per_month"),
        _clean_numeric("Amount_invested_monthly").alias("amount_invested_monthly"),
        # '!@9#%8' is the junk placeholder here
        F.when(F.col("Payment_Behaviour").rlike(r"^[A-Za-z_]+$"), F.col("Payment_Behaviour"))
         .otherwise(F.lit("Unknown")).alias("payment_behaviour"),
        _clean_numeric("Monthly_Balance").alias("monthly_balance"),
        F.col("Type_of_Loan").alias("type_of_loan_raw"),
        F.col("Credit_History_Age").alias("credit_history_raw"),
        F.to_date("snapshot_date").alias("snapshot_date"),
    )

    # 'X Years and Y Months' -> integer month count, so it's usable as a number
    history = F.coalesce(F.col("credit_history_raw"), F.lit(""))
    out = out.withColumn(
        "credit_history_months",
        F.regexp_extract(history, r"(\d+)\s+Year", 1).cast(T.IntegerType()) * F.lit(12)
        + F.regexp_extract(history, r"(\d+)\s+Month", 1).cast(T.IntegerType()))
    out = out.withColumn(
        "credit_history_months",
        F.when(F.col("credit_history_months") > 0, F.col("credit_history_months"))
    ).drop("credit_history_raw")

    # Explode free-text loan list into one boolean per product.
    # A model can't read "Student Loan, Credit-Builder Loan, and Payday Loan";
    # eight flags plus a count make the same information usable.
    loan_types = ["Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan",
                  "Home Equity Loan", "Mortgage Loan", "Payday Loan",
                  "Personal Loan", "Student Loan"]
    text = F.coalesce(F.col("type_of_loan_raw"), F.lit(""))
    flag_cols = []
    for lt in loan_types:
        slug = lt.lower().replace("-", "_").replace(" loan", "").replace(" ", "_")
        colname = f"has_{slug}_loan"
        out = out.withColumn(colname,
                             F.when(text.contains(lt), F.lit(1))
                              .otherwise(F.lit(0)).cast(T.IntegerType()))
        flag_cols.append(colname)

    out = out.withColumn("num_loan_types",
                         sum(F.col(c) for c in flag_cols).cast(T.IntegerType()))
    return out.drop("type_of_loan_raw")


# ---------------------------------------------------------------------------
# feature_clickstream — already clean, so silver only enforces types
# ---------------------------------------------------------------------------
def clean_clickstream(df):
    """No nulls, no junk values in this source.

    Saying so explicitly matters: it documents that the absence of cleaning
    here is a finding, not an oversight.
    """
    cols = [F.col("Customer_ID").cast(T.StringType()),
            F.to_date("snapshot_date").alias("snapshot_date")]
    cols += [F.col(f"fe_{i}").cast(T.IntegerType()).alias(f"fe_{i}") for i in range(1, 21)]
    return df.select(*cols)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
CLEANERS = {
    "lms_loan_daily":      clean_loan_daily,
    "feature_attributes":  clean_attributes,
    "feature_financials":  clean_financials,
    "feature_clickstream": clean_clickstream,
}


def run_silver_layer(spark, snapshot_dates) -> None:
    print("\n[2/4] SILVER LAYER - cleaned, typed, schema-enforced")
    os.makedirs(config.SILVER_DIR, exist_ok=True)

    for table_name, cleaner in CLEANERS.items():
        ensure_clean_dir(config.SILVER_TABLES[table_name])
        total_rows = 0
        partitions_written = 0

        for snapshot_date in snapshot_dates:
            bronze_df = _read_bronze_partition(spark, table_name, snapshot_date)
            if bronze_df is None:
                continue

            silver_df = cleaner(bronze_df)
            out_path = os.path.join(
                config.SILVER_TABLES[table_name],
                partition_name(f"silver_{table_name}", snapshot_date, "parquet"))
            silver_df.write.mode("overwrite").parquet(out_path)

            total_rows += silver_df.count()
            partitions_written += 1

        print(f"  [silver] {table_name:22s} {partitions_written:3d} partitions | "
              f"{total_rows:>7,} rows")