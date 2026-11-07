"""
Gold layer — label store. One row per loan, carrying the supervised target.

Label: 1 if more than 30 days past due at 6 months on book.

Why MOB 6 (from EDA): arrears plateau at 28.8%, and at MOB 6 the 30 DPD rate
converges with the raw arrears rate — anyone still behind is at least 30 days
behind. Earlier, the definition is still moving.

snapshot_date here is when the label becomes OBSERVABLE (origination + 6
months), NOT the application date. Keeping those distinct makes the time gap
between features and labels explicit.
"""

import os
from pyspark.sql import functions as F
from pyspark.sql import types as T

from utils import config
from utils.spark_utils import ensure_clean_dir, partition_name


def build_label_store(spark, snapshot_dates) -> int:
    ensure_clean_dir(config.GOLD_LABEL_STORE)

    # read every silver loan partition as one DataFrame
    loans = spark.read.parquet(
        os.path.join(config.SILVER_TABLES["lms_loan_daily"], "*.parquet"))

    # keep only the ledger row at the performance window we label on
    at_mob = loans.filter(F.col("mob") == config.MOB_THRESHOLD)

    labels = at_mob.select(
        "loan_id", "Customer_ID", "loan_start_date", "snapshot_date", "mob", "dpd",
        F.when(F.col("dpd") >= config.DPD_THRESHOLD, F.lit(1))
         .otherwise(F.lit(0)).cast(T.IntegerType()).alias("label"),
        # the definition travels WITH the data, so nobody has to guess later
        F.lit(config.LABEL_DEFINITION).alias("label_def"),
    )

    total_rows = 0
    partitions_written = 0

    for snapshot_date in snapshot_dates:
        stamp = snapshot_date.strftime("%Y-%m-%d")
        part = labels.filter(F.col("snapshot_date") == stamp)

        n = part.count()
        if n == 0:
            continue

        part.write.mode("overwrite").parquet(os.path.join(
            config.GOLD_LABEL_STORE,
            partition_name("gold_label_store", snapshot_date, "parquet")))

        total_rows += n
        partitions_written += 1

    default_rate = labels.agg(F.avg("label")).collect()[0][0]
    print(f"  [gold]   label_store            {partitions_written:3d} partitions | "
          f"{total_rows:>7,} rows | default rate {default_rate:.2%} "
          f"| def={config.LABEL_DEFINITION}")
    return total_rows
