"""
Read every monitoring partition and save deck-ready PNGs to
datamart/gold/model_monitoring/charts/.
    python3 utils/model_visualise.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import common

OUT = os.path.join(common.MONITORING_DIR, "charts")
COLORS = {"logreg": "#4F8CFF", "xgb": "#F2A33A"}


def read_all(kind):
    files = glob.glob(os.path.join(common.MONITORING_DIR, "*", kind, "*.parquet"))
    if not files:
        return None
    df = pd.concat(pd.read_parquet(f) for f in files)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df.sort_values(["model_name", "snapshot_date"])


def main():
    stab, perf = read_all("stability"), read_all("performance")
    if stab is None and perf is None:
        print("no monitoring data yet")
        return
    os.makedirs(OUT, exist_ok=True)

    # ---- stability over time ----
    if stab is not None:
        stab.to_csv(os.path.join(OUT, "stability_all.csv"), index=False)
        fig, ax = plt.subplots(figsize=(14, 6))
        for m, g in stab.groupby("model_name"):
            ax.plot(g["snapshot_date"], g["psi_score"], marker="o", lw=2.5,
                    label=f"{m} score PSI", color=COLORS.get(m))
        top = max(0.5, stab["psi_score"].max() * 1.15)
        ax.axhspan(0.10, 0.25, color="#F2A33A", alpha=0.12, label="watch (0.10 - 0.25)")
        ax.axhspan(0.25, top, color="#E05A4F", alpha=0.12, label="action (> 0.25)")
        ax.set_ylim(0, top); ax.set_ylabel("PSI")
        ax.set_title("Score distribution stability by scoring month (PSI vs training baseline)")
        ax.grid(alpha=0.3); ax.legend(loc="upper left"); fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(os.path.join(OUT, "stability_over_time.png"), dpi=150); plt.close(fig)

    # ---- performance over time ----
    if perf is not None:
        perf.to_csv(os.path.join(OUT, "performance_all.csv"), index=False)
        fig, ax = plt.subplots(figsize=(14, 6))
        for m, g in perf.groupby("model_name"):
            ax.plot(g["snapshot_date"], g["auc"], marker="o", lw=2.5,
                    label=f"{m} AUC", color=COLORS.get(m))
            base = g["auc_oot_baseline"].iloc[-1]
            ax.axhline(base, ls="--", lw=1, color=COLORS.get(m), alpha=0.6,
                       label=f"{m} OOT baseline {base:.3f}")
        # months scored but whose labels have not matured yet
        if stab is not None:
            live = [pd.Timestamp(d) for d in stab["snapshot_date"].unique()]
            evaluated = {pd.Timestamp(d) for d in perf["snapshot_date"].unique()}
            pending = sorted(d for d in live if d not in evaluated)
            if pending:
                ax.axvspan(min(pending), max(pending) + pd.offsets.MonthEnd(0),
                           color="#9aa5b8", alpha=0.15,
                           label=f"labels not yet matured ({common.LABEL_LAG_MONTHS}-month lag)")
        ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUC")
        ax.set_title("Model performance by application month (evaluated once labels mature)")
        ax.grid(alpha=0.3); ax.legend(loc="lower left"); fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(os.path.join(OUT, "performance_over_time.png"), dpi=150); plt.close(fig)

    print("charts saved to", OUT)


if __name__ == "__main__":
    main()
