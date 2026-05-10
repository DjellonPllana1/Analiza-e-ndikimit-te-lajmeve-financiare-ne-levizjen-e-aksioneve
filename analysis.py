from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind


@dataclass(frozen=True)
class Paths:
    merged_csv: Path
    output_dir: Path


def default_paths() -> Paths:
    root = Path(__file__).resolve().parent
    merged_csv = root / "DataPreparation" / "datasets" / "cleanedData" / "Combined_News_DJIA_Merged.csv"
    output_dir = root / "outputs"
    return Paths(merged_csv=merged_csv, output_dir=output_dir)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["daily_return"] = df["Close"].pct_change()
    return df


def plot_close_and_label(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Close"], linewidth=1)
    ax.set_title("DJIA Close (kohë)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    fig.tight_layout()
    fig.savefig(output_dir / "djia_close_timeseries.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="daily_return", hue="Label", bins=60, element="step", stat="density", common_norm=False, ax=ax)
    ax.set_title("Shpërndarja e kthimeve ditore sipas Label")
    ax.set_xlabel("daily_return (Close % change)")
    fig.tight_layout()
    fig.savefig(output_dir / "daily_returns_by_label.png", dpi=160)
    plt.close(fig)


def stats_return_difference(df: pd.DataFrame) -> dict[str, float]:
    d = df.dropna(subset=["daily_return", "Label"]).copy()
    r0 = d.loc[d["Label"] == 0, "daily_return"]
    r1 = d.loc[d["Label"] == 1, "daily_return"]
    if len(r0) < 5 or len(r1) < 5:
        return {"n0": float(len(r0)), "n1": float(len(r1)), "mean0": float(r0.mean()), "mean1": float(r1.mean()), "pvalue": float("nan")}

    stat = ttest_ind(r1, r0, equal_var=False, nan_policy="omit")
    return {
        "n0": float(len(r0)),
        "n1": float(len(r1)),
        "mean0": float(r0.mean()),
        "mean1": float(r1.mean()),
        "pvalue": float(stat.pvalue),
    }


def main() -> int:
    p = default_paths()
    if not p.merged_csv.exists():
        raise FileNotFoundError(f"Nuk u gjet dataset-i i bashkuar: {p.merged_csv}")

    df = load_data(p.merged_csv)
    plot_close_and_label(df, p.output_dir)
    s = stats_return_difference(df)

    summary = (
        "Return difference test (Label=1 vs Label=0)\n"
        f"n0={int(s['n0'])}, mean0={s['mean0']:.6f}\n"
        f"n1={int(s['n1'])}, mean1={s['mean1']:.6f}\n"
        f"p-value={s['pvalue']:.6g}\n"
    )
    (p.output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved outputs to: {p.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

