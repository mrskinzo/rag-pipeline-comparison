"""Turn evaluation_results.csv into summary tables and charts under reports/.

Outputs:
  reports/summary.csv                   — mean metric per config
  reports/best_config_per_question.csv  — winning config for each question
  reports/quality_scores.png            — grouped bar chart of quality metrics
  reports/latency.png                   — bar chart of mean latency

Charts require matplotlib; if it isn't installed the CSVs are still written.
"""

import argparse
import os

import pandas as pd

QUALITY_METRICS = ["faithfulness", "answer_relevancy", "context_precision",
                   "answer_correctness"]


def write_summary(df: pd.DataFrame, outdir: str) -> pd.DataFrame:
    metrics = [m for m in QUALITY_METRICS if m in df.columns] + ["latency"]
    summary = df.groupby("config")[metrics].mean().round(3)
    path = os.path.join(outdir, "summary.csv")
    summary.to_csv(path)
    print(f"Wrote {path}")
    return summary


def write_best_per_question(df: pd.DataFrame, outdir: str) -> None:
    metrics = [m for m in QUALITY_METRICS if m in df.columns]
    scored = df.copy()
    scored["quality_score"] = scored[metrics].mean(axis=1)
    best = scored.loc[
        scored.groupby("question")["quality_score"].idxmax(),
        ["question", "config", "quality_score", "latency"],
    ].sort_values("question")
    path = os.path.join(outdir, "best_config_per_question.csv")
    best.to_csv(path, index=False)
    print(f"Wrote {path}")


def write_charts(summary: pd.DataFrame, outdir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping charts "
              "(pip install matplotlib to enable)")
        return

    metrics = [m for m in QUALITY_METRICS if m in summary.columns]

    ax = summary[metrics].plot.bar(figsize=(10, 6), rot=15)
    ax.set_title("Quality scores by configuration (mean)")
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.figure.tight_layout()
    quality_path = os.path.join(outdir, "quality_scores.png")
    ax.figure.savefig(quality_path, dpi=150)
    plt.close(ax.figure)
    print(f"Wrote {quality_path}")

    ax = summary["latency"].plot.bar(figsize=(8, 5), rot=15, color="#4c72b0")
    ax.set_title("Mean end-to-end latency by configuration")
    ax.set_ylabel("Seconds")
    ax.figure.tight_layout()
    latency_path = os.path.join(outdir, "latency.png")
    ax.figure.savefig(latency_path, dpi=150)
    plt.close(ax.figure)
    print(f"Wrote {latency_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize and chart RAG evaluation results."
    )
    parser.add_argument("--input", type=str, default="evaluation_results.csv",
                        help="evaluation results CSV produced by evaluate.py")
    parser.add_argument("--outdir", type=str, default="reports",
                        help="directory for summary CSVs and charts")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    os.makedirs(args.outdir, exist_ok=True)

    summary = write_summary(df, args.outdir)
    write_best_per_question(df, args.outdir)
    write_charts(summary, args.outdir)

    print("\n" + summary.to_string())


if __name__ == "__main__":
    main()
