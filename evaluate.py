"""
TraceAudit — Benchmarking Suite (evaluate.py)
==============================================
Computes precision, recall, F1-score and generates visual reports.

CLI Usage:
    # Evaluate CoT predictions vs ground truth
    python src/evaluate.py --predictions results/predictions.jsonl

    # Compare CoT vs Zero-Shot (A/B test)
    python src/evaluate.py \\
        --predictions results/predictions.jsonl \\
        --compare results/predictions_zero_shot.jsonl \\
        --output-dir results/

    # Full report including linter baseline
    python src/evaluate.py --predictions results/predictions.jsonl --linter-baseline

Outputs (in results/):
    metrics_report.json     — Full per-class and macro metrics
    confusion_matrix.png    — Heatmap of predicted vs actual
    f1_comparison.png       — CoT vs Zero-Shot F1 bar chart (if --compare given)
    cost_benefit.png        — Token cost vs F1 gain chart (if --compare given)
    linter_comparison.png   — AI vs Bandit/Flake8 detection rates (if --linter-baseline)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ── Path bootstrap ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluator")

# ── Constants ─────────────────────────────────────────────────────────────────
BUG_LABELS = [
    "RESOURCE_LEAK",
    "BOUNDARY_VIOLATION",
    "INSECURE_DATA_FLOW",
    "CONCURRENCY_HAZARD",
    "IMPROPER_ERROR_HANDLING",
]

# Aesthetics
PALETTE_DARK = "#0d1117"
PALETTE_ACCENT = "#58a6ff"
PALETTE_WARN = "#f0883e"
PALETTE_SUCCESS = "#3fb950"
PALETTE_DANGER = "#f85149"
FIGURE_DPI = 150


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_predictions(path: Path) -> pd.DataFrame:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prediction = r.get("prediction", {})
            records.append(
                {
                    "function_name": r.get("original_function_name", "unknown"),
                    "ground_truth": r.get("bug_type", "UNKNOWN"),
                    "predicted": prediction.get("predicted_bug_type", "UNKNOWN"),
                    "confidence": prediction.get("confidence", "low"),
                    "mode": prediction.get("mode", "cot"),
                    "prompt_tokens": r.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": r.get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": r.get("usage", {}).get("total_tokens", 0),
                    "is_detectable_by_linter": r.get("is_detectable_by_linter", False),
                    "difficulty": r.get("difficulty", "medium"),
                    "bug_line_number": r.get("bug_line_number", None),
                    "predicted_bug_line": prediction.get("predicted_bug_line", None),
                    "mutated_code": r.get("mutated_code", ""),
                }
            )
    return pd.DataFrame(records)


# ── Core Metrics ──────────────────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame, label: str = "CoT") -> dict:
    y_true = df["ground_truth"].tolist()
    y_pred = df["predicted"].tolist()

    # Use only labels that appear in y_true for cleaner reports
    active_labels = [l for l in BUG_LABELS if l in y_true]

    report = classification_report(
        y_true, y_pred,
        labels=active_labels,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = f1_score(y_true, y_pred, labels=active_labels, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, labels=active_labels, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=active_labels, average="macro", zero_division=0)

    accuracy = (df["ground_truth"] == df["predicted"]).mean()

    # Per-difficulty breakdown
    difficulty_f1 = {}
    for diff in ["easy", "medium", "hard"]:
        subset = df[df["difficulty"] == diff]
        if len(subset) > 0:
            difficulty_f1[diff] = f1_score(
                subset["ground_truth"],
                subset["predicted"],
                labels=active_labels,
                average="macro",
                zero_division=0,
            )

    # Per-confidence breakdown
    confidence_accuracy = {}
    for conf in ["low", "medium", "high"]:
        subset = df[df["confidence"] == conf]
        if len(subset) > 0:
            confidence_accuracy[conf] = (
                subset["ground_truth"] == subset["predicted"]
            ).mean()

    avg_tokens = df["total_tokens"].mean()

    return {
        "label": label,
        "n_samples": len(df),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "avg_tokens_per_sample": round(avg_tokens, 1),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1": round(report[cls]["f1-score"], 4),
                "support": report[cls]["support"],
            }
            for cls in active_labels
            if cls in report
        },
        "per_difficulty_f1": {k: round(v, 4) for k, v in difficulty_f1.items()},
        "per_confidence_accuracy": {k: round(v, 4) for k, v in confidence_accuracy.items()},
    }


def print_metrics_table(metrics: dict) -> None:
    print(f"\n{'═'*60}")
    print(f"  📊 BENCHMARK RESULTS — {metrics['label']}")
    print(f"{'═'*60}")
    print(f"  Samples          : {metrics['n_samples']}")
    print(f"  Accuracy         : {metrics['accuracy']:.1%}")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision  : {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall     : {metrics['macro_recall']:.4f}")
    print(f"  Avg Tokens/Sample: {metrics['avg_tokens_per_sample']:,.0f}")
    print(f"\n  Per-Class Breakdown:")
    print(f"  {'Bug Type':<30} {'P':>6} {'R':>6} {'F1':>6} {'N':>5}")
    print(f"  {'-'*55}")
    for cls, vals in metrics["per_class"].items():
        print(
            f"  {cls:<30} {vals['precision']:>6.3f} {vals['recall']:>6.3f} "
            f"{vals['f1']:>6.3f} {vals['support']:>5}"
        )
    if metrics.get("per_difficulty_f1"):
        print(f"\n  Macro F1 by Difficulty:")
        for diff, f1 in metrics["per_difficulty_f1"].items():
            bar = "█" * int(f1 * 20)
            print(f"  {diff:<8} {f1:.3f} {bar}")
    print(f"{'═'*60}\n")


# ── Visualizations ────────────────────────────────────────────────────────────
def _apply_dark_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE_DARK,
            "axes.facecolor": "#161b22",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#c9d1d9",
            "xtick.color": "#8b949e",
            "ytick.color": "#8b949e",
            "text.color": "#c9d1d9",
            "grid.color": "#21262d",
            "grid.linewidth": 0.5,
            "font.family": "monospace",
        }
    )


def plot_confusion_matrix(df: pd.DataFrame, output_path: Path, title: str = "CoT Reviewer") -> None:
    y_true = df["ground_truth"]
    y_pred = df["predicted"]
    active_labels = [l for l in BUG_LABELS if l in y_true.values]

    cm = confusion_matrix(y_true, y_pred, labels=active_labels)
    # Normalize by row (true label) → shows recall per class
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    _apply_dark_theme()
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(PALETTE_DARK)

    short_labels = [l.replace("_", "\n") for l in active_labels]

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap=sns.color_palette("Blues", as_cmap=True),
        xticklabels=short_labels,
        yticklabels=short_labels,
        linewidths=0.5,
        linecolor="#30363d",
        ax=ax,
        cbar_kws={"shrink": 0.8},
        vmin=0.0,
        vmax=1.0,
    )

    # Raw counts as secondary annotation
    for i in range(len(active_labels)):
        for j in range(len(active_labels)):
            ax.text(
                j + 0.5, i + 0.78,
                f"n={cm[i, j]}",
                ha="center", va="center",
                fontsize=7, color="#8b949e",
            )

    ax.set_title(
        f"Confusion Matrix — {title}\n(normalized by true label; diagonal = recall per class)",
        fontsize=13, pad=16, color="#e6edf3", fontweight="bold",
    )
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Label", fontsize=11, labelpad=10)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=PALETTE_DARK)
    plt.close()
    log.info("Saved: %s", output_path)


def plot_f1_comparison(
    metrics_cot: dict,
    metrics_zs: dict,
    output_path: Path,
) -> None:
    """Side-by-side per-class F1 bar chart: CoT vs Zero-Shot."""
    active = list(metrics_cot["per_class"].keys())
    cot_f1 = [metrics_cot["per_class"][c]["f1"] for c in active]
    zs_f1 = [metrics_zs["per_class"].get(c, {}).get("f1", 0.0) for c in active]

    x = np.arange(len(active))
    width = 0.35

    _apply_dark_theme()
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(PALETTE_DARK)

    bars_cot = ax.bar(x - width / 2, cot_f1, width, label="CoT (Trace)", color=PALETTE_ACCENT, alpha=0.9)
    bars_zs = ax.bar(x + width / 2, zs_f1, width, label="Zero-Shot", color=PALETTE_WARN, alpha=0.9)

    for bar in bars_cot:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=8, color=PALETTE_ACCENT,
        )
    for bar in bars_zs:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=8, color=PALETTE_WARN,
        )

    # Delta annotations
    for i, (c, z) in enumerate(zip(cot_f1, zs_f1)):
        delta = c - z
        color = PALETTE_SUCCESS if delta >= 0 else PALETTE_DANGER
        ax.text(
            x[i], max(c, z) + 0.07,
            f"Δ{delta:+.2f}",
            ha="center", va="bottom", fontsize=9, color=color, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", "\n") for l in active], fontsize=9)
    ax.set_ylim(0, 1.25)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_title(
        "CoT Trace vs Zero-Shot — Per-Class F1 Comparison\n"
        f"Macro F1: CoT={metrics_cot['macro_f1']:.3f} | Zero-Shot={metrics_zs['macro_f1']:.3f} | "
        f"Gain: {metrics_cot['macro_f1'] - metrics_zs['macro_f1']:+.3f}",
        fontsize=12, pad=14, color="#e6edf3", fontweight="bold",
    )
    ax.legend(fontsize=10, framealpha=0.2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=PALETTE_DARK)
    plt.close()
    log.info("Saved: %s", output_path)


def plot_cost_benefit(
    metrics_cot: dict,
    metrics_zs: dict,
    output_path: Path,
) -> None:
    """Scatter/bubble chart: avg token cost vs macro F1."""
    _apply_dark_theme()
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(PALETTE_DARK)

    data = [
        ("Zero-Shot", metrics_zs["avg_tokens_per_sample"], metrics_zs["macro_f1"], PALETTE_WARN),
        ("CoT Trace", metrics_cot["avg_tokens_per_sample"], metrics_cot["macro_f1"], PALETTE_ACCENT),
    ]

    for label, tokens, f1, color in data:
        ax.scatter(tokens, f1, s=300, c=color, zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(
            f"{label}\n{tokens:.0f} tokens\nF1={f1:.3f}",
            xy=(tokens, f1),
            xytext=(15, -15),
            textcoords="offset points",
            fontsize=10,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

    # Arrow showing the tradeoff
    ax.annotate(
        "",
        xy=(data[1][1], data[1][2]),
        xytext=(data[0][1], data[0][2]),
        arrowprops=dict(arrowstyle="->", color="#8b949e", lw=1.5, linestyle="dashed"),
    )

    ax.set_xlabel("Avg Tokens per Sample (Cost Proxy)", fontsize=11)
    ax.set_ylabel("Macro F1-Score", fontsize=11)
    ax.set_title(
        "Cost–Benefit Analysis: Token Usage vs Detection Quality",
        fontsize=12, pad=12, color="#e6edf3", fontweight="bold",
    )
    ax.grid(alpha=0.3)

    # Cost annotation
    token_diff = data[1][1] - data[0][1]
    f1_gain = data[1][2] - data[0][2]
    ax.text(
        0.05, 0.05,
        f"CoT uses {token_diff:+.0f} tokens/sample\nfor {f1_gain:+.3f} F1 gain",
        transform=ax.transAxes, fontsize=9, color="#8b949e",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d", edgecolor="#30363d", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=PALETTE_DARK)
    plt.close()
    log.info("Saved: %s", output_path)


def plot_linter_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart: bugs detectable by linter vs AI detection rate."""
    _apply_dark_theme()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(PALETTE_DARK)

    results = []
    for bug_type in BUG_LABELS:
        subset = df[df["ground_truth"] == bug_type]
        if len(subset) == 0:
            continue
        ai_correct = (subset["ground_truth"] == subset["predicted"]).mean()
        linter_detectable = subset["is_detectable_by_linter"].mean()
        results.append(
            {
                "bug_type": bug_type,
                "ai_detection_rate": ai_correct,
                "linter_detectable_rate": linter_detectable,
            }
        )

    res_df = pd.DataFrame(results)
    x = np.arange(len(res_df))
    width = 0.35

    ax.bar(
        x - width / 2,
        res_df["ai_detection_rate"],
        width,
        label="AI (CoT) Detection Rate",
        color=PALETTE_ACCENT,
        alpha=0.9,
    )
    ax.bar(
        x + width / 2,
        res_df["linter_detectable_rate"],
        width,
        label="Linter Detectable Rate (Ground Truth)",
        color="#6e7681",
        alpha=0.9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", "\n") for l in res_df["bug_type"]], fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Detection Rate", fontsize=11)
    ax.set_title(
        "TraceAudit AI vs Static Linter — Bug Detection Comparison\n"
        "(Linter rate = fraction of ground-truth bugs labeled 'linter detectable')",
        fontsize=11, pad=12, color="#e6edf3", fontweight="bold",
    )
    ax.legend(fontsize=9, framealpha=0.2)
    ax.grid(axis="y", alpha=0.3)

    # Annotation: "AI advantage"
    for i, row in res_df.iterrows():
        delta = row["ai_detection_rate"] - row["linter_detectable_rate"]
        color = PALETTE_SUCCESS if delta > 0 else PALETTE_DANGER
        ax.text(
            x[i], max(row["ai_detection_rate"], row["linter_detectable_rate"]) + 0.06,
            f"AI {delta:+.0%}",
            ha="center", fontsize=8, color=color, fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=PALETTE_DARK)
    plt.close()
    log.info("Saved: %s", output_path)


# ── Linter Baseline (run bandit/flake8 on the mutated code) ──────────────────
def run_linter_baseline(df: pd.DataFrame) -> dict:
    """
    Write each mutated function to a temp file, run bandit on it,
    and check if bandit surfaces any issues.
    Returns a dict: bug_type -> (n_flagged, n_total)
    """
    try:
        subprocess.run(["bandit", "--version"], capture_output=True, check=True)
        bandit_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        log.warning("bandit not found — skipping linter baseline (pip install bandit)")
        bandit_available = False

    results = defaultdict(lambda: {"flagged": 0, "total": 0})

    for _, row in df.iterrows():
        bug_type = row["ground_truth"]
        results[bug_type]["total"] += 1

        if not bandit_available:
            continue

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(row["mutated_code"])
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                ["bandit", "-q", "-f", "json", tmp_path],
                capture_output=True, text=True, timeout=10,
            )
            output = json.loads(result.stdout or "{}")
            issues = output.get("results", [])
            if issues:
                results[bug_type]["flagged"] += 1
        except Exception:
            pass
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    summary = {}
    for bt, counts in results.items():
        summary[bt] = {
            "flagged": counts["flagged"],
            "total": counts["total"],
            "detection_rate": counts["flagged"] / max(counts["total"], 1),
        }
    return summary


# ── JSON Report ───────────────────────────────────────────────────────────────
def save_metrics_report(
    metrics: dict,
    output_path: Path,
    compare_metrics: dict | None = None,
    linter_baseline: dict | None = None,
) -> None:
    report = {"cot_metrics": metrics}
    if compare_metrics:
        report["zero_shot_metrics"] = compare_metrics
        report["cot_vs_zero_shot"] = {
            "macro_f1_gain": round(metrics["macro_f1"] - compare_metrics["macro_f1"], 4),
            "accuracy_gain": round(metrics["accuracy"] - compare_metrics["accuracy"], 4),
            "token_overhead": round(
                metrics["avg_tokens_per_sample"] - compare_metrics["avg_tokens_per_sample"], 1
            ),
        }
    if linter_baseline:
        report["linter_baseline"] = linter_baseline

    with output_path.open("w") as f:
        json.dump(report, f, indent=2)
    log.info("Saved metrics report: %s", output_path)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="TraceAudit Benchmarking Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions",
        default=str(PROJECT_ROOT / "results" / "predictions.jsonl"),
        help="CoT predictions file from reviewer.py",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Zero-shot predictions file for A/B comparison (optional)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results"),
        help="Directory to save all charts and reports",
    )
    parser.add_argument(
        "--linter-baseline",
        action="store_true",
        help="Run bandit on mutated code to generate linter comparison chart",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and evaluate CoT predictions ─────────────────────────────────────
    log.info("Loading CoT predictions from: %s", args.predictions)
    df_cot = load_predictions(Path(args.predictions))
    metrics_cot = compute_metrics(df_cot, label="CoT (Virtual Trace)")
    print_metrics_table(metrics_cot)

    # ── Confusion Matrix ───────────────────────────────────────────────────────
    plot_confusion_matrix(
        df_cot,
        output_dir / "confusion_matrix.png",
        title="CoT Reviewer",
    )

    # ── Linter Comparison Chart ────────────────────────────────────────────────
    if args.linter_baseline:
        log.info("Running linter baseline...")
        linter_results = run_linter_baseline(df_cot)
        plot_linter_comparison(df_cot, output_dir / "linter_comparison.png")
        log.info("Linter Baseline: %s", json.dumps(linter_results, indent=2))
    else:
        linter_results = None

    # ── CoT vs Zero-Shot A/B Test ──────────────────────────────────────────────
    compare_metrics = None
    if args.compare:
        log.info("Loading zero-shot predictions from: %s", args.compare)
        df_zs = load_predictions(Path(args.compare))
        compare_metrics = compute_metrics(df_zs, label="Zero-Shot (Direct)")
        print_metrics_table(compare_metrics)

        plot_f1_comparison(
            metrics_cot,
            compare_metrics,
            output_dir / "f1_comparison.png",
        )
        plot_cost_benefit(
            metrics_cot,
            compare_metrics,
            output_dir / "cost_benefit.png",
        )

        print("\n── CoT vs Zero-Shot Summary ─────────────────────────")
        f1_gain = metrics_cot["macro_f1"] - compare_metrics["macro_f1"]
        token_overhead = metrics_cot["avg_tokens_per_sample"] - compare_metrics["avg_tokens_per_sample"]
        print(f"  Macro F1 Gain     : {f1_gain:+.4f}")
        print(f"  Accuracy Gain     : {metrics_cot['accuracy'] - compare_metrics['accuracy']:+.4f}")
        print(f"  Avg Token Overhead: {token_overhead:+.1f} tokens/sample")
        print("─────────────────────────────────────────────────────\n")

    # ── Save JSON Report ───────────────────────────────────────────────────────
    save_metrics_report(
        metrics_cot,
        output_dir / "metrics_report.json",
        compare_metrics=compare_metrics,
        linter_baseline=linter_results,
    )

    print(f"\n✓ All outputs saved to: {output_dir}/")
    print("  📊 confusion_matrix.png")
    if args.compare:
        print("  📊 f1_comparison.png")
        print("  📊 cost_benefit.png")
    if args.linter_baseline:
        print("  📊 linter_comparison.png")
    print("  📄 metrics_report.json")


if __name__ == "__main__":
    main()
