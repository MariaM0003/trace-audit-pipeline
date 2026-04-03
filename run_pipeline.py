"""
TraceAudit — Master Pipeline Orchestrator
==========================================
Run the full pipeline end-to-end with a single command.

Usage:
    # Full run: generate 50 samples, review with CoT + zero-shot, evaluate
    python run_pipeline.py --samples 50

    # Large-scale run for portfolio
    python run_pipeline.py --samples 500 --engine gpt-4o-mini --concurrency 20

    # Dry-run: only generate (no review/evaluate)
    python run_pipeline.py --samples 20 --only generate

    # Evaluate existing dataset (skip generation)
    python run_pipeline.py --only evaluate
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET = PROJECT_ROOT / "results" / "dataset.jsonl"
PREDICTIONS_COT = PROJECT_ROOT / "results" / "predictions.jsonl"
PREDICTIONS_ZS = PROJECT_ROOT / "results" / "predictions_zero_shot.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"


def run_stage(name: str, cmd: list[str]) -> None:
    log.info("┌── Starting stage: %s", name)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log.error("└── Stage '%s' failed with code %d", name, result.returncode)
        sys.exit(result.returncode)
    log.info("└── Stage '%s' complete ✓", name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TraceAudit — Full Pipeline Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--engine", default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument(
        "--only",
        choices=["generate", "review", "evaluate", "all"],
        default="all",
        help="Run only a specific stage (default: all)",
    )
    parser.add_argument(
        "--skip-zero-shot",
        action="store_true",
        help="Skip zero-shot review (saves ~50%% API cost; skips A/B comparison charts)",
    )
    parser.add_argument(
        "--linter-baseline",
        action="store_true",
        help="Include bandit linter comparison in evaluation",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    stages = args.only if args.only != "all" else "generate,review,evaluate"
    run_generate = "generate" in stages
    run_review = "review" in stages
    run_evaluate = "evaluate" in stages

    print("\n" + "═" * 60)
    print("  🔬 TraceAudit Pipeline")
    print("═" * 60)
    print(f"  Samples     : {args.samples}")
    print(f"  Model       : {args.engine}")
    print(f"  Concurrency : {args.concurrency}")
    print(f"  Stages      : {stages}")
    print("═" * 60 + "\n")

    # ── Stage 1: Generate mutated dataset ─────────────────────────────────────
    if run_generate:
        run_stage(
            "MUTATION ENGINE",
            [
                sys.executable, "src/generator.py",
                "--samples", str(args.samples),
                "--engine", args.engine,
                "--output", str(DATASET),
                "--concurrency", str(args.concurrency),
            ],
        )

    # ── Stage 2a: CoT Review ───────────────────────────────────────────────────
    if run_review:
        run_stage(
            "CoT REVIEWER",
            [
                sys.executable, "src/reviewer.py",
                "--dataset", str(DATASET),
                "--output", str(PREDICTIONS_COT),
                "--engine", args.engine,
                "--mode", "cot",
                "--concurrency", str(max(args.concurrency // 2, 4)),
            ],
        )

    # ── Stage 2b: Zero-Shot Review (for A/B comparison) ───────────────────────
    if run_review and not args.skip_zero_shot:
        run_stage(
            "ZERO-SHOT REVIEWER",
            [
                sys.executable, "src/reviewer.py",
                "--dataset", str(DATASET),
                "--output", str(PREDICTIONS_ZS.with_stem("predictions")),
                "--engine", args.engine,
                "--mode", "zero_shot",
                "--concurrency", str(max(args.concurrency // 2, 4)),
            ],
        )

    # ── Stage 3: Evaluate ─────────────────────────────────────────────────────
    if run_evaluate and PREDICTIONS_COT.exists():
        eval_cmd = [
            sys.executable, "src/evaluate.py",
            "--predictions", str(PREDICTIONS_COT),
            "--output-dir", str(RESULTS_DIR),
        ]
        if PREDICTIONS_ZS.exists() and not args.skip_zero_shot:
            eval_cmd += ["--compare", str(PREDICTIONS_ZS)]
        if args.linter_baseline:
            eval_cmd.append("--linter-baseline")
        run_stage("BENCHMARKING SUITE", eval_cmd)

    print("\n" + "═" * 60)
    print("  ✅ TraceAudit Pipeline Complete")
    print("═" * 60)
    print(f"  Results saved to: {RESULTS_DIR}/")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
