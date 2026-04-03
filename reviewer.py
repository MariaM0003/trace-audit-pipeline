"""
TraceAudit — CoT Reasoning Engine (reviewer.py)
================================================
Two-stage Chain-of-Thought reviewer that detects semantic bugs via
Virtual Execution Tracing.

CLI Usage:
    python src/reviewer.py --dataset results/dataset.jsonl --output results/predictions.jsonl
    python src/reviewer.py --dataset results/dataset.jsonl --mode zero_shot  # for A/B comparison

Architecture:
    Stage 1 (TRACE): The model performs a step-by-step virtual execution,
                     tracking variable state, control flow, and resource lifecycle.
    Stage 2 (AUDIT): The model compares the trace against the style_guide.json
                     rules and produces a structured classification.

    "zero_shot" mode: Skips the trace and asks for a direct classification.
    This is used by evaluate.py to prove CoT's value over naive prompting.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# ── Path bootstrap ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reviewer")

# ── Constants ────────────────────────────────────────────────────────────────
BugType = Literal[
    "RESOURCE_LEAK",
    "BOUNDARY_VIOLATION",
    "INSECURE_DATA_FLOW",
    "CONCURRENCY_HAZARD",
    "IMPROPER_ERROR_HANDLING",
    "NO_BUG",
]

ReviewMode = Literal["cot", "zero_shot"]

# ── Load style guide once ─────────────────────────────────────────────────────
def _load_style_guide() -> str:
    guide_path = PROJECT_ROOT / "data" / "style_guide.json"
    with guide_path.open() as f:
        guide = json.load(f)
    lines = []
    for cat in guide["bug_categories"].values():
        lines.append(f"### {cat['id']} (severity: {cat['severity']})")
        lines.append(f"  Description: {cat['description']}")
        lines.append("  Detection signals:")
        for s in cat["detection_signals"]:
            lines.append(f"    - {s}")
        lines.append("  Trace checklist:")
        for c in cat["trace_checklist"]:
            lines.append(f"    - {c}")
        lines.append("")
    return "\n".join(lines)


STYLE_GUIDE_TEXT = _load_style_guide()

# ── Pydantic output schema ────────────────────────────────────────────────────
class TraceStep(BaseModel):
    step: int
    description: str = Field(..., description="What happens at this step in the virtual execution")
    variables: dict[str, str] = Field(
        default_factory=dict,
        description="Snapshot of key variable states after this step",
    )
    flag: str | None = Field(
        None,
        description="Optional flag if something suspicious is noticed at this step",
    )


class ReviewPrediction(BaseModel):
    """The full structured output from the CoT reviewer."""

    function_name: str
    mode: ReviewMode
    trace: list[TraceStep] | None = Field(
        None,
        description="Virtual execution trace (CoT mode only)",
    )
    audit_notes: str = Field(
        ...,
        description="Narrative comparison of trace findings against the style guide rules",
    )
    predicted_bug_type: BugType = Field(
        ...,
        description="The model's best classification of the bug",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        ...,
        description="Model's confidence in the classification",
    )
    predicted_bug_line: int | None = Field(
        None,
        description="Best guess at the 1-indexed line number of the bug",
    )
    reasoning_summary: str = Field(
        ...,
        description="One-sentence explanation of why this is the predicted bug type",
    )


# ── System Prompts ────────────────────────────────────────────────────────────

COT_SYSTEM_PROMPT = f"""\
You are a senior Python security engineer performing a deep semantic code review.
Your methodology is the Virtual Execution Trace — you do NOT just read the code,
you mentally RUN it, tracking every variable's state, every resource acquired,
and every edge case.

## Your Two-Stage Process

### STAGE 1: VIRTUAL EXECUTION TRACE
Simulate executing the function step-by-step.
For each logical step, record:
  - What operation is being performed
  - The state of key variables
  - Whether any resource is being acquired or released
  - Whether any sensitive data changes hands
  - Whether the current path is the "happy path" or an exception/edge case path

Pay special attention to:
  - What happens BEFORE and AFTER exception points
  - What happens on the FIRST and LAST iteration of any loop
  - What happens with EMPTY or SINGLE-ELEMENT inputs
  - What happens across MULTIPLE THREADS (if applicable)

### STAGE 2: AUDIT AGAINST STYLE GUIDE
After building your trace, compare it against these bug detection rules:

{STYLE_GUIDE_TEXT}

Based on the trace and audit, produce your final classification.

## Output Format
Respond with a single JSON object — no markdown, no preamble:

{{
  "function_name": "<name of the function reviewed>",
  "mode": "cot",
  "trace": [
    {{
      "step": 1,
      "description": "<what happens at this step>",
      "variables": {{"var_name": "state_description", ...}},
      "flag": "<optional: suspicious finding or null>"
    }},
    ...
  ],
  "audit_notes": "<narrative: what does the trace reveal when compared against the style guide?>",
  "predicted_bug_type": "<RESOURCE_LEAK | BOUNDARY_VIOLATION | INSECURE_DATA_FLOW | CONCURRENCY_HAZARD | IMPROPER_ERROR_HANDLING | NO_BUG>",
  "confidence": "<low | medium | high>",
  "predicted_bug_line": <integer or null>,
  "reasoning_summary": "<one sentence: why this bug type>"
}}
"""

ZERO_SHOT_SYSTEM_PROMPT = f"""\
You are a Python code reviewer. Examine the function below and classify any bug present.

Valid bug types: RESOURCE_LEAK, BOUNDARY_VIOLATION, INSECURE_DATA_FLOW, CONCURRENCY_HAZARD, IMPROPER_ERROR_HANDLING, NO_BUG

Style guide for reference:
{STYLE_GUIDE_TEXT}

Respond with a single JSON object — no markdown, no preamble:

{{
  "function_name": "<name>",
  "mode": "zero_shot",
  "trace": null,
  "audit_notes": "<brief notes>",
  "predicted_bug_type": "<bug type>",
  "confidence": "<low | medium | high>",
  "predicted_bug_line": <integer or null>,
  "reasoning_summary": "<one sentence>"
}}
"""


def build_review_prompt(mutated_code: str, function_name: str) -> str:
    return f"""\
## Function to Review: `{function_name}`

```python
{mutated_code}
```

Apply your methodology and produce the JSON review.
"""


# ── OpenAI async call ─────────────────────────────────────────────────────────
async def review_one(
    client: AsyncOpenAI,
    record: dict,
    model: str,
    mode: ReviewMode,
    semaphore: asyncio.Semaphore,
    retries: int = 3,
) -> dict | None:
    """Run CoT or zero-shot review on one mutated record."""
    function_name = record.get("original_function_name", "unknown")
    mutated_code = record.get("mutated_code", "")
    system_prompt = COT_SYSTEM_PROMPT if mode == "cot" else ZERO_SHOT_SYSTEM_PROMPT

    async with semaphore:
        for attempt in range(1, retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=0.0,  # deterministic for evaluation
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": build_review_prompt(mutated_code, function_name),
                        },
                    ],
                    timeout=90,
                )
                raw = response.choices[0].message.content
                prediction_data = json.loads(raw)
                prediction = ReviewPrediction(**prediction_data)

                # Merge prediction with original record for evaluation
                output = {
                    **record,
                    "prediction": prediction.model_dump(),
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                }
                return output

            except Exception as exc:
                log.warning(
                    "Review attempt %d/%d failed for %s (%s): %s",
                    attempt, retries, function_name, mode, exc,
                )
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)

        log.error("All review retries exhausted for %s", function_name)
        return None


# ── Pipeline ──────────────────────────────────────────────────────────────────
async def run_review_pipeline(
    dataset_path: Path,
    output_path: Path,
    model: str,
    mode: ReviewMode,
    concurrency: int,
    limit: int | None,
) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found. Add it to your .env file.")

    # Load dataset
    records: list[dict] = []
    with dataset_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if limit:
        records = records[:limit]

    log.info(
        "Starting review: %d samples | model=%s | mode=%s | concurrency=%d",
        len(records), model, mode, concurrency,
    )

    # Clear output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    written = 0
    start = time.time()

    async def process_record(record: dict, idx: int) -> None:
        nonlocal written
        result = await review_one(client, record, model, mode, semaphore)
        if result:
            async with write_lock:
                with output_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
                written += 1
                if written % 10 == 0:
                    elapsed = time.time() - start
                    log.info(
                        "Review progress: %d/%d (%.1f/sec)",
                        written, len(records), written / elapsed,
                    )

    await asyncio.gather(*[process_record(r, i) for i, r in enumerate(records)])

    elapsed = time.time() - start
    log.info(
        "✓ Review complete. %d/%d predictions written in %.1fs → %s",
        written, len(records), elapsed, output_path,
    )

    # Token cost estimate
    _print_token_summary(output_path)


def _print_token_summary(path: Path) -> None:
    total_prompt = 0
    total_completion = 0
    count = 0
    try:
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                usage = r.get("usage", {})
                total_prompt += usage.get("prompt_tokens", 0)
                total_completion += usage.get("completion_tokens", 0)
                count += 1
    except Exception:
        return

    # gpt-4o-mini pricing (approximate as of 2024): $0.15/1M input, $0.60/1M output
    input_cost = (total_prompt / 1_000_000) * 0.15
    output_cost = (total_completion / 1_000_000) * 0.60

    print("\n── Token Usage Summary ──────────────────────────────")
    print(f"  Reviews completed : {count}")
    print(f"  Prompt tokens     : {total_prompt:,}  (≈ ${input_cost:.4f})")
    print(f"  Completion tokens : {total_completion:,}  (≈ ${output_cost:.4f})")
    print(f"  Total tokens      : {total_prompt + total_completion:,}  (≈ ${input_cost + output_cost:.4f})")
    print(f"  Avg tokens/sample : {(total_prompt + total_completion) // max(count, 1):,}")
    print("─────────────────────────────────────────────────────\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="TraceAudit CoT Reviewer — classify bugs via Virtual Execution Tracing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "results" / "dataset.jsonl"),
        help="Path to dataset.jsonl produced by generator.py",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "predictions.jsonl"),
        help="Output path for predictions",
    )
    parser.add_argument("--engine", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument(
        "--mode",
        choices=["cot", "zero_shot"],
        default="cot",
        help="Review mode: 'cot' (full trace) or 'zero_shot' (direct classification)",
    )
    parser.add_argument("--concurrency", type=int, default=8, help="Max parallel API calls")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (for testing)")
    args = parser.parse_args()

    output_filename = Path(args.output)
    if args.mode == "zero_shot":
        # Auto-suffix the output so both modes can be compared
        output_filename = output_filename.with_stem(output_filename.stem + "_zero_shot")

    asyncio.run(
        run_review_pipeline(
            dataset_path=Path(args.dataset),
            output_path=output_filename,
            model=args.engine,
            mode=args.mode,
            concurrency=args.concurrency,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
