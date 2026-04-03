"""
TraceAudit — Mutation Engine (generator.py)
============================================
Injects one of 5 semantic bug types into clean Python seed functions.

CLI Usage:
    python src/generator.py --samples 100 --engine gpt-4o-mini --output results/dataset.jsonl
    python src/generator.py --samples 500 --engine gpt-4o-mini --concurrency 20

Architecture:
    1. Load seed functions from data/seeds.py
    2. For each sample: pick a seed + bug type at random
    3. Call GPT-4o-mini with the adversarial mutation prompt (async, batched)
    4. Validate the JSON response with Pydantic
    5. Write to dataset.jsonl (thread-safe streaming writes)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator

# ── Path bootstrap (run from project root or src/) ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generator")

# ── Constants ────────────────────────────────────────────────────────────────
BugType = Literal[
    "RESOURCE_LEAK",
    "BOUNDARY_VIOLATION",
    "INSECURE_DATA_FLOW",
    "CONCURRENCY_HAZARD",
    "IMPROPER_ERROR_HANDLING",
]

ALL_BUG_TYPES: list[BugType] = [
    "RESOURCE_LEAK",
    "BOUNDARY_VIOLATION",
    "INSECURE_DATA_FLOW",
    "CONCURRENCY_HAZARD",
    "IMPROPER_ERROR_HANDLING",
]

# ── Pydantic output schema ────────────────────────────────────────────────────
class MutationRecord(BaseModel):
    """A single mutated sample with ground-truth metadata."""

    original_function_name: str = Field(..., description="Name of the seed function")
    original_code: str = Field(..., description="Original clean source code")
    mutated_code: str = Field(..., description="Source with the injected bug")
    bug_type: BugType = Field(..., description="Canonical bug category label")
    bug_line_number: int = Field(..., ge=1, description="1-indexed line in mutated_code where the bug manifests")
    bug_description: str = Field(
        ...,
        description="Concise English explanation of exactly what was changed and why it is a bug",
    )
    is_detectable_by_linter: bool = Field(
        ...,
        description="True only if flake8 or bandit would flag this exact issue",
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ...,
        description="Estimated difficulty for an LLM to detect: easy=obvious, hard=requires deep trace",
    )

    @field_validator("mutated_code")
    @classmethod
    def must_be_valid_python(cls, v: str) -> str:
        try:
            compile(v, "<mutated>", "exec")
        except SyntaxError as exc:
            raise ValueError(f"mutated_code contains a SyntaxError: {exc}") from exc
        return v


# ── Adversarial Mutation Prompt ───────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an adversarial code mutation engine for a security and reliability research project.
Your sole purpose is to inject exactly ONE subtle semantic bug into a provided Python function.

## CRITICAL RULES — READ CAREFULLY

1. **SYNTACTICALLY VALID**: The mutated code MUST compile without errors. No syntax mistakes.
2. **FUNCTIONALLY SIMILAR**: The function signature, imports, and overall shape must remain intact.
   A quick visual scan should NOT reveal anything wrong.
3. **SEMANTICALLY SUBTLE**: The bug must NOT be detectable by flake8, pylint, or bandit in most cases.
   It must require step-by-step logical reasoning (tracing variable state, control flow, and edge cases).
4. **SINGLE MUTATION**: Change AS LITTLE AS POSSIBLE. One targeted mutation, not a rewrite.
5. **NO OBVIOUS TELLS**: Do NOT add obvious comments, unusual variable names, or restructure the code.

## Bug Type Definitions and Mutation Strategies

### RESOURCE_LEAK
- **Goal**: Ensure the resource is NOT released on at least one code path.
- **Techniques**: Remove the `with` statement and use explicit open/close, then omit the close in the
  exception path. Or acquire a threading.Lock with `.acquire()` but have a code path that returns early
  before `.release()`. Do NOT use a simple `with` block — that would be an obvious fix.
- **Subtlety target**: The bug should only manifest when an exception is thrown mid-function,
  or when an early return is taken.

### BOUNDARY_VIOLATION
- **Goal**: Introduce an off-by-one or missing-guard error that is silent under normal input.
- **Techniques**: Change a loop range from `range(len(x)-1)` to `range(len(x))`, or change a
  slice from `[1:]` to `[0:]` or `[:n]` to `[:n-1]`. The mutation must be a single character/integer change.
- **Subtlety target**: The function produces correct results for the "happy path" (e.g., list with 5 items)
  but fails on the last element or an empty list.

### INSECURE_DATA_FLOW
- **Goal**: Route sensitive data into an insecure sink.
- **Techniques**: Add a single `logging.debug(...)` or `print(...)` call that includes a password, token,
  or key variable. Or change `hmac.compare_digest(a, b)` to `a == b` (timing attack).
  Or embed a secret in a URL query string.
- **Subtlety target**: The logic is functionally correct; the vulnerability is in the data handling.

### CONCURRENCY_HAZARD
- **Goal**: Create an unsynchronized access to shared mutable state.
- **Techniques**: Remove the `with lock:` wrapper from a results list append, or remove the lock
  around a counter increment. Or change a local variable to a shared one. Or introduce a
  check-then-act race (read a value, then modify it without holding the lock across both operations).
- **Subtlety target**: The code works correctly 99% of the time; the race only manifests under load.

### IMPROPER_ERROR_HANDLING
- **Goal**: Swallow an exception and allow corrupt state to propagate silently.
- **Techniques**: Wrap a critical operation in `try/except Exception: pass` or `try/except Exception: return []`.
  Or catch a specific exception and silently continue without re-raising. Or remove a finally block.
- **Subtlety target**: The function appears to handle errors gracefully; the issue is that the caller
  can't distinguish between a legitimate empty result and a failure.

## Output Format
Respond with a single valid JSON object matching this schema exactly — no markdown, no explanation outside JSON:

{
  "original_function_name": "<string>",
  "original_code": "<the ORIGINAL unmodified code, verbatim>",
  "mutated_code": "<the mutated code with EXACTLY ONE subtle bug>",
  "bug_type": "<one of: RESOURCE_LEAK | BOUNDARY_VIOLATION | INSECURE_DATA_FLOW | CONCURRENCY_HAZARD | IMPROPER_ERROR_HANDLING>",
  "bug_line_number": <integer — 1-indexed line in mutated_code where the core bug manifests>,
  "bug_description": "<one sentence: what exactly changed and WHY it is a semantic bug>",
  "is_detectable_by_linter": <true | false>,
  "difficulty": "<easy | medium | hard>"
}
"""

def build_user_prompt(seed: dict, bug_type: BugType) -> str:
    return f"""\
## Target Bug Type: {bug_type}

## Seed Function to Mutate:
```python
{seed["source"]}
```

Inject a **{bug_type}** bug. Remember: ONE subtle change. Syntactically valid. Visually similar.
Return only the JSON object described in your instructions.
"""


# ── OpenAI async call ─────────────────────────────────────────────────────────
async def mutate_one(
    client: AsyncOpenAI,
    seed: dict,
    bug_type: BugType,
    model: str,
    semaphore: asyncio.Semaphore,
    retries: int = 3,
) -> MutationRecord | None:
    """Call the LLM to mutate one seed function; validates with Pydantic."""
    async with semaphore:
        for attempt in range(1, retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=0.4,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(seed, bug_type)},
                    ],
                    timeout=60,
                )
                raw = response.choices[0].message.content
                data = json.loads(raw)
                # Enforce the original code and name from our known-good seed
                data["original_function_name"] = seed["name"]
                data["original_code"] = seed["source"]
                record = MutationRecord(**data)
                return record

            except Exception as exc:
                log.warning(
                    "Attempt %d/%d failed for %s+%s: %s",
                    attempt, retries, seed["name"], bug_type, exc,
                )
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)

        log.error("All retries exhausted for %s+%s", seed["name"], bug_type)
        return None


# ── Writer (thread-safe streaming JSONL) ────────────────────────────────────
class StreamingWriter:
    """Writes validated records to JSONL immediately (no buffering)."""

    def __init__(self, output_path: Path) -> None:
        self.path = output_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._count = 0

    async def write(self, record: MutationRecord) -> None:
        async with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
            self._count += 1

    @property
    def count(self) -> int:
        return self._count


# ── Main pipeline ─────────────────────────────────────────────────────────────
async def run_pipeline(
    num_samples: int,
    model: str,
    output_path: Path,
    concurrency: int,
) -> None:
    from data.seeds import SEED_FUNCTIONS

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found. Add it to your .env file.")

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    writer = StreamingWriter(output_path)

    # Clear existing output to allow fresh runs
    if output_path.exists():
        output_path.unlink()
        log.info("Cleared existing output file: %s", output_path)

    # Build task list: balance across all bug types and all seeds
    random.seed(42)
    tasks_spec: list[tuple[dict, BugType]] = []
    for i in range(num_samples):
        seed = random.choice(SEED_FUNCTIONS)
        bug_type = ALL_BUG_TYPES[i % len(ALL_BUG_TYPES)]
        tasks_spec.append((seed, bug_type))

    log.info(
        "Starting pipeline: %d samples | model=%s | concurrency=%d | output=%s",
        num_samples, model, concurrency, output_path,
    )
    start = time.time()

    async def task_wrapper(spec: tuple[dict, BugType], idx: int) -> None:
        seed, bug_type = spec
        record = await mutate_one(client, seed, bug_type, model, semaphore)
        if record:
            await writer.write(record)
            if writer.count % 10 == 0:
                elapsed = time.time() - start
                rate = writer.count / elapsed
                log.info(
                    "Progress: %d/%d written (%.1f samples/sec)",
                    writer.count, num_samples, rate,
                )

    coroutines = [task_wrapper(spec, i) for i, spec in enumerate(tasks_spec)]
    await asyncio.gather(*coroutines)

    elapsed = time.time() - start
    log.info(
        "✓ Pipeline complete. %d/%d records written in %.1fs → %s",
        writer.count, num_samples, elapsed, output_path,
    )

    # Print a brief summary
    _print_summary(output_path)


def _print_summary(path: Path) -> None:
    """Print bug-type distribution of the generated dataset."""
    from collections import Counter
    counts: Counter = Counter()
    try:
        with path.open() as f:
            for line in f:
                record = json.loads(line)
                counts[record["bug_type"]] += 1
    except Exception:
        return

    print("\n── Dataset Summary ──────────────────────────")
    total = sum(counts.values())
    for bug_type in ALL_BUG_TYPES:
        n = counts.get(bug_type, 0)
        bar = "█" * n + "░" * max(0, 20 - n)
        print(f"  {bug_type:<30} {n:>4} ({n/total*100:.1f}%) {bar}")
    print(f"  {'TOTAL':<30} {total:>4}")
    print("─────────────────────────────────────────────\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="TraceAudit Mutation Engine — generate synthetic buggy Python dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=50, help="Number of mutated samples to generate")
    parser.add_argument("--engine", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "dataset.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--concurrency", type=int, default=10, help="Max parallel API calls")
    args = parser.parse_args()

    asyncio.run(
        run_pipeline(
            num_samples=args.samples,
            model=args.engine,
            output_path=Path(args.output),
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
