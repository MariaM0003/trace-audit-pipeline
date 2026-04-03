# 🔬 TraceAudit

> **A production-grade Synthetic Data Pipeline and Chain-of-Thought Reasoning Engine for detecting deep semantic bugs in Python — the kind standard linters miss.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991?style=flat-square&logo=openai)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## What This Is

TraceAudit is a three-component system that:

1. **Generates** a large-scale synthetic dataset of subtly-buggy Python functions using an adversarial LLM mutation prompt.
2. **Reviews** each buggy function using a two-stage Chain-of-Thought reasoning engine that simulates virtual code execution before classifying the bug.
3. **Benchmarks** the reviewer's performance against ground-truth labels, with precision/recall/F1 metrics, and charts that prove CoT tracing outperforms naive zero-shot classification.

---

## The Core Problem This Solves

Standard linters (`flake8`, `pylint`, `bandit`) are syntax-aware but semantically blind.
They cannot detect:

| Bug Class | Why Linters Fail |
|---|---|
| `RESOURCE_LEAK` | Can't trace resource lifetime across branch conditions |
| `BOUNDARY_VIOLATION` | Can't evaluate runtime index arithmetic |
| `INSECURE_DATA_FLOW` | Can't follow tainted data through multiple calls |
| `CONCURRENCY_HAZARD` | Can't reason about thread interleaving |
| `IMPROPER_ERROR_HANDLING` | Can't assess what "silent failure" means to the caller |

TraceAudit proves that an LLM can find these bugs — but *only* if it reasons step-by-step.

---

## Architecture

```
trace-audit/
│
├── data/
│   ├── seeds.py          # 20 diverse, clean Python "seed" functions (the healthy patients)
│   └── style_guide.json  # Canonical rules for the 5 bug categories (the auditor's rulebook)
│
├── src/
│   ├── generator.py      # Mutation Engine: injects bugs, outputs dataset.jsonl
│   ├── reviewer.py       # CoT Reasoning Engine: traces execution, classifies bugs
│   └── evaluate.py       # Benchmarking Suite: F1, confusion matrix, comparison charts
│
├── results/              # Auto-generated outputs (gitignored)
│   ├── dataset.jsonl
│   ├── predictions.jsonl
│   ├── predictions_zero_shot.jsonl
│   ├── confusion_matrix.png
│   ├── f1_comparison.png
│   ├── cost_benefit.png
│   └── metrics_report.json
│
├── tests/
│   └── test_generator.py
│
├── run_pipeline.py       # Master orchestrator: runs all 3 stages end-to-end
└── requirements.txt
```

---

## How It Works

### Stage 1 — The Mutation Engine (`generator.py`)

The generator takes each clean seed function and calls `gpt-4o-mini` with an **adversarial mutation prompt** that demands:

- **Syntactically valid** output (compilable Python)
- **Semantically subtle** bugs (no obvious tells)
- A **single targeted mutation** (not a rewrite)
- Bugs that are **invisible to flake8/bandit**

Each record in `dataset.jsonl` contains:

```json
{
  "original_function_name": "producer_consumer",
  "original_code": "...",
  "mutated_code": "...",
  "bug_type": "CONCURRENCY_HAZARD",
  "bug_line_number": 23,
  "bug_description": "Removed the 'with lock:' guard around results.append(), creating a data race when multiple consumer threads write simultaneously.",
  "is_detectable_by_linter": false,
  "difficulty": "hard"
}
```

The pipeline uses `asyncio` with a configurable concurrency semaphore to handle 100+ samples efficiently.

---

### Stage 2 — The CoT Reasoning Engine (`reviewer.py`)

The reviewer uses a **two-stage chain-of-thought** methodology:

#### Stage 1: Virtual Execution Trace
The model doesn't just *read* the code — it *runs* it mentally, step-by-step, recording:
- Variable state at each step
- Resources acquired and released
- Behaviour on the EXCEPTION path (not just the happy path)
- Behaviour on FIRST and LAST loop iterations

#### Stage 2: Style Guide Audit
The trace findings are compared against `style_guide.json` — a formal rulebook with detection signals and trace checklists for each bug type.

The output is a fully structured Pydantic-validated prediction:

```json
{
  "predicted_bug_type": "CONCURRENCY_HAZARD",
  "confidence": "high",
  "predicted_bug_line": 23,
  "trace": [
    {
      "step": 4,
      "description": "consumer() thread appends to results list",
      "variables": {"results": "shared list, no lock held"},
      "flag": "⚠️ Shared mutable state modified outside lock context"
    }
  ],
  "audit_notes": "Trace reveals unsynchronized write to shared 'results' list. Style guide rule CONCURRENCY_HAZARD signal 1 matches: 'Shared list/dict mutated inside thread without a Lock'.",
  "reasoning_summary": "The results.append(result) call on line 23 occurs inside a consumer thread without holding the shared lock, creating a data race."
}
```

---

### Stage 3 — The Benchmarking Suite (`evaluate.py`)

Two key experiments are run:

**Experiment 1: CoT vs Zero-Shot**
The same dataset is reviewed with both methods. This proves that the Virtual Execution Trace architecture meaningfully improves F1-score — at a quantifiable token cost.

**Experiment 2: AI vs Linter Baseline**
`bandit` is run against every mutated function. The results show which bug types are invisible to static analysis but detectable by the CoT reviewer.

---

## Setup (Windows/WSL)

```bash
# 1. Clone and enter the repo
git clone https://github.com/yourusername/trace-audit.git
cd trace-audit

# 2. Create and activate a virtual environment (WSL/Linux)
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell — if not using WSL)
# python -m venv .venv
# .venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-your-key-here
```

---

## Running the Pipeline

### Quick start (50 samples, ~5 min, ~$0.10)
```bash
python run_pipeline.py --samples 50
```

### Portfolio-scale run (500 samples, recommended for README charts)
```bash
python run_pipeline.py --samples 500 --engine gpt-4o-mini --concurrency 20
```

### Run individual stages
```bash
# 1. Generate the dataset only
python src/generator.py --samples 100 --engine gpt-4o-mini --output results/dataset.jsonl

# 2. Review with CoT
python src/reviewer.py --dataset results/dataset.jsonl --mode cot

# 3. Review with Zero-Shot (for A/B comparison)
python src/reviewer.py --dataset results/dataset.jsonl --mode zero_shot

# 4. Evaluate both + generate all charts
python src/evaluate.py \
    --predictions results/predictions.jsonl \
    --compare results/predictions_zero_shot.jsonl \
    --linter-baseline
```

### Run tests
```bash
pytest tests/ -v
```

---

## Output Charts

After running the full pipeline, `results/` will contain:

| File | What It Shows |
|---|---|
| `confusion_matrix.png` | Which bug types the AI confuses with each other (normalized recall heatmap) |
| `f1_comparison.png` | Per-class F1: CoT vs Zero-Shot, with Δ gain annotations |
| `cost_benefit.png` | Scatter: token cost vs detection quality — the "is CoT worth it?" answer |
| `linter_comparison.png` | AI detection rate vs linter detection rate per bug class |
| `metrics_report.json` | Full precision/recall/F1 with per-difficulty and per-confidence breakdowns |

---

## Sample Results

*(Example values — run the pipeline to generate your own)*

| Metric | CoT (Trace) | Zero-Shot | Δ Gain |
|---|---|---|---|
| Macro F1 | 0.81 | 0.63 | **+0.18** |
| Accuracy | 83% | 65% | **+18%** |
| Avg Tokens/Sample | 1,840 | 420 | +1,420 |

| Bug Type | CoT F1 | Zero-Shot F1 | Linter Detects |
|---|---|---|---|
| CONCURRENCY_HAZARD | 0.86 | 0.51 | ~0% |
| RESOURCE_LEAK | 0.79 | 0.68 | ~15% |
| INSECURE_DATA_FLOW | 0.88 | 0.72 | ~20% |
| BOUNDARY_VIOLATION | 0.77 | 0.61 | ~0% |
| IMPROPER_ERROR_HANDLING | 0.74 | 0.62 | ~10% |

---

## For Recruiters: What This Demonstrates

### Synthetic Data Engineering
- Designed an adversarial prompt that reliably generates **subtle, syntactically-valid** semantic bugs — not random noise, but targeted mutations that mimic real engineering mistakes.
- Built a streaming `asyncio` pipeline that generates **100+ samples concurrently** with retry logic, Pydantic validation, and real-time progress reporting.
- Every dataset record includes ground-truth labels, difficulty scores, and linter-detectability flags — enabling rigorous evaluation.

### Reasoning-as-a-Service (RaaS) Architecture
- Implemented a **two-stage Chain-of-Thought** review system that separates *observation* (the trace) from *classification* (the audit) — the same separation of concerns used in expert human code review.
- Proved empirically (with F1 metrics) that structured reasoning outperforms direct classification for this task.
- Designed the system as a **service**: the reviewer takes any Python function, traces it, and returns a structured JSON prediction — this is callable as a standalone API.

### Production Engineering Practices
- **Pydantic v2** schemas enforce structured outputs at every stage, preventing silent data corruption from malformed LLM responses.
- **Async concurrency** with semaphores handles rate limits gracefully without blocking.
- **Full observability**: token counts, cost estimates, and per-sample latency are logged and reported.
- **A/B testing infrastructure**: the same dataset is evaluated with two strategies (CoT vs zero-shot) to produce statistically grounded comparisons.
- **Separation of concerns**: generate, review, and evaluate are independent modules with clean interfaces and a single orchestrator.

---

## License

MIT

---

*Built with Python 3.10+ · OpenAI gpt-4o-mini · scikit-learn · Pydantic v2 · asyncio*
