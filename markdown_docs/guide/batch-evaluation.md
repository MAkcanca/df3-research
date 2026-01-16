# Batch Evaluation

Run systematic evaluations across datasets to measure DF3 performance with comprehensive metrics.

---

## Overview

The batch evaluator (`scripts/evaluate_llms.py`) enables:

- Testing multiple models side-by-side
- Comparing tools vs no-tools modes
- Computing accuracy, precision, recall, F1, and more
- Generating reproducible benchmarks

---

## Dataset Format

Create a JSONL file with one record per line:

```json
{"id": "sample-001", "image": "images/photo1.jpg", "label": "real"}
{"id": "sample-002", "image": "images/photo2.jpg", "label": "fake"}
{"id": "sample-003", "image": "images/photo3.png", "label": "real"}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for the sample |
| `image` | string | Path to image file (relative to JSONL or absolute) |
| `label` | string | Ground truth: `"real"` or `"fake"` |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `meta` | object | Additional metadata (preserved in results) |

---

## Basic Usage

```powershell
python scripts/evaluate_llms.py \
    --dataset data/eval_dataset.jsonl \
    --models gpt-5.1 \
    --tools both
```

### Command-Line Options

```
Required:
  --dataset PATH        Path to JSONL dataset file

Model Options:
  --models LIST         Comma-separated model names (e.g., gpt-5.1,gpt-5-mini)
  --vision-model MODEL  Override vision step model
  --structuring-model MODEL
                        Override structuring step model

Tool Options:
  --tools MODE          "tools", "no-tools", or "both" (default: both)

Execution Options:
  --temperature FLOAT   LLM temperature (default: 0.0)
  --max-iterations INT  Max tool calls per sample (default: 15)
  --trials INT          Number of trials per config (default: 1)
  --num-workers INT     Parallel workers (default: 1)
  --limit INT           Max samples to evaluate
  --shuffle             Shuffle dataset before evaluation
  --seed INT            Random seed for shuffling

Cache Options:
  --enable-tool-cache   Enable caching (default: True)
  --disable-tool-cache  Disable caching
  --tool-cache-dir PATH Custom cache directory

Output Options:
  --output PATH         Per-sample results file (default: eval_results.jsonl)
  --metrics-output PATH Aggregated metrics file (default: eval_metrics.json)

Provider Options:
  --provider NAME       "openai" or "openrouter" (default: openai)
  --api-key KEY         API key (overrides env var)
  --base-url URL        Custom API endpoint
```

---

## Common Evaluation Scenarios

### Compare Models

Test multiple models on the same dataset:

```powershell
python scripts/evaluate_llms.py \
    --dataset data/benchmark.jsonl \
    --models gpt-5.1,gpt-5-mini,gpt-5.2 \
    --tools both \
    --output results/model_comparison.jsonl
```

### Tools vs No-Tools

Compare tool-augmented vs vision-only:

```powershell
python scripts/evaluate_llms.py \
    --dataset data/benchmark.jsonl \
    --models gpt-5.1 \
    --tools both \
    --output results/tools_comparison.jsonl
```

### Multi-Trial Evaluation

Run multiple trials to estimate variance:

```powershell
python scripts/evaluate_llms.py \
    --dataset data/benchmark.jsonl \
    --models gpt-5.1 \
    --trials 5 \
    --temperature 0.0 \
    --output results/multi_trial.jsonl
```

### Parallel Evaluation

Speed up with multiple workers (careful with rate limits):

```powershell
python scripts/evaluate_llms.py \
    --dataset data/benchmark.jsonl \
    --models gpt-5.1 \
    --num-workers 4 \
    --output results/parallel.jsonl
```

### Vision Model Override

Use different models for vision vs agent:

```powershell
python scripts/evaluate_llms.py \
    --dataset data/benchmark.jsonl \
    --models gpt-5.1 \
    --vision-model gpt-5.2 \
    --structuring-model gpt-5-mini \
    --output results/mixed_models.jsonl
```

---

## Output Files

### Per-Sample Results (`--output`)

JSONL file with detailed results for each sample:

```json
{
  "id": "sample-001",
  "model": "gpt-5.1",
  "use_tools": true,
  "trial": 0,
  "label": "fake",
  "image": "/path/to/image.jpg",
  "prediction": "fake",
  "confidence": 0.87,
  "rationale": "Multiple anatomical anomalies...",
  "tool_usage": ["perform_trufor", "perform_ela"],
  "tool_results": [...],
  "latency_seconds": 12.3,
  "timings": {
    "vision_llm_seconds": 3.2,
    "agent_graph_seconds": 8.1,
    "total_seconds": 12.3
  },
  "models": {
    "agent": "gpt-5.1",
    "vision": "gpt-5.2",
    "structuring": "gpt-5-mini"
  }
}
```

### Aggregated Metrics (`--metrics-output`)

JSON file with computed metrics per configuration:

```json
{
  "_meta": {
    "temperature": 0.0,
    "max_iterations": 15,
    "trials": 1,
    "generated_at_unix": 1736956800
  },
  "gpt-5.1|tools": {
    "per_trial": {
      "0": {
        "total": 500,
        "correct": 425,
        "accuracy": 0.85,
        "accuracy_answered": 0.89,
        "coverage": 0.95,
        "precision_fake": 0.88,
        "recall_fake": 0.84,
        "f1_fake": 0.86,
        ...
      }
    },
    "summary": {
      "mean": {"accuracy": 0.85, ...},
      "std": {"accuracy": 0.02, ...}
    }
  }
}
```

---

## Understanding Metrics

### Core Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **accuracy** | (TP + TN) / N | Overall correct rate (abstentions count as wrong) |
| **accuracy_answered** | (TP + TN) / answered | Correct rate among answered samples |
| **coverage** | answered / N | Fraction of samples with verdict |

### Class-Specific Metrics

| Metric | Description |
|--------|-------------|
| **precision_fake** | TP / (TP + FP) — How often "fake" predictions are correct |
| **recall_fake** | TP / (TP + FN) — What fraction of fakes are caught |
| **f1_fake** | Harmonic mean of precision and recall for fake class |
| **precision_real** | TN / (TN + FN) — How often "real" predictions are correct |
| **recall_real** | TN / (TN + FP) — What fraction of reals are passed |
| **f1_real** | Harmonic mean for real class |

### Triage Metrics

| Metric | Description |
|--------|-------------|
| **fake_slip_rate** | FN / N_fake — Fakes incorrectly passed as real |
| **real_false_flag_rate** | FP / N_real — Reals incorrectly flagged as fake |
| **fake_catch_rate** | TP / N_fake — Fakes correctly caught |
| **real_pass_rate** | TN / N_real — Reals correctly passed |
| **abstain_rate** | N_uncertain / N — Rate of "uncertain" verdicts |

### Balanced Metrics

| Metric | Description |
|--------|-------------|
| **balanced_accuracy** | (TPR_fake + TPR_real) / 2 — Balanced across classes |
| **avg_confidence** | Mean confidence score on answered samples |

---

## Regenerating Tables

After evaluation, regenerate summary tables:

```powershell
python scripts/summarize_results.py \
    --results-dir results \
    --out artifacts/eval_summary.json \
    --out-md markdown_docs/evaluation_report.generated.md
```

This produces:

- Machine-readable JSON with all metrics
- Markdown tables for documentation
- Paired statistical comparisons

---

## Cache Management

### Enable/Disable

```powershell
# With caching (default - faster repeat runs)
python scripts/evaluate_llms.py --enable-tool-cache ...

# Without caching (for latency measurements)
python scripts/evaluate_llms.py --disable-tool-cache ...
```

### Cache Location

Default: `.tool_cache/` in project root

Override:
```powershell
python scripts/evaluate_llms.py --tool-cache-dir /path/to/cache ...
```

### Cache Statistics

View cache status:

```python
from src.tools.forensic.cache import get_cache

cache = get_cache()
stats = cache.get_stats()
print(f"Entries: {stats['entry_count']}")
print(f"Size: {stats['total_size_mb']:.1f} MB")
```

!!! warning "Caching and Latency"
    When caching is enabled, reported latencies may reflect cache hits (< 250ms) rather than actual model/tool execution time. Disable caching for accurate latency measurements.

---

## Reproducibility

### For Reproducible Results

```powershell
python scripts/evaluate_llms.py \
    --dataset data/benchmark.jsonl \
    --models gpt-5.1 \
    --temperature 0.0 \    # Deterministic
    --seed 42 \            # Fixed shuffle seed
    --disable-tool-cache \ # No caching
    --output results/reproducible.jsonl
```

### What's Recorded

Each result includes provenance:

- Model versions (agent, vision, structuring)
- Prompt hashes
- Run configuration
- Timing breakdown
- Tool outputs

---

## Best Practices

### For Accurate Benchmarks

1. **Use temperature 0.0** for deterministic results
2. **Disable caching** when measuring latency
3. **Run multiple trials** to estimate variance
4. **Use the same dataset** for all comparisons

### For Large Evaluations

1. **Start with `--limit`** to test setup
2. **Use `--num-workers`** carefully (watch rate limits)
3. **Enable caching** for repeat runs
4. **Save intermediate results** (default output files)

### For Valid Comparisons

1. **Same dataset** for all configurations
2. **Same evaluation date** (model versions may change)
3. **Document provenance** from metrics output
4. **Report uncertainty** from multi-trial runs

---

## Example Evaluation Script

```python
"""Complete evaluation workflow."""
import json
from pathlib import Path
import subprocess

# Configuration
DATASET = "data/benchmark_500.jsonl"
MODELS = ["gpt-5.1", "gpt-5-mini"]
OUTPUT_DIR = Path("results/2026-01-15")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Run evaluation
for model in MODELS:
    output_file = OUTPUT_DIR / f"{model.replace('/', '_')}.jsonl"
    metrics_file = OUTPUT_DIR / f"{model.replace('/', '_')}.metrics.json"
    
    cmd = [
        "python", "scripts/evaluate_llms.py",
        "--dataset", DATASET,
        "--models", model,
        "--tools", "both",
        "--temperature", "0.0",
        "--disable-tool-cache",
        "--output", str(output_file),
        "--metrics-output", str(metrics_file),
    ]
    
    print(f"Evaluating {model}...")
    subprocess.run(cmd, check=True)

# Summarize results
subprocess.run([
    "python", "scripts/summarize_results.py",
    "--results-dir", str(OUTPUT_DIR),
    "--out", str(OUTPUT_DIR / "summary.json"),
    "--out-md", str(OUTPUT_DIR / "report.md"),
], check=True)

print(f"Results saved to {OUTPUT_DIR}")
```

---

## Next Steps

- [Metrics Reference](../evaluation/metrics.md) — Detailed metric definitions
- [Interpreting Results](interpreting-results.md) — Understand what results mean
- [Reproducibility](../research/reproducibility.md) — Research-grade evaluation
