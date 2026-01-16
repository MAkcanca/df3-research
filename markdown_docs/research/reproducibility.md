# Reproducibility Guide

How to ensure your DF3 experiments can be reproduced.

---

## Why Reproducibility Matters

Forensic conclusions must be:

- **Verifiable** — Others can check your work
- **Consistent** — Same inputs produce same outputs
- **Documented** — Process is fully recorded

---

## Sources of Non-Reproducibility

### LLM Variability

| Factor | Impact | Mitigation |
|--------|--------|------------|
| Temperature > 0 | Random sampling | Use `temperature: 0.0` |
| Model updates | Behavior changes | Record model version |
| Prompt changes | Different reasoning | Hash prompts |
| API variability | Non-deterministic | Multiple trials |

### Data Variability

| Factor | Impact | Mitigation |
|--------|--------|------------|
| Dataset changes | Different samples | Record dataset digest |
| Sample ordering | Processing order effects | Use seed for shuffling |
| Missing samples | Incomplete comparison | Verify sample counts |

### Environment Variability

| Factor | Impact | Mitigation |
|--------|--------|------------|
| Package versions | Behavior differences | Pin versions |
| Hardware differences | Floating point variance | Document hardware |
| Cache state | Different code paths | Document cache usage |

---

## Reproducibility Checklist

### 1. Record Configuration

```json
{
    "model": "gpt-5.1",
    "vision_model": "gpt-5.1",
    "structuring_model": "gpt-5.1",
    "temperature": 0.0,
    "max_iterations": 15,
    "cache_enabled": true,
    "timestamp": "2026-01-15T10:30:00Z"
}
```

### 2. Record Dataset

```json
{
    "dataset_path": "data/benchmark.jsonl",
    "sample_count": 500,
    "real_count": 250,
    "fake_count": 250,
    "id_digest": "f987165daff0de70",
    "shuffle_seed": 42
}
```

### 3. Record Environment

```powershell
# Generate requirements snapshot
pip freeze > requirements-frozen.txt

# Record Python version
python --version > environment.txt
```

### 4. Record Results

```json
{
    "run_id": "eval-2026-01-15-001",
    "git_commit": "abc123...",
    "prompt_hash": "def456...",
    "results_file": "results/eval.jsonl",
    "metrics_file": "results/metrics.json"
}
```

---

## Recommended Workflow

### Before Running

```powershell
# 1. Commit or stash changes
git status

# 2. Record commit hash
git rev-parse HEAD

# 3. Verify dataset
python scripts/verify_dataset.py --dataset data/benchmark.jsonl

# 4. Clear cache (optional, for latency measurement)
Remove-Item -Recurse -Force .tool_cache
```

### During Run

```powershell
# Run with full configuration
python scripts/evaluate_llms.py `
    --dataset data/benchmark.jsonl `
    --models gpt-5.1 `
    --tools both `
    --temperature 0.0 `
    --trials 3 `
    --seed 42 `
    --output results/eval_$(Get-Date -Format "yyyyMMdd").jsonl `
    --metrics-output results/metrics_$(Get-Date -Format "yyyyMMdd").json
```

### After Running

```powershell
# 1. Archive results with metadata
python scripts/archive_results.py --run-dir results/

# 2. Verify metrics
python scripts/summarize_results.py --results-dir results/

# 3. Commit results (if using git for tracking)
git add results/
git commit -m "Evaluation run 2026-01-15"
```

---

## Multi-Trial Evaluation

For robust results with variance estimates:

```powershell
python scripts/evaluate_llms.py `
    --dataset data/benchmark.jsonl `
    --models gpt-5.1 `
    --trials 5 `
    --temperature 0.0 `
    --seed 42
```

This produces:

- Individual results for each trial
- Summary statistics (mean, std) across trials
- Per-trial metrics breakdown

---

## Comparing Configurations

### Valid Comparison

```powershell
# Same dataset, same seed, different models
python scripts/evaluate_llms.py `
    --dataset data/benchmark.jsonl `
    --models gpt-5.1,gpt-5-mini `
    --seed 42 `
    --temperature 0.0
```

### Invalid Comparison

```powershell
# Different datasets - NOT comparable
python scripts/evaluate_llms.py --dataset datasetA.jsonl --models gpt-5.1
python scripts/evaluate_llms.py --dataset datasetB.jsonl --models gpt-5-mini
```

### Paired Analysis

Use `summarize_results.py` for statistical tests:

```powershell
python scripts/summarize_results.py `
    --results-dir results/ `
    --paired-comparison "gpt-5.1 vs gpt-5-mini"
```

---

## Dataset Digest

Compute dataset identifier for comparison:

```python
import hashlib
import json

with open("data/benchmark.jsonl") as f:
    samples = [json.loads(line) for line in f]

ids = sorted([s["id"] for s in samples])
digest = hashlib.sha256(str(ids).encode()).hexdigest()[:16]
print(f"Dataset digest: {digest}")
```

Compare digests to ensure same dataset:

| Run | Dataset Digest | Comparable? |
|-----|---------------|-------------|
| A | f987165daff0de70 | — |
| B | f987165daff0de70 | ✅ Yes |
| C | 1f78e35118013ed4 | ❌ No |

---

## Prompt Versioning

### Hash Prompts

```python
import hashlib
from src.agents.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

prompt_content = SYSTEM_PROMPT + USER_PROMPT_TEMPLATE
prompt_hash = hashlib.sha256(prompt_content.encode()).hexdigest()[:16]
```

### Record in Results

```json
{
    "prompt_version": {
        "system_prompt_hash": "abc123...",
        "user_prompt_hash": "def456...",
        "baml_version": "v1.2.3"
    }
}
```

---

## Known Non-Determinism

Even with best practices, some variability exists:

| Source | Typical Impact | Notes |
|--------|---------------|-------|
| LLM API | ~1-2% accuracy variance | Temperature 0 helps |
| Floating point | Minimal | Hardware dependent |
| Tool execution order | Minimal | Usually consistent |
| Network latency | Latency only | Not accuracy |

### Accounting for Variability

- Run multiple trials
- Report confidence intervals
- Use Wilson score for proportions
- Acknowledge in limitations

---

## Artifact Storage

### Directory Structure

```
results/
├── eval_20260115/
│   ├── raw_results.jsonl
│   ├── metrics.json
│   ├── config.json
│   ├── environment.txt
│   └── README.md
```

### Metadata File

```json
{
    "run_id": "eval_20260115",
    "timestamp": "2026-01-15T10:30:00Z",
    "git_commit": "abc123def456",
    "dataset_digest": "f987165daff0de70",
    "models": ["gpt-5.1"],
    "notes": "Baseline evaluation with tools enabled"
}
```

---

## External Reproduction

For others to reproduce your results:

### Provide

1. **Code version** — Git commit or release tag
2. **Dataset manifest** — IDs, sources, hashes (if data can't be shared)
3. **Configuration** — All parameters used
4. **Environment** — Python version, package versions
5. **Results** — Raw outputs for verification

### Document

1. **Steps to run** — Exact commands
2. **Expected outputs** — Sample results
3. **Known issues** — Any caveats

---

## See Also

- [Methodology](../evaluation/methodology.md) — Evaluation framework
- [Dataset Provenance](../evaluation/dataset-provenance.md) — Dataset documentation
- [Limitations](limitations.md) — Known limitations
