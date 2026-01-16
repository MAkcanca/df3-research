# Running evaluate_llms.py with BAML Support

## Quick Start

### Step 1: Install BAML (if not already installed)

```bash
pip install baml-py==0.214.0
```

### Step 2: Prepare Your Dataset

Create a JSONL file with this format:

```jsonl
{"id": "sample-1", "image": "path/to/image1.jpg", "label": "real"}
{"id": "sample-2", "image": "path/to/image2.jpg", "label": "fake"}
{"id": "sample-3", "image": "path/to/image3.jpg", "label": "real"}
```

### Step 3: Run Evaluation

```bash
# BAML is always enabled by default
python scripts/evaluate_llms.py \
    --dataset path/to/dataset.jsonl \
    --models gpt-5.1,gpt-5-mini \
    --tools both
```

## Interpreting Metrics (Important)

`evaluate_llms.py` supports a **triage-style** workflow where the model can abstain with `uncertain`.

- **`overall_acc`**: accuracy over *all* samples, where `uncertain` counts as not-correct.
- **`acc_answered`**: accuracy on the subset where the model answered `real` or `fake`.
- **`coverage`**: fraction of samples answered (`real`/`fake`) rather than abstained (`uncertain`).
- **`review_rate`**: abstain rate (how often it routes to manual review).

If `review_rate` is high, `overall_acc` will look artificially low even when `acc_answered` is strong.

## Decision Policy: triage vs forced-binary

By default, the agent runs in **`triage`** mode (may output `uncertain`).

For benchmarking like a standard classifier, use **forced-binary** mode:

```bash
python scripts/evaluate_llms.py \
    --dataset data/my_eval.jsonl \
    --models gpt-5-mini \
    --tools both \
    --decision-policy forced
```

- `--decision-policy triage`: allows `uncertain` (manual review) and prefers abstaining over guessing.
- `--decision-policy forced`: requires a binary verdict (`real`/`fake`). The agent may still explain uncertainty, but must choose a class.

## Example Commands

### Basic Evaluation

```bash
python scripts/evaluate_llms.py \
    --dataset data/test_set.jsonl \
    --models gpt-5.1 \
    --tools both \
    --output results.jsonl \
    --metrics-output metrics.json
```

### Compare Multiple Models with BAML

```bash
python scripts/evaluate_llms.py \
    --dataset data/test_set.jsonl \
    --models gpt-5.1,gpt-5-mini,claude-sonnet-4 \
    --tools both \
    --use-baml \
    --trials 3 \
    --num-workers 2
```

### Vision-Only Evaluation (No Tools)

```bash
python scripts/evaluate_llms.py \
    --dataset data/test_set.jsonl \
    --models gpt-5.1 \
    --tools no-tools
```

### With OpenRouter

```bash
export OPENROUTER_API_KEY="your-key"

python scripts/evaluate_llms.py \
    --dataset data/test_set.jsonl \
    --models anthropic/claude-sonnet-4 \
    --provider openrouter \
    --tools both
```

## Command Line Options

### Required
- `--dataset`: Path to JSONL dataset file
- `--models`: Comma-separated list of model names

### Optional
- `--tools`: `both`, `tools`, or `no-tools` (default: `both`)
- `--temperature`: LLM temperature (default: 0.0)
- `--max-iterations`: Max agent iterations (default: 30)
- `--trials`: Number of trials per configuration (default: 1)
- `--num-workers`: Parallel workers (default: 1)
- `--limit`: Max number of samples to evaluate
- `--shuffle`: Shuffle dataset before evaluation
- `--seed`: Random seed for reproducibility
- `--output`: Output file for per-sample results (default: `eval_results.jsonl`)
- `--metrics-output`: Output file for aggregated metrics (default: `eval_metrics.json`)

### Provider Options
- `--provider`: `openai` or `openrouter` (default: `openai`)
- `--api-key`: API key (or use environment variable)
- `--base-url`: Optional API base URL
- `--referer`: HTTP-Referer header for OpenRouter
- `--title`: X-Title header for OpenRouter

## Output Files

### Per-Sample Results (`eval_results.jsonl`)

Each line contains:
```json
{
  "id": "sample-1",
  "model": "gpt-5.1",
  "use_tools": true,
  "trial": 0,
  "label": "real",
  "prediction": "real",
  "confidence": 0.95,
  "rationale": "...",
  "latency_seconds": 2.3,
  "tool_usage": ["analyze_jpeg_compression", "extract_residuals"]
}
```

### Aggregated Metrics (`eval_metrics.json`)

```json
{
  "gpt-5.1|tools": {
    "per_trial": {
      "0": {
        "accuracy": 0.85,
        "balanced_accuracy": 0.83,
        "f1_fake": 0.82,
        "f1_real": 0.84
      }
    },
    "summary": {
      "mean": {
        "accuracy": 0.85,
        "balanced_accuracy": 0.83
      },
      "std": {
        "accuracy": 0.02
      }
    }
  }
}
```

## What BAML Mode Does

When `--use-baml` is enabled:

1. **Vision-only step**: Uses BAML's unstructured reasoning (preserves reasoning quality)
2. **Agent step**: Still uses LangChain for tool calling (unchanged)
3. **Structuring step**: Uses BAML to extract structured data from reasoning

This multi-step approach avoids reasoning degradation that happens when LLMs are constrained to strict output formats.

## Troubleshooting

### "BAML client not available"
- Install BAML: `pip install baml-py==0.214.0`
- BAML is required for the evaluation script to run

### API Key Issues
- Set `OPENAI_API_KEY` or `OPENROUTER_API_KEY` environment variable
- Or use `--api-key` flag

### Dataset Format Errors
- Ensure JSONL file has `id`, `image`, and `label` fields
- Image paths can be relative (to dataset file directory) or absolute

### Memory Issues with Multiple Workers
- Reduce `--num-workers` if you run out of memory
- Each worker creates its own agent instance

