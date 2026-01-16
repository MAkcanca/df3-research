# Configuration Guide

Complete reference for configuring DF3 behavior.

---

## Configuration Methods

DF3 accepts configuration through:

1. **Command-line arguments** — Highest priority
2. **Environment variables** — See [Environment Variables](environment.md)
3. **Code defaults** — Fallback values

---

## Agent Configuration

### Model Selection

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Agent Model | `--model` | `gpt-5.1` | Primary model for reasoning and tools |
| Vision Model | `--vision-model` | Same as agent | Model for initial vision analysis |
| Structuring Model | `--structuring-model` | Same as agent | Model for BAML structuring |

**Example: Split models for cost optimization**

```powershell
python scripts/analyze_image.py ^
    --image photo.jpg ^
    --model gpt-5.1 ^
    --vision-model gemini-3-flash ^
    --structuring-model gpt-5-mini
```

### Temperature

| Parameter | CLI Flag | Default | Range |
|-----------|----------|---------|-------|
| Temperature | `--temperature` | `0.2` | 0.0 - 2.0 |

- `0.0` — Most deterministic, recommended for evaluation
- `0.2` — Default, slight variation
- Higher — More creative but less consistent

### Agent Behavior

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Max Iterations | `--max-iterations` | `15` | Maximum tool-calling iterations |
| Tools Enabled | `--no-tools` (flag) | Tools enabled | Disable tool-calling |
| Reasoning Effort | `--reasoning-effort` | None | Provider-specific reasoning control |

---

## Caching Configuration

### Cache Control

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Disable Cache | `--disable-tool-cache` | Enabled | Turn off all caching |
| Cache Directory | `--tool-cache-dir` | `.tool_cache` | Custom cache location |

### Cache Behavior

When enabled, DF3 caches:

- **Tool outputs** — Same image + tool = cached result
- **Vision model outputs** — Same image + model = cached description

### Clearing Cache

```powershell
# Remove entire cache
Remove-Item -Recurse -Force .tool_cache

# Remove specific tool cache
Remove-Item -Recurse -Force .tool_cache\perform_trufor
```

!!! warning "Cache and Reproducibility"
    For valid latency measurements, disable caching:
    ```powershell
    --disable-tool-cache
    ```

---

## Evaluation Configuration

### Dataset Options

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Dataset | `--dataset` | Required | Path to JSONL dataset |
| Limit | `--limit` | All | Process only first N samples |
| Shuffle | `--shuffle` | No | Randomize sample order |
| Seed | `--seed` | None | Random seed for shuffling |

### Output Options

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Output | `--output` | `eval_results.jsonl` | Per-sample results |
| Metrics Output | `--metrics-output` | `eval_metrics.json` | Aggregated metrics |

### Parallel Processing

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Workers | `--num-workers` | `1` | Parallel worker threads |
| Trials | `--trials` | `1` | Repeat runs for variance |

**Example: Fast evaluation**

```powershell
python scripts/evaluate_llms.py ^
    --dataset data/benchmark.jsonl ^
    --models gpt-5.1 ^
    --num-workers 4 ^
    --tools both
```

---

## Tool Configuration

### TruFor Settings

TruFor automatically:

- Downloads weights on first use (to `weights/trufor/`)
- Selects GPU if available, falls back to CPU
- Caches results by image hash

### DRUNet (Residuals) Settings

DRUNet weights stored in `src/tools/forensic/drunet/weights/`.

### Custom Tool Directory

To add custom tools, extend `src/tools/forensic_tools.py`.

---

## BAML Configuration

### Client Configuration

BAML clients are defined in `baml_src/clients.baml`:

```baml
client<llm> DynamicForensicClient {
    provider "openai"
    options {
        model "gpt-5.1"
        temperature 0.2
    }
}
```

### Runtime Override

The Python code in `src/agents/baml_forensic.py` overrides the client at runtime:

```python
cr = baml_py_core.ClientRegistry()
cr.add_llm_client(
    name="DynamicForensicClient",
    provider=provider,
    options={"model": model_name, ...}
)
```

### Regenerating BAML Client

After modifying `.baml` files:

```powershell
baml-cli generate
```

---

## Provider Configuration

### OpenAI

```ini
# .env
OPENAI_API_KEY=sk-...
```

### OpenRouter

```ini
# .env
OPENROUTER_API_KEY=sk-or-v1-...
```

Access multiple providers through a single API.

### Anthropic

```ini
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

### Google (Gemini)

```ini
# .env
GOOGLE_API_KEY=...
```

---

## Logging Configuration

### Verbosity

| Level | Description |
|-------|-------------|
| Default | Standard output only |
| Verbose | Include tool call details |
| Debug | Full LLM prompts/responses |

Controlled via logging module:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Resource Limits

### Memory

- TruFor model: ~2GB GPU memory (or RAM on CPU)
- DRUNet model: ~500MB
- Image processing: Varies by image size

### Rate Limits

Consider provider rate limits when:

- Running batch evaluation
- Using multiple workers
- Processing large datasets

---

## Configuration Examples

### Minimal Analysis

```powershell
python scripts/analyze_image.py --image photo.jpg
```

### Production Evaluation

```powershell
python scripts/evaluate_llms.py ^
    --dataset data/benchmark.jsonl ^
    --models gpt-5.1 ^
    --tools both ^
    --temperature 0.0 ^
    --trials 3 ^
    --num-workers 4 ^
    --disable-tool-cache ^
    --output results/eval.jsonl ^
    --metrics-output results/metrics.json
```

### Cost-Optimized Analysis

```powershell
python scripts/analyze_image.py ^
    --image photo.jpg ^
    --model gpt-5-mini ^
    --vision-model gpt-5.1 ^
    --structuring-model gpt-5-mini
```

### Quick Validation

```powershell
python scripts/evaluate_llms.py ^
    --dataset data/benchmark.jsonl ^
    --models gpt-5-mini ^
    --limit 50 ^
    --tools no-tools
```

---

## See Also

- [Environment Variables](environment.md) — API keys and secrets
- [Troubleshooting](troubleshooting.md) — Common issues
- [CLI Reference](../guide/analyzing-images.md) — Command-line usage
