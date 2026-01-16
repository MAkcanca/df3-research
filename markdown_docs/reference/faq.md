# FAQ

---

## General

### What is DF3?

DF3 is an agentic forensic image analysis system that combines vision-capable LLMs with specialized forensic tools to classify images as `real`, `fake`, or `uncertain`.

### What does the verdict mean?

| Verdict | Meaning |
|---------|---------|
| `real` | No manipulation or generation indicators detected |
| `fake` | Manipulation or AI-generation indicators detected |
| `uncertain` | Evidence insufficient or conflicting |

### What types of fakes does DF3 target?

- AI-generated images (DALL-E, Midjourney, Stable Diffusion, etc.)
- Manipulated images (splicing, inpainting, copy-move)
- Face swaps and identity manipulations

---

## Technical

### Which LLMs are supported?

DF3 uses [LangChain](https://www.langchain.com/) and [BAML](https://boundaryml.ai/) libraries for model API integration, standardized on either OpenRouter-style or OpenAI-style API calls. **Any vision-capable model that supports tool calling (at least in prompt format) is supported**, including:

- Models accessible via OpenAI-compatible endpoints (cloud or local)
- Models accessible via OpenRouter
- Locally run LLMs (via OpenAI-compatible local servers or LangChain's local integrations)
- Any provider supported by LangChain's model integrations

The system does not require specific model implementations—it works with any model that can process images and handle tool/function calling semantics, whether hosted remotely or running locally.

### What forensic tools are included?

| Tool | Function |
|------|----------|
| TruFor | Neural forgery detection |
| ELA | JPEG compression anomaly detection |
| JPEG Analysis | Quantization table analysis |
| Frequency Analysis | DCT/FFT pattern detection |
| Residual Extraction | DRUNet noise analysis |
| Metadata | EXIF/XMP/C2PA extraction |
| Code Execution | Custom Python analysis |

### Is GPU required?

No. TruFor and DRUNet run on CPU but are faster with CUDA.

### What image formats are supported?

JPEG, PNG, BMP, TIFF, WEBP

---

## Usage

### Single image analysis

```powershell
python scripts/analyze_image.py --image photo.jpg
```

### Vision-only mode (no tools)

```powershell
python scripts/analyze_image.py --image photo.jpg --no-tools
```

### Batch evaluation

```powershell
python scripts/evaluate_llms.py --dataset data/eval.jsonl --models gpt-5.1 --tools both
```

### Different vision and agent models

```powershell
python scripts/analyze_image.py --image photo.jpg \
    --model gpt-5.1 \
    --vision-model gemini-3-flash \
    --structuring-model gpt-5-mini
```

---

## Results

### What does the confidence score mean?

The LLM's self-reported certainty (0-1). Use for triage ranking, not as a calibrated probability.

### Why does the model output UNCERTAIN?

- Conflicting evidence between visual analysis and tools
- Low internal confidence
- Image quality insufficient for analysis
- Ambiguous content

### TruFor says "no manipulation" but image looks AI-generated?

Expected. TruFor detects **manipulation** (editing), not **generation**. AI-generated images are internally consistent and score low on manipulation detectors.

---

## Integration

### Library usage

```python
from src.agents.forensic_agent import ForensicAgent

agent = ForensicAgent(llm_model="gpt-5.1")
result = agent.analyze("photo.jpg")
print(result["verdict"])
```

### Adding custom tools

Implement in `src/tools/forensic/` and register in `forensic_tools.py`.

---

## Data

### Are images sent externally?

Images are sent to the configured LLM provider (OpenAI, Anthropic, Google, etc.). Forensic tools run locally.

### Local caching

Tool outputs and vision results can be cached in `.tool_cache/`. Disable with `--disable-tool-cache`.

---

## See Also

- [Quickstart](../getting-started/quickstart.md)
- [Troubleshooting](troubleshooting.md)
- [Limitations](../research/limitations.md)
