# Analyzing Images

Complete guide to analyzing images with DF3, including all options and advanced usage patterns.

---

## Command-Line Interface

### Basic Usage

```powershell
python scripts/analyze_image.py --image path/to/image.jpg
```

### All Options

```powershell
python scripts/analyze_image.py [OPTIONS]

Required:
  --image PATH          Path to the image file to analyze

Optional:
  --model MODEL         LLM model to use (default: gpt-5.1)
  --vision-model MODEL  Model for vision step (defaults to --model)
  --structuring-model MODEL
                        Model for structuring step (defaults to --model)
  --temperature FLOAT   LLM temperature (default: 0.2)
  --provider NAME       Provider for LLM calls: "openai" or "openrouter" (default: openai)
  --api-key KEY         API key (defaults to provider env var)
  --base-url URL        Optional API base URL (e.g., https://openrouter.ai/api/v1)
  --referer TEXT        Optional HTTP-Referer header (OpenRouter)
  --title TEXT          Optional X-Title header (OpenRouter)
  --no-tools            Run vision-only analysis without forensic tools
  --query TEXT          Specific question about the image
```

### Examples

```powershell
# Standard analysis with tools
python scripts/analyze_image.py --image suspicious.jpg

# Vision-only (faster)
python scripts/analyze_image.py --image photo.jpg --no-tools

# Specific question
python scripts/analyze_image.py --image portrait.jpg \
    --query "Does this face show signs of deepfake manipulation?"

# Use different model
python scripts/analyze_image.py --image photo.jpg --model gpt-5-mini

# Use OpenRouter (multi-provider) from the CLI
python scripts/analyze_image.py --image photo.jpg \
    --provider openrouter \
    --model google/gemini-3-flash-preview

# Use different models for each step
python scripts/analyze_image.py --image photo.jpg \
    --model gpt-5.1 \
    --vision-model gpt-5.2 \
    --structuring-model gpt-5-mini
```

---

## Programmatic API

### Basic Usage

```python
from src.agents import ForensicAgent

# Initialize agent
agent = ForensicAgent(llm_model="gpt-5.1")

# Analyze an image
result = agent.analyze("path/to/image.jpg")

# Access results
print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']}")
print(f"Rationale: {result['rationale']}")
```

### Full Configuration

```python
from src.agents import ForensicAgent

agent = ForensicAgent(
    llm_model="gpt-5.1",           # Model for agent/tool-calling
    vision_model="gpt-5.2",         # Model for vision analysis
    structuring_model="gpt-5-mini", # Model for output structuring
    temperature=0.0,                # Deterministic output
    api_key="sk-...",               # Optional: override env var
    base_url=None,                  # Optional: custom API endpoint
    max_iterations=15,              # Max tool calls per analysis
    enable_checkpointer=True,       # Enable LangGraph checkpointing
)

# Full analysis with tools
result = agent.analyze(
    image_path="photo.jpg",
    user_query="Is this AI-generated?",
    use_tools=True,
    pass_image_to_agent=False,  # Only pass text description to agent
)
```

### Result Structure

```python
result = {
    # Core verdict
    "verdict": "fake",          # "real" | "fake" | "uncertain"
    "confidence": 0.87,         # 0.0 - 1.0
    "rationale": "Multiple anatomical anomalies...",
    
    # Descriptions
    "visual_description": "Portrait of a woman...",
    "forensic_summary": "TruFor: 0.12, ELA: 1.4...",
    
    # Tool information
    "tool_usage": ["perform_trufor", "perform_ela"],
    "tool_details": [
        {"tool": "perform_trufor", "seconds": 2.3, "status": "completed"},
        {"tool": "perform_ela", "seconds": 0.8, "status": "completed"},
    ],
    "tool_results": [
        {"tool": "perform_trufor", "parsed": {"manipulation_probability": 0.12}},
        {"tool": "perform_ela", "parsed": {"ela_anomaly_score": 1.4}},
    ],
    
    # Timing
    "timings": {
        "vision_llm_seconds": 3.2,
        "agent_graph_seconds": 5.1,
        "total_seconds": 8.7,
    },
    
    # Provenance
    "models": {
        "agent": "gpt-5.1",
        "vision": "gpt-5.2",
        "structuring": "gpt-5-mini",
    },
    
    # Raw outputs
    "raw_text": "### Visual Analysis\n...",
    "raw_parsed": {...},
    "prompts": {
        "vision_system": "...",
        "vision_user": "...",
        "agent_system": "...",
        "agent_user": "...",
    },
    
    # SWGDE-style report
    "report_markdown": "## Image Authentication Report\n...",
}
```

---

## Analysis Modes

### Vision-Only Mode

Best for fast screening where tool latency is unacceptable.

```python
result = agent.analyze(image_path, use_tools=False)
```

**Characteristics:**

- Faster (3-7 seconds vs 8-30 seconds)
- Relies entirely on visual analysis
- Best for obvious AI-generated images
- May miss subtle manipulations

### Tool-Augmented Mode

Best for high-stakes decisions requiring technical evidence.

```python
result = agent.analyze(image_path, use_tools=True)
```

**Characteristics:**

- More thorough analysis
- Technical evidence supports verdict
- Better for manipulation detection
- Higher latency

---

## Working with Different Providers

### OpenAI (Default)

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

agent = ForensicAgent(llm_model="gpt-5.1")
```

### OpenRouter

Access 100+ models through a single API:

```python
agent = ForensicAgent(
    llm_model="google/gemini-3-flash-preview",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
```

### Anthropic via OpenRouter

```python
agent = ForensicAgent(
    llm_model="anthropic/claude-sonnet-4",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
```

---

## Handling Different Image Types

### JPEG Images

Full tool support including compression analysis:

```python
# All tools available
result = agent.analyze("photo.jpg")
# ELA, JPEG quantization, compression analysis all work
```

### PNG/WebP Images

JPEG-specific tools automatically skip:

```python
result = agent.analyze("screenshot.png")
# ELA reports: "skipped - JPEG-specific tool"
# TruFor, frequency analysis, residuals still work
```

### Large Images

Large images may be slow or exceed memory:

```python
from PIL import Image

# Resize before analysis if needed
img = Image.open("huge_photo.jpg")
if max(img.size) > 4096:
    img.thumbnail((4096, 4096), Image.LANCZOS)
    img.save("resized.jpg", quality=95)
    result = agent.analyze("resized.jpg")
```

---

## Error Handling

### Common Errors

```python
from src.agents import ForensicAgent

agent = ForensicAgent()

try:
    result = agent.analyze("nonexistent.jpg")
except FileNotFoundError as e:
    print(f"Image not found: {e}")

try:
    result = agent.analyze("corrupted.jpg")
except Exception as e:
    print(f"Analysis failed: {e}")
```

### Handling Uncertain Results

```python
result = agent.analyze("ambiguous_image.jpg")

if result["verdict"] == "uncertain":
    print("Image requires human review")
    print(f"Reason: {result['rationale']}")
    
    # Queue for manual review
    queue_for_review(
        image_path="ambiguous_image.jpg",
        confidence=result["confidence"],
        evidence=result["tool_results"],
    )
```

---

## Batch Processing

For analyzing multiple images, see [Batch Evaluation](batch-evaluation.md).

Quick example:

```python
from pathlib import Path
from src.agents import ForensicAgent

agent = ForensicAgent()
results = []

for image_path in Path("images/").glob("*.jpg"):
    result = agent.analyze(str(image_path))
    results.append({
        "image": str(image_path),
        "verdict": result["verdict"],
        "confidence": result["confidence"],
    })

# Process results
fake_count = sum(1 for r in results if r["verdict"] == "fake")
print(f"Found {fake_count} fake images out of {len(results)}")
```

---

## Best Practices

### For Accurate Results

1. **Use original images** when possible — avoid screenshots or re-saved copies
2. **Prefer JPEG** for manipulation detection — compression tools are JPEG-specific
3. **Use tools for high-stakes decisions** — vision-only may miss subtle manipulations
4. **Review uncertain results** — the system is appropriately calibrated to say "I don't know"

### For Performance

1. **Use vision-only mode** for fast screening
2. **Enable caching** for repeated analyses of similar images
3. **Use smaller models** (gpt-5-mini) for lower latency
4. **Batch similar images** to amortize model loading time

### For Reliability

1. **Set temperature to 0** for deterministic results
2. **Log full results** including tool outputs for audit trails
3. **Preserve raw images** — don't delete originals after analysis
4. **Document uncertain cases** with full reasoning

---

## Next Steps

- [Batch Evaluation](batch-evaluation.md) — Process many images with metrics
- [Interpreting Results](interpreting-results.md) — Understand what results mean
- [Tool Reference](../tools/overview.md) — Deep dive into forensic tools
