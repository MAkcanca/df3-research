# ForensicAgent API Reference

Complete API reference for the `ForensicAgent` class.

---

## Class Definition

```python
from src.agents import ForensicAgent

class ForensicAgent:
    """
    Forensic image analysis agent that combines vision-capable LLMs
    with forensic tools to detect AI-generated and manipulated images.
    """
```

---

## Constructor

```python
def __init__(
    self,
    llm_model: str = "gpt-5.1",
    vision_model: Optional[str] = None,
    structuring_model: Optional[str] = None,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
    decision_policy: Optional[str] = None,
    max_iterations: Optional[int] = 15,
    enable_checkpointer: bool = True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_model` | str | `"gpt-5.1"` | Model for agent reasoning and tool calling |
| `vision_model` | str \| None | None | Model for vision analysis (defaults to `llm_model`) |
| `structuring_model` | str \| None | None | Model for BAML structuring (defaults to `llm_model`) |
| `temperature` | float | `0.0` | LLM temperature (0.0 for deterministic) |
| `reasoning_effort` | str \| None | None | Reasoning effort level (provider-specific) |
| `api_key` | str \| None | None | API key (defaults to `OPENAI_API_KEY` env var) |
| `base_url` | str \| None | None | Custom API base URL (e.g., OpenRouter) |
| `default_headers` | Dict \| None | None | Custom HTTP headers (e.g., OpenRouter headers) |
| `decision_policy` | str \| None | None | Reserved for future use |
| `max_iterations` | int \| None | `15` | Maximum tool calls per analysis |
| `enable_checkpointer` | bool | `True` | Enable LangGraph state checkpointing |

### Example

```python
# Basic initialization
agent = ForensicAgent()

# Custom configuration
agent = ForensicAgent(
    llm_model="gpt-5.2",
    vision_model="gpt-5.2",
    structuring_model="gpt-5-mini",
    temperature=0.0,
    max_iterations=20,
)

# OpenRouter configuration
agent = ForensicAgent(
    llm_model="anthropic/claude-sonnet-4",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
```

---

## Methods

### analyze()

Main analysis method for processing images.

```python
def analyze(
    self,
    image_path: str,
    user_query: Optional[str] = None,
    use_tools: bool = True,
    pass_image_to_agent: bool = False,
) -> Dict[str, Any]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | str | required | Path to the image file |
| `user_query` | str \| None | None | Optional specific question about the image |
| `use_tools` | bool | `True` | Whether to use forensic tools |
| `pass_image_to_agent` | bool | `False` | Include image in agent message (vs text only) |

#### Returns

Dictionary containing analysis results:

```python
{
    # Core verdict
    "verdict": str,           # "real" | "fake" | "uncertain"
    "confidence": float,      # 0.0 - 1.0
    "rationale": str,         # Explanation for verdict
    
    # Descriptions
    "visual_description": str,
    "forensic_summary": str,
    
    # Tool information
    "tool_usage": List[str],  # Tool names used
    "tool_details": List[Dict],
    "tool_results": List[Dict],
    
    # Timing
    "timings": {
        "vision_llm_seconds": float,
        "agent_graph_seconds": float,
        "total_seconds": float,
    },
    
    # Model provenance
    "models": {
        "agent": str,
        "vision": str,
        "structuring": str,
    },
    
    # Raw outputs
    "raw_text": str,
    "raw_parsed": Dict,
    "prompts": Dict[str, str],
    
    # Report
    "report_markdown": str,   # SWGDE-style report
    
    # Metadata
    "image_path": str,
}
```

#### Example

```python
agent = ForensicAgent()

# Basic analysis with tools
result = agent.analyze("photo.jpg")
print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']:.2f}")

# Vision-only analysis
result = agent.analyze("photo.jpg", use_tools=False)

# With specific question
result = agent.analyze(
    "portrait.jpg",
    user_query="Is this a deepfake?",
    use_tools=True,
)
```

---

## Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `llm` | ChatOpenAI | LangChain LLM instance for agent |
| `llm_model` | str | Model identifier |
| `vision_model` | str | Vision step model identifier |
| `structuring_model` | str | Structuring step model identifier |
| `tools` | List[Tool] | Registered forensic tools |
| `agent_executor` | CompiledGraph | LangGraph agent graph |
| `system_prompt` | str | Agent system prompt |
| `max_iterations` | int \| None | Maximum iterations |

---

## Class Methods

### _normalize_verdict()

Normalize verdict string to canonical form.

```python
@staticmethod
def _normalize_verdict(verdict: Optional[str]) -> str:
    """
    Normalize verdict to 'real', 'fake', or 'uncertain'.
    
    Handles variations like 'AI-generated' -> 'fake',
    'authentic' -> 'real', 'inconclusive' -> 'uncertain'.
    """
```

---

## Usage Patterns

### Single Image Analysis

```python
from src.agents import ForensicAgent

agent = ForensicAgent()
result = agent.analyze("suspicious_image.jpg")

if result["verdict"] == "fake":
    print(f"Image appears fake: {result['rationale']}")
elif result["verdict"] == "uncertain":
    print("Image needs human review")
else:
    print("Image appears authentic")
```

### Batch Processing

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

# Analyze results
fake_count = sum(1 for r in results if r["verdict"] == "fake")
print(f"Found {fake_count} potentially fake images")
```

### Comparing Modes

```python
agent = ForensicAgent()

# Vision-only (faster)
vision_result = agent.analyze("image.jpg", use_tools=False)

# Tool-augmented (more thorough)
tools_result = agent.analyze("image.jpg", use_tools=True)

# Compare
print(f"Vision-only: {vision_result['verdict']} ({vision_result['confidence']:.2f})")
print(f"With tools: {tools_result['verdict']} ({tools_result['confidence']:.2f})")
print(f"Tools used: {tools_result['tool_usage']}")
```

### Error Handling

```python
from src.agents import ForensicAgent

agent = ForensicAgent()

try:
    result = agent.analyze("nonexistent.jpg")
except FileNotFoundError as e:
    print(f"Image not found: {e}")
except Exception as e:
    print(f"Analysis failed: {e}")
```

---

## Memory Management

### Image Cache

Encoded images are cached to avoid redundant base64 encoding:

```python
from src.agents.forensic_agent import clear_image_cache

# Clear the image encoding cache
clear_image_cache()
```

### Tool Cache

Tool outputs are cached separately. See [Configuration](../reference/configuration.md) for cache control.

---

## Thread Safety

`ForensicAgent` instances are **not thread-safe**. For parallel processing:

1. Create separate agent instances per thread
2. Use thread-local storage
3. Or use the batch evaluator which handles this

```python
import threading

thread_local = threading.local()

def get_agent():
    if not hasattr(thread_local, "agent"):
        thread_local.agent = ForensicAgent()
    return thread_local.agent
```

---

## See Also

- [BAML Functions](baml-functions.md) — Structured output functions
- [Tool Functions](tool-functions.md) — Forensic tool reference
- [Configuration](../reference/configuration.md) — Configuration options
