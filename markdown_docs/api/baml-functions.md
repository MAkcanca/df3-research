# BAML Functions API Reference

Reference documentation for BAML-based structured output functions.

---

## Overview

DF3 uses BAML for reliable structured outputs. The key functions are:

| Function | Purpose | Output |
|----------|---------|--------|
| `AnalyzeImageVisionOnly` | Free-form vision reasoning | string |
| `AnalyzeImageVisionOnlyStructured` | Vision with structured output | ForensicAnalysisResult |
| `StructureForensicAnalysis` | Extract structure from text | ForensicAnalysisResult |

---

## Types

### Verdict

```python
class Verdict(Enum):
    REAL = "real"        # Image appears authentic
    FAKE = "fake"        # Image appears AI-generated or manipulated
    UNCERTAIN = "uncertain"  # Insufficient or conflicting evidence
```

### ForensicAnalysisResult

```python
@dataclass
class ForensicAnalysisResult:
    verdict: Verdict          # Classification result
    confidence: float         # 0.0 - 1.0
    rationale: str            # Brief justification (max 80 words)
    visual_description: str   # Description of image contents
    forensic_summary: str     # Summary of forensic analysis
    full_text: str            # Complete narrative
```

---

## Python Functions

### analyze_vision_only_structured_baml()

Combined vision analysis with structured output.

```python
async def analyze_vision_only_structured_baml(
    image_path: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | str | required | Path to image file |
| `model` | str \| None | `"gpt-5-mini"` | LLM model to use |
| `api_key` | str \| None | env var | API key |
| `base_url` | str \| None | None | Custom API endpoint |
| `default_headers` | Dict \| None | None | Custom HTTP headers |

#### Returns

```python
{
    "verdict": "fake",
    "confidence": 0.85,
    "rationale": "Multiple anatomical anomalies...",
    "visual_description": "Portrait of a woman...",
    "forensic_summary": "No tools used",
    "full_text": "### Visual Description\n..."
}
```

#### Example

```python
import asyncio
from src.agents.baml_forensic import analyze_vision_only_structured_baml

result = asyncio.run(
    analyze_vision_only_structured_baml(
        "photo.jpg",
        model="gpt-5.1",
    )
)
print(f"Verdict: {result['verdict']}")
```

---

### structure_analysis_baml()

Extract structured data from free-form reasoning text.

```python
async def structure_analysis_baml(
    reasoning_output: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reasoning_output` | str | required | Free-form analysis text |
| `model` | str \| None | `"gpt-5-mini"` | LLM model to use |
| `api_key` | str \| None | env var | API key |
| `base_url` | str \| None | None | Custom API endpoint |
| `default_headers` | Dict \| None | None | Custom HTTP headers |

#### Returns

Same structure as `analyze_vision_only_structured_baml()`.

#### Example

```python
import asyncio
from src.agents.baml_forensic import structure_analysis_baml

agent_output = """
### Visual Analysis
The image shows a person with anatomical anomalies...

### Conclusion
**Verdict: fake**
**Confidence (0-1): 0.85**
"""

result = asyncio.run(
    structure_analysis_baml(agent_output, model="gpt-5-mini")
)
```

---

### analyze_vision_only_baml()

Free-form vision reasoning without structured output.

```python
async def analyze_vision_only_baml(
    image_path: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]
```

#### Returns

```python
{
    "reasoning_output": "### Visual Description\n...",
    "image_path": "/path/to/image.jpg"
}
```

---

## BAML Function Definitions

### AnalyzeImageVisionOnlyStructured

```baml
function AnalyzeImageVisionOnlyStructured(image: image) -> ForensicAnalysisResult {
  client DynamicForensicClient
  prompt #"
    You are a forensic image analyst. Analyze this image and assess 
    whether it appears AI-generated, synthetic, or a deepfake.
    
    CRITICAL: You MUST always start your analysis by describing what 
    is actually in the image...
    
    {{ _.role("user") }} {{ image }}
    
    {{ ctx.output_format }}
  "#
}
```

### StructureForensicAnalysis

```baml
function StructureForensicAnalysis(reasoning_output: string) -> ForensicAnalysisResult {
  client DynamicForensicClient
  prompt #"
    Extract structured information from this forensic analysis 
    reasoning output...
    
    {{ reasoning_output }}
    
    {{ ctx.output_format }}
  "#
}
```

---

## Client Registry

The `DynamicForensicClient` is overridden at runtime:

```python
def _create_client_registry(
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> ClientRegistry:
    """Create BAML ClientRegistry with specified model."""
    cr = ClientRegistry()
    
    options = {
        "model": model,
        "api_key": api_key or os.environ.get("OPENAI_API_KEY"),
    }
    
    if base_url:
        options["base_url"] = base_url
    
    provider = "openai-responses"
    if base_url and "openrouter.ai" in base_url.lower():
        provider = "openai"
    
    cr.add_llm_client(
        name="DynamicForensicClient",
        provider=provider,
        options=options,
    )
    
    return cr
```

---

## Vision Caching

Vision outputs are cached to avoid redundant API calls:

```python
# Check cache
cached = cache.get_vision_output(
    vision_model=model,
    image_path=image_path,
    cache_tag=_vision_cache_tag(),
)

# Store in cache
cache.set_vision_output(
    vision_model=model,
    image_path=image_path,
    output=result,
    cache_tag=_vision_cache_tag(),
)
```

### Cache Tag

The cache tag ensures invalidation when prompts change:

```python
def _vision_cache_tag() -> str:
    # Check environment override
    env = os.getenv("DF3_VISION_CACHE_TAG")
    if env:
        return env
    
    # Hash BAML source file
    baml_path = "baml_src/forensic_analysis.baml"
    if baml_path.exists():
        return f"baml:{hashlib.sha256(baml_path.read_bytes()).hexdigest()[:16]}"
    
    return "unknown"
```

---

## Error Handling

```python
from baml_py import BamlValidationError

try:
    result = await analyze_vision_only_structured_baml("image.jpg")
except BamlValidationError as e:
    # Output didn't match schema
    print(f"Validation error: {e}")
except FileNotFoundError:
    # Image not found
    print("Image file not found")
except Exception as e:
    # Other errors
    print(f"Analysis failed: {e}")
```

---

## See Also

- [ForensicAgent](forensic-agent.md) — Main API reference
- [BAML Integration](../architecture/baml-integration.md) — Architecture details
- [Configuration](../reference/configuration.md) — Model configuration
