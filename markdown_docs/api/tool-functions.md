# Tool Functions API Reference

Reference documentation for forensic tool functions.

---

## Tool Registration

Tools are registered in `src/tools/forensic_tools.py`:

```python
from src.tools.forensic_tools import create_forensic_tools

tools = create_forensic_tools(timing_hook=None)
```

---

## TruFor

### perform_trufor()

```python
def perform_trufor(input_str: str) -> str:
    """
    Run TruFor forgery detection and localization.
    
    Args:
        input_str: JSON string or plain path
            - Plain: "/path/to/image.jpg"
            - JSON: {"path": "...", "return_map": false}
    
    Returns:
        JSON string with results
    """
```

#### Input Schema

```python
class TruForInput(BaseModel):
    path: str  # Path to image file
```

#### Output

```json
{
    "tool": "perform_trufor",
    "status": "completed",
    "image_path": "/path/to/image.jpg",
    "manipulation_probability": 0.15,
    "detection_score": 0.15,
    "localization_map": null,
    "localization_map_size": null,
    "note": "TruFor combines RGB features with Noiseprint++..."
}
```

### prewarm_trufor_model()

```python
def prewarm_trufor_model(device: str = None) -> bool:
    """
    Pre-warm TruFor model cache before workers start.
    
    Args:
        device: Optional device override (default: auto-detect)
    
    Returns:
        True if successful, False otherwise
    """
```

---

## ELA

### perform_ela()

```python
def perform_ela(input_str: str) -> str:
    """
    Run Error Level Analysis.
    
    Args:
        input_str: JSON string or plain path
            - JSON: {"path": "...", "quality": 75, "return_map": false}
    
    Returns:
        JSON string with results (or skipped for non-JPEG)
    """
```

#### Input Schema

```python
class ELAInput(BaseModel):
    path: str                    # Path to image file
    quality: int = 75            # JPEG recompression quality (1-100)
```

#### Output

```json
{
    "tool": "perform_ela",
    "status": "completed",
    "image_path": "/path/to/image.jpg",
    "quality": 75,
    "ela_mean": 12.5,
    "ela_std": 8.3,
    "ela_anomaly_score": 2.1,
    "ela_map": null,
    "ela_map_size": null,
    "note": "ELA recompresses at fixed JPEG quality..."
}
```

---

## Metadata

### metadata()

```python
def metadata(input_str: str) -> str:
    """
    Extract image metadata (EXIF/XMP/ICC) and C2PA credentials.
    
    Args:
        input_str: JSON string or plain path
    
    Returns:
        JSON string with metadata fields
    """
```

#### Output

```json
{
    "tool": "metadata",
    "status": "completed",
    "image_path": "/path/to/image.jpg",
    "exif": {...},
    "xmp": {...},
    "icc": {...},
    "c2pa": {...}
}
```

---

## JPEG Analysis

### analyze_jpeg_compression()

```python
def analyze_jpeg_compression(image_path: str) -> str:
    """
    Analyze JPEG compression artifacts.
    
    Args:
        image_path: Plain string path
    
    Returns:
        JSON string (or skipped for non-JPEG)
    """
```

### detect_jpeg_quantization()

```python
def detect_jpeg_quantization(image_path: str) -> str:
    """
    Extract quantization tables and estimate quality.
    
    Args:
        image_path: Plain string path
    
    Returns:
        JSON string (or skipped for non-JPEG)
    """
```

---

## Frequency Analysis

### analyze_frequency_domain()

```python
def analyze_frequency_domain(image_path: str) -> str:
    """
    Analyze DCT/FFT frequency domain features.
    
    Args:
        image_path: Plain string path
    
    Returns:
        JSON string with frequency features
    """
```

---

## Residual Analysis

### extract_residuals()

```python
def extract_residuals(image_path: str) -> str:
    """
    Extract DRUNet residual statistics.
    
    Args:
        image_path: Plain string path
    
    Returns:
        JSON string with residual statistics
    """
```

#### Output

```json
{
    "tool": "extract_residuals",
    "status": "completed",
    "image_path": "/path/to/image.jpg",
    "residual_mean": 0.0012,
    "residual_std": 8.45,
    "residual_skew": 0.23,
    "residual_kurtosis": 3.12,
    "residual_energy": 71.4,
    "residual_energy_mean": 0.0089,
    "residual_energy_std": 0.0034,
    "residual_energy_p95": 0.0156
}
```

### prewarm_residual_extractor()

```python
def prewarm_residual_extractor() -> bool:
    """
    Pre-warm DRUNet model before workers start.
    
    Returns:
        True if successful, False otherwise
    """
```

---

## Code Execution

### run_code_interpreter()

```python
def run_code_interpreter(input_str: str) -> str:
    """
    Execute Python code for custom analysis.
    
    Args:
        input_str: JSON string
            {"code": "...", "image_path": "..."}
    
    Returns:
        JSON string with execution output
    """
```

#### Input Schema

```python
class PythonCodeInput(BaseModel):
    code: str                     # Python code to execute
    image_path: Optional[str]     # Optional image path
```

#### Output

```json
{
    "tool": "execute_python_code",
    "status": "completed",
    "output": "stdout from code execution",
    "artifacts": ["path/to/saved/file.png"],
    "error": null
}
```

---

## Tool Caching

### ToolCache

```python
from src.tools.forensic.cache import ToolCache, get_cache, set_cache

# Get global cache instance
cache = get_cache()

# Set custom cache
custom_cache = ToolCache(
    cache_dir="/path/to/cache",
    enabled=True,
)
set_cache(custom_cache)

# Get cached output
output = cache.get(tool_name, image_path, params)

# Set cached output
cache.set(tool_name, image_path, output, params)

# Get cache statistics
stats = cache.get_stats()
```

### Cache Statistics

```python
{
    "cache_dir": "/path/to/cache",
    "entry_count": 150,
    "total_size_mb": 45.2,
    "enabled": True
}
```

---

## Timing Hook

Tools support timing instrumentation:

```python
def timing_hook(tool_name: str, seconds: float, error: Optional[str]) -> None:
    """Called after each tool execution."""
    print(f"{tool_name} took {seconds:.2f}s")

tools = create_forensic_tools(timing_hook=timing_hook)
```

---

## Error Handling

All tools return JSON with error information on failure:

```json
{
    "tool": "perform_trufor",
    "status": "error",
    "error": "Description of what went wrong"
}
```

### Status Values

| Status | Meaning |
|--------|---------|
| `completed` | Tool executed successfully |
| `error` | Tool encountered an error |
| `skipped` | Tool not applicable (e.g., JPEG tool on PNG) |

---

## Direct Tool Usage

Tools can be called directly without the agent:

```python
from src.tools.forensic import perform_trufor, perform_ela
import json

# TruFor
result = perform_trufor('{"path": "image.jpg"}')
data = json.loads(result)
print(f"Manipulation probability: {data['manipulation_probability']}")

# ELA
result = perform_ela('{"path": "image.jpg", "quality": 75}')
data = json.loads(result)
print(f"Anomaly score: {data['ela_anomaly_score']}")
```

---

## See Also

- [Tools Overview](../tools/overview.md) — High-level tool documentation
- [ForensicAgent](forensic-agent.md) — Agent API reference
- [Configuration](../reference/configuration.md) — Cache and device configuration
