"""
Forensic tools for agent use.
"""

import json
import time
import functools
from typing import Any, Callable, List, Optional

from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from pathlib import Path

from .forensic import (
    analyze_frequency_domain,
    analyze_jpeg_compression,
    detect_jpeg_quantization,
    extract_residuals,
    metadata,
    perform_ela,
    perform_trufor,
    run_code_interpreter,
    # perform_cfa_detection,  # Disabled for now
)
from .forensic.cache import get_cache


# Pydantic schemas for structured tools
class TruForInput(BaseModel):
    """Input schema for TruFor tool."""
    path: str = Field(description="Path to the image file to analyze")


class ELAInput(BaseModel):
    """Input schema for ELA tool."""
    path: str = Field(description="Path to the image file to analyze")
    quality: int = Field(default=75, description="JPEG quality for recompression (1-100)")


# CFADetectionInput disabled for now
# class CFADetectionInput(BaseModel):
#     """Input schema for CFA detection tool."""
#     image_path: str = Field(description="Path to the image file to analyze")
#     mode: str = Field(default="analyze", description="Analysis mode (default: 'analyze').")
#     window: int = Field(default=256, description="Window size for analysis")
#     pattern: str = Field(default="RGGB", description="CFA pattern (RGGB, BGGR, GRBG, GBRG)")
#     channel: str = Field(default="G", description="Color channel to analyze (R, G, B)")


class PythonCodeInput(BaseModel):
    """Input schema for Python code execution tool."""
    code: str = Field(description="Python code to execute")
    image_path: Optional[str] = Field(default=None, description="Path to the image file (optional)")

TimingHook = Callable[[str, float, Optional[str]], None]


def _wrap_timed(tool_name: str, fn: Callable[..., str], timing_hook: Optional[TimingHook]) -> Callable[..., str]:
    """
    Wrap a tool callable to record wall-clock duration, without modifying tool outputs.

    timing_hook(tool_name, seconds, error_str_or_None)
    """
    if timing_hook is None:
        return fn

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> str:
        t0 = time.perf_counter()
        err: Optional[str] = None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err = str(e)
            raise
        finally:
            dt = time.perf_counter() - t0
            try:
                timing_hook(tool_name, float(dt), err)
            except Exception:
                # Never let instrumentation affect tool execution
                pass

    return wrapped


def _wrap_cached(
    tool_name: str,
    fn: Callable[..., str],
    extract_params: Optional[Callable[[Any, Any], tuple[str, Optional[dict]]]] = None,
) -> Callable[..., str]:
    """
    Wrap a tool callable with caching.
    
    Args:
        tool_name: Name of the tool for cache key generation
        fn: The tool function to wrap
        extract_params: Optional function to extract (image_path, params_dict) from args/kwargs.
                       If None, assumes first positional arg is image_path.
    
    Returns:
        Wrapped function that checks cache before execution and stores results.
    """
    cache = get_cache()
    
    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> str:
        # Extract image path and params for cache key
        if extract_params:
            image_path, params = extract_params(args, kwargs)
        else:
            # Default: first positional arg is image_path
            image_path = args[0] if args else None
            params = None
        
        # Try cache first
        if image_path:
            cached_output = cache.get(tool_name, image_path, params)
            if cached_output is not None:
                return cached_output
        
        # Cache miss - execute tool
        result = fn(*args, **kwargs)
        
        # Store in cache
        if image_path:
            cache.set(tool_name, image_path, result, params)
        
        return result
    
    return wrapped


def _extract_image_path_string(args: tuple, kwargs: dict) -> tuple[str, Optional[dict]]:
    """Extract image path from string-argument tools (first positional arg)."""
    image_path = args[0] if args else None
    return image_path, None


def _extract_image_path_json(args: tuple, kwargs: dict) -> tuple[str, Optional[dict]]:
    """Extract image path from JSON-string tools (parse JSON from first arg)."""
    if not args:
        return None, None
    try:
        input_data = json.loads(args[0]) if isinstance(args[0], str) else args[0]
        if isinstance(input_data, dict):
            image_path = input_data.get("path") or input_data.get("image_path")
            # Extract params (exclude path/image_path)
            params = {k: v for k, v in input_data.items() if k not in ("path", "image_path")}
            return image_path, params if params else None
    except Exception:
        pass
    # Fallback: treat as plain string
    return args[0] if args else None, None


def _trufor_structured(path: str) -> str:
    """Wrapper for perform_trufor that accepts structured arguments."""
    # GPU selection is handled internally - not exposed to LLM
    input_dict = {"path": path, "return_map": False}
    return perform_trufor(json.dumps(input_dict))


def _ela_structured(path: str, quality: int = 75) -> str:
    """Wrapper for perform_ela that accepts structured arguments."""
    # ELA is fundamentally a JPEG recompression heuristic. Running it on PNG/WEBP
    # often yields confusing outputs and wastes latency. Skip for non-JPEG inputs.
    try:
        ext = Path(path).suffix.lower()
        if ext not in (".jpg", ".jpeg"):
            return json.dumps(
                {
                    "tool": "perform_ela",
                    "status": "skipped",
                    "image_path": path,
                    "quality": int(quality),
                    "reason": f"ELA is JPEG-specific; skipping for {ext or 'unknown'} input.",
                }
            )
    except Exception:
        # Fall back to running ELA if path parsing fails
        pass
    input_dict = {"path": path, "quality": quality, "max_size": 0, "return_map": False}
    return perform_ela(json.dumps(input_dict))


def _jpeg_only_tool(tool_name: str, fn: Callable[[str], str]) -> Callable[[str], str]:
    """Skip JPEG-specific tools for non-JPEG inputs to reduce noise and wasted calls."""
    def wrapped(image_path: str) -> str:
        try:
            ext = Path(str(image_path)).suffix.lower()
            if ext not in (".jpg", ".jpeg"):
                return json.dumps(
                    {
                        "tool": tool_name,
                        "status": "skipped",
                        "image_path": image_path,
                        "reason": f"{tool_name} is JPEG-specific; skipping for {ext or 'unknown'} input.",
                    }
                )
        except Exception:
            pass
        return fn(image_path)
    return wrapped


def _extract_structured_params(args: tuple, kwargs: dict) -> tuple[str, Optional[dict]]:
    """Extract image path and params from structured tool args (path, quality, etc.)."""
    image_path = args[0] if args else kwargs.get("path") or kwargs.get("image_path")
    # Extract other params (like quality for ELA)
    params = {}
    if len(args) > 1:
        # For ELA: second arg is quality
        params["quality"] = args[1]
    params.update({k: v for k, v in kwargs.items() if k not in ("path", "image_path")})
    return image_path, params if params else None


def _extract_python_code_params(args: tuple, kwargs: dict) -> tuple[str, Optional[dict]]:
    """Extract image path from Python code tool (may be in code or image_path arg)."""
    image_path = kwargs.get("image_path")
    if not image_path and args:
        # Try to extract from code string
        code = args[0] if args else kwargs.get("code", "")
        if isinstance(code, str):
            import re
            match = re.search(r'image_path\s*=\s*["\']([^"\']+)["\']', code)
            if match:
                image_path = match.group(1)
    # Don't cache based on code content (too variable), only image_path
    params = {"has_code": bool(args or kwargs.get("code"))}
    return image_path, params if image_path else None


# CFA detection disabled for now
# def _cfa_detection_structured(
#     image_path: str,
#     mode: str = "analyze",
#     window: int = 256,
#     pattern: str = "RGGB",
#     channel: str = "G"
# ) -> str:
#     """Wrapper for perform_cfa_detection that accepts structured arguments."""
#     input_dict = {
#         "mode": mode,
#         "image_path": image_path,
#         "window": window,
#         "pattern": pattern,
#         "channel": channel
#     }
#     return perform_cfa_detection(json.dumps(input_dict))


def _python_code_structured(code: str, image_path: Optional[str] = None) -> str:
    """Wrapper for run_code_interpreter that accepts structured arguments."""
    input_dict = {"code": code}
    if image_path:
        input_dict["image_path"] = image_path
    return run_code_interpreter(json.dumps(input_dict))


def create_forensic_tools(timing_hook: Optional[TimingHook] = None) -> List:
    """
    Create LangChain tools for forensic operations.

    Returns:
        List of Tool instances
    """
    
    # Wrap tools with caching, then timing
    # Order: cache -> timing -> original function
    def wrap_with_cache_and_timing(tool_name: str, fn: Callable, extract_fn: Optional[Callable] = None):
        cached_fn = _wrap_cached(tool_name, fn, extract_fn)
        return _wrap_timed(tool_name, cached_fn, timing_hook)

    tools = [
        Tool(
            name="metadata",
            func=wrap_with_cache_and_timing(
                "metadata",
                metadata,
                _extract_image_path_json,
            ),
            description=(
                "Extract image metadata (EXIF/XMP/ICC) and detect C2PA / Content Credentials. "
                "Use this to check camera/device provenance, editing software hints, timestamps, GPS (if present), "
                "and whether Content Credentials (C2PA) exist. "
                "Input format: 'image_path' or JSON {\"path\": \"...\"}. Example: 'path/to/image.jpg'"
            ),
        ),
        Tool(
            name="analyze_jpeg_compression",
            func=wrap_with_cache_and_timing(
                "analyze_jpeg_compression",
                _jpeg_only_tool("analyze_jpeg_compression", analyze_jpeg_compression),
                _extract_image_path_string,
            ),
            description=(
                "Analyze JPEG compression artifacts and quantization tables. "
                "Use this to detect compression history and quality inconsistencies. "
                "Input format: 'image_path'. Example: 'path/to/image.jpg'"
            ),
        ),
        Tool(
            name="analyze_frequency_domain",
            func=wrap_with_cache_and_timing(
                "analyze_frequency_domain",
                analyze_frequency_domain,
                _extract_image_path_string,
            ),
            description=(
                "Analyze DCT/FFT frequency domain features. "
                "Use this to detect frequency domain anomalies that may indicate manipulation. "
                "Input format: 'image_path'. Example: 'path/to/image.jpg'"
            ),
        ),
        Tool(
            name="extract_residuals",
            func=wrap_with_cache_and_timing(
                "extract_residuals",
                extract_residuals,
                _extract_image_path_string,
            ),
            description=(
                "Extract denoiser residual statistics using DRUNet (deep learning denoiser). "
                "This tool applies a state-of-the-art neural network denoiser and analyzes the residual patterns. "
                "Returns comprehensive statistics: residual_mean, residual_std, residual_skew, residual_kurtosis, "
                "residual_energy, residual_energy_mean, residual_energy_std, residual_energy_p95. "
                "Use this to detect statistical anomalies in image residuals that may indicate manipulation, "
                "AI generation, or compression artifacts. Higher energy values or unusual distributions may indicate tampering. "
                "Input format: 'image_path'. Example: 'path/to/image.jpg'"
            ),
        ),
        Tool(
            name="detect_jpeg_quantization",
            func=wrap_with_cache_and_timing(
                "detect_jpeg_quantization",
                _jpeg_only_tool("detect_jpeg_quantization", detect_jpeg_quantization),
                _extract_image_path_string,
            ),
            description=(
                "Extract JPEG quantization tables, estimate quality, and flag double-compression periodicity. "
                "Use this for detailed JPEG forensics including quality estimation and double-compression detection. "
                "Input format: 'image_path'. Example: 'path/to/image.jpg'"
            ),
        ),
        StructuredTool.from_function(
            func=wrap_with_cache_and_timing(
                "perform_ela",
                _ela_structured,
                _extract_structured_params,
            ),
            name="perform_ela",
            description=(
                "Run Error Level Analysis (recompress at fixed JPEG quality, compute error map). "
                "Outputs ela_mean, ela_std, ela_anomaly_score (z-score, higher = more localized anomalies). "
                "Interpretation: ela_anomaly_score is relative - compare across regions or use judgment. "
                "Higher scores suggest localized edits; very low scores suggest uniform compression. "
                "Do NOT apply fixed thresholds - interpret in context of the specific image."
            ),
            args_schema=ELAInput,
        ),
        StructuredTool.from_function(
            func=wrap_with_cache_and_timing(
                "perform_trufor",
                _trufor_structured,
                _extract_structured_params,
            ),
            name="perform_trufor",
            description=(
                "Run TruFor AI-driven forgery detection and localization. "
                "TruFor combines RGB features with Noiseprint++ using transformer fusion. "
                "Outputs manipulation_probability (0-1 scale) and detection_score. "
                "Interpretation: manipulation_probability is calibrated - values near 0 suggest authentic, "
                "values near 1 suggest manipulated/forged. Mid-range values require additional evidence. "
                "This is your primary tool for detecting manipulated images."
            ),
            args_schema=TruForInput,
        ),
        StructuredTool.from_function(
            func=wrap_with_cache_and_timing(
                "execute_python_code",
                _python_code_structured,
                _extract_python_code_params,
            ),
            name="execute_python_code",
            description=(
                "Execute Python code dynamically for custom image analysis. "
                "This tool allows you to write Python code to analyze images, zoom in on regions, "
                "crop areas, compute statistics, create visualizations, etc. "
                "Available variables: 'image' (PIL Image), 'image_array' (numpy array), 'image_path' (str), "
                "'np' (numpy), 'Image' (PIL), 'Path' (pathlib.Path), 'base64', 'json'. "
                "You can zoom in by cropping: 'crop = image.crop((x1, y1, x2, y2)); print(crop.size)' "
                "Or analyze regions: 'region = image_array[y1:y2, x1:x2]; print(f\"Mean: {np.mean(region)}\")'"
            ),
            args_schema=PythonCodeInput,
        ),
        # CFA detection tool disabled for now
        # StructuredTool.from_function(
        #     func=_wrap_timed("perform_cfa_detection", _cfa_detection_structured, timing_hook),
        #     name="perform_cfa_detection",
        #     description=(
        #         "CFA (Color Filter Array) consistency analyzer for SPLICE DETECTION. "
        #         "Analyzes demosaicing pattern uniformity across image windows to find "
        #         "regions with inconsistent CFA artifacts (potential splices or mixed sources). "
        #         "NOTE: This tool is for LOCALIZATION, not whole-image authenticity. "
        #         "Use TruFor for AI-generated image detection. "
        #         "Output: cfa_consistency_score (0-1), distribution analysis, outlier windows. "
        #         "Low consistency or outlier windows may indicate spliced regions."
        #     ),
        #     args_schema=CFADetectionInput,
        # ),
    ]

    return tools
