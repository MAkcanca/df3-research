"""
BAML-based forensic analysis functions.

This module implements the multi-step approach to avoid reasoning degradation:
1. Step 1: Vision-only reasoning (unstructured) - allows LLM to reason freely
2. Step 2: Agent reasoning with tools (handled by LangChain, but uses unstructured output)
3. Step 3: Structuring (structured) - extracts structured data from unstructured reasoning

This separation ensures reasoning quality is not degraded by format constraints,
as described in: https://www.instill-ai.com/blog/llm-structured-outputs
"""

import os
import base64
import logging
import hashlib
from typing import Dict, Any, Optional
from pathlib import Path

from baml_client import b
from baml_py import Image as BamlImage
import baml_py.baml_py as baml_py_core

# Import cache for vision model output caching
try:
    from ..tools.forensic.cache import get_cache
except ImportError:
    # Fallback if cache module not available
    get_cache = None

logger = logging.getLogger(__name__)

# Default model if not specified
DEFAULT_BAML_MODEL = "gpt-5-mini"


def _vision_cache_tag() -> str:
    """
    Return a cache discriminator for vision outputs so cached runs are comparable/reproducible.

    Why: the vision cache key must change if prompts / BAML definitions change; otherwise
    old outputs can be silently reused after prompt edits, invalidating evaluations.
    """
    env = os.getenv("DF3_VISION_CACHE_TAG")
    if env:
        return str(env)
    # Best-effort: hash the BAML source file that defines the vision prompts.
    # baml_forensic.py -> src/agents; repo root is 2 parents up.
    try:
        repo_root = Path(__file__).resolve().parents[2]
        baml_path = repo_root / "baml_src" / "forensic_analysis.baml"
        if baml_path.exists():
            h = hashlib.sha256(baml_path.read_bytes()).hexdigest()[:16]
            return f"baml:{h}"
    except Exception:
        pass
    return "unknown"


def _create_client_registry(
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> baml_py_core.ClientRegistry:
    """
    Create a BAML ClientRegistry that overrides DynamicForensicClient with the specified model.
    
    Args:
        model: Model name to use (e.g., "gpt-5.1", "gpt-5-mini")
        api_key: Optional API key (defaults to OPENAI_API_KEY env var)
        base_url: Optional base URL for the API
        default_headers: Optional default headers
        
    Returns:
        ClientRegistry configured to use the specified model and options
    """
    # Create a new client registry
    cr = baml_py_core.ClientRegistry()
    
    # Build options dict
    options: Dict[str, Any] = {
        "model": model,
        "api_key": api_key or os.environ.get("OPENAI_API_KEY"),
    }
    
    if base_url:
        options["base_url"] = base_url
    
    if default_headers:
        options["default_headers"] = default_headers
    
    # Override the DynamicForensicClient with the specified model.
    #
    # Provider notes:
    # - For native OpenAI, we prefer "openai-responses" (Responses API).
    # - For OpenRouter (and other OpenAI-compatible gateways), the Responses API is often
    #   not supported; "openai" (Chat Completions) is more broadly compatible.
    #
    # The client name must match what's defined in clients.baml.
    provider = "openai-responses"
    if base_url and "openrouter.ai" in str(base_url).lower():
        provider = "openai"

    cr.add_llm_client(
        name="DynamicForensicClient",
        provider=provider,
        options=options,
    )
    
    return cr


async def analyze_vision_only_baml(
    image_path: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Step 1: Vision-only analysis using BAML (unstructured output).
    
    This allows the LLM to reason freely without format constraints,
    avoiding the reasoning degradation problem.
    
    Args:
        image_path: Path to the image file
        model: Model name to use (defaults to gpt-5-mini if not specified)
        
    Returns:
        Dictionary with unstructured reasoning output
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    model = model or DEFAULT_BAML_MODEL
    logger.info(f"Running vision-only BAML analysis for: {image_path} (model: {model})")
    
    # Load image using BAML's image type
    # Read file and convert to base64 as per BAML docs
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Determine MIME type from extension
    image_ext = Path(image_path).suffix.lower()
    mime_type = "image/jpeg" if image_ext in [".jpg", ".jpeg"] else "image/png" if image_ext == ".png" else "image/jpeg"
    
    image = BamlImage.from_base64(mime_type, encoded_string)
    
    # Create client registry with the specified model and call BAML function
    client_registry = _create_client_registry(model, api_key, base_url, default_headers)
    # Call BAML function - returns unstructured string (free reasoning)
    reasoning_output = await b.AnalyzeImageVisionOnly(image, baml_options={"client_registry": client_registry})
    
    return {
        "reasoning_output": reasoning_output,
        "image_path": image_path,
    }


async def structure_analysis_baml(
    reasoning_output: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Step 3: Structure the unstructured reasoning output.
    
    This step extracts structured data from the free-form reasoning,
    ensuring we get reliable structured output without degrading reasoning quality.
    
    Args:
        reasoning_output: Unstructured reasoning text from vision or agent analysis
        model: Model name to use (defaults to gpt-5-mini if not specified)
        
    Returns:
        Structured dictionary matching ForensicAnalysisResult
    """
    model = model or DEFAULT_BAML_MODEL
    logger.info(f"Structuring reasoning output using BAML (model: {model})")
    
    # Create client registry with the specified model and call BAML function
    client_registry = _create_client_registry(model, api_key, base_url, default_headers)
    # Call BAML structuring function
    structured_result = await b.StructureForensicAnalysis(reasoning_output, baml_options={"client_registry": client_registry})
    
    # Convert BAML result to dictionary
    return {
        "verdict": structured_result.verdict.value.lower(),  # Convert enum to string
        "confidence": structured_result.confidence,
        "rationale": structured_result.rationale,
        "visual_description": structured_result.visual_description,
        "forensic_summary": structured_result.forensic_summary,
        "full_text": structured_result.full_text,
    }


async def analyze_vision_only_structured_baml(
    image_path: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Combined vision-only analysis with structured output (for simple cases).
    
    This is a convenience function that combines vision reasoning + structuring.
    For complex cases, prefer the two-step approach (analyze_vision_only_baml + structure_analysis_baml).
    
    Args:
        image_path: Path to the image file
        model: Model name to use (defaults to gpt-5-mini if not specified)
        
    Returns:
        Structured dictionary matching ForensicAnalysisResult
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    model = model or DEFAULT_BAML_MODEL
    
    # Check cache first
    if get_cache is not None:
        cache = get_cache()
        cached_result = cache.get_vision_output(
            vision_model=model,
            image_path=image_path,
            cache_tag=_vision_cache_tag(),
        )
        if cached_result is not None:
            logger.info(f"Using cached vision output for: {image_path} (model: {model})")
            return cached_result
    
    logger.info(f"Running structured vision-only BAML analysis for: {image_path} (model: {model})")
    
    # Load image using BAML's image type
    # Read file and convert to base64 as per BAML docs
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Determine MIME type from extension
    image_ext = Path(image_path).suffix.lower()
    mime_type = "image/jpeg" if image_ext in [".jpg", ".jpeg"] else "image/png" if image_ext == ".png" else "image/jpeg"
    
    image = BamlImage.from_base64(mime_type, encoded_string)
    
    # Create client registry with the specified model and call BAML function
    client_registry = _create_client_registry(model, api_key, base_url, default_headers)
    # Call BAML function - returns structured output directly
    structured_result = await b.AnalyzeImageVisionOnlyStructured(image, baml_options={"client_registry": client_registry})
    
    # Convert BAML result to dictionary
    result = {
        "verdict": structured_result.verdict.value.lower(),  # Convert enum to string
        "confidence": structured_result.confidence,
        "rationale": structured_result.rationale,
        "visual_description": structured_result.visual_description,
        "forensic_summary": structured_result.forensic_summary,
        "full_text": structured_result.full_text,
    }
    
    # Cache the result
    if get_cache is not None:
        cache = get_cache()
        cache.set_vision_output(
            vision_model=model,
            image_path=image_path,
            output=result,
            cache_tag=_vision_cache_tag(),
        )
    
    return result


def analyze_with_multi_step_baml(image_path: str, agent_reasoning_output: Optional[str] = None) -> Dict[str, Any]:
    """
    Multi-step analysis following the Instill AI approach:
    1. Vision-only reasoning (unstructured)
    2. Agent reasoning with tools (handled separately by LangChain)
    3. Structure the final output
    
    Args:
        image_path: Path to the image file
        agent_reasoning_output: Optional unstructured output from agent analysis with tools
        
    Returns:
        Structured dictionary with analysis results
    """
    if agent_reasoning_output:
        # Use agent reasoning output if provided
        logger.info("Structuring agent reasoning output")
        structured = structure_analysis_baml(agent_reasoning_output)
    else:
        # Fall back to vision-only analysis
        logger.info("Running vision-only analysis")
        vision_result = analyze_vision_only_baml(image_path)
        structured = structure_analysis_baml(vision_result["reasoning_output"])
    
    return structured

