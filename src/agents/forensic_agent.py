"""
Forensic Agent

A simplified LLM agent that receives images directly and uses forensic tools
to analyze them. No model classification - pure agent reasoning with tools.
"""

import os
import json
import base64
import sys
import logging
import time
import asyncio
import uuid
import re
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Load environment variables first (before setting LangSmith vars)
from dotenv import load_dotenv
load_dotenv()

# Completely disable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGSMITH_TRACING"] = "false"

# Suppress OpenTelemetry warning about mixed types in langgraph_path attribute
# This is a known issue in LangGraph's tracing where node names (str) and step indices (int)
# are mixed in the path sequence. It's harmless but noisy.
logging.getLogger('opentelemetry.attributes').setLevel(logging.ERROR)

import logfire  # noqa: E402

from langchain_openai import ChatOpenAI  # noqa: E402

# Configure logfire after imports - completely disabled (no tracing, no sending to LangSmith)
logfire.configure(scrubbing=False, console=False, send_to_logfire=False)
from langchain_core.messages import HumanMessage, ToolMessage  # noqa: E402
try:
    from langchain.agents import create_react_agent
except ImportError:
    from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from ..tools.forensic_tools import create_forensic_tools  # noqa: E402
from .prompts import (  # noqa: E402
    build_agent_prompt,
    get_system_prompt,
    get_vision_system_prompt,
    get_vision_user_prompt,
)

from .baml_forensic import (  # noqa: E402
    structure_analysis_baml,
    analyze_vision_only_structured_baml,
)

from ..reporting import generate_image_authentication_report  # noqa: E402

# Set up logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Constants for memory management
IMAGE_CACHE_SIZE = 32  # Number of encoded images to cache


def _get_image_hash(image_path: str) -> str:
    """Get SHA256 hash of image file content."""
    try:
        with open(image_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return ""


@lru_cache(maxsize=IMAGE_CACHE_SIZE)
def _cached_encode_image(image_path: str, image_hash: str) -> str:
    """
    Encode image to base64 with caching based on path and file hash.
    
    The image_hash parameter ensures cache invalidation when the file content changes.
    Same content = same hash, regardless of path or modification time.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def clear_image_cache() -> None:
    """Clear the image encoding cache. Useful for freeing memory."""
    _cached_encode_image.cache_clear()


class ForensicAgent:
    """
    Simplified forensic agent that receives images directly.
    
    The agent:
    1. Receives an image path
    2. Analyzes it using vision-capable LLM
    3. Can use forensic tools to gather more evidence
    4. Provides reasoning and analysis
    """
    
    def __init__(self,
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
                 enable_checkpointer: bool = True):
        """
        Args:
            llm_model: OpenAI model name (should support vision, e.g., gpt-5.1)
            temperature: LLM temperature
            reasoning_effort: Reasoning effort level for the model
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            max_iterations: Maximum number of agent iterations (tool calls + reasoning cycles).
                          Default is 15, which balances tool usage with performance. Set to None for no limit
                          (not recommended as it could run indefinitely).
        """
        # Agent/tool-calling model (LangGraph). Vision + structuring can be overridden separately.
        llm_kwargs = {
            "model": llm_model,
            "temperature": temperature,
            "api_key": api_key or os.getenv("OPENAI_API_KEY"),
        }

        if reasoning_effort:
            llm_kwargs["reasoning_effort"] = reasoning_effort
        if base_url:
            llm_kwargs["base_url"] = base_url
        if default_headers:
            llm_kwargs["default_headers"] = default_headers
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # Store API configuration for BAML functions
        self.llm_model = llm_model
        self.vision_model = vision_model or llm_model
        self.structuring_model = structuring_model or llm_model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.default_headers = default_headers
        # Currently unused; accepted for UI/backwards-compat while policies evolve.
        self.decision_policy = decision_policy

        self.max_iterations = max_iterations
        self.enable_checkpointer = bool(enable_checkpointer)

        # Create the agent executor (tool-calling graph) for agentic flow
        self._current_tool_timings: Optional[List[Dict[str, Any]]] = None
        self.tools = create_forensic_tools(timing_hook=self._tool_timing_hook)
        self.agent_executor = self._create_agent()

    def _tool_timing_hook(self, tool_name: str, seconds: float, error: Optional[str]) -> None:
        """Internal hook used by wrapped tools to record timing per analyze() call."""
        if self._current_tool_timings is None:
            return
        self._current_tool_timings.append(
            {
                "tool": tool_name,
                "seconds": float(seconds),
                "error": error,
            }
        )
    
    def _create_agent(self):
        """Create LangGraph agent with forensic tools."""
        system_prompt = get_system_prompt()

        kwargs = {
            "model": self.llm,
            "tools": self.tools,
            "prompt": system_prompt,
        }
        if self.enable_checkpointer:
            kwargs["checkpointer"] = MemorySaver()

        graph = create_react_agent(**kwargs)
        
        self.system_prompt = system_prompt
        return graph
    
    @staticmethod
    def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON, tolerating extra text around the object."""
        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    return None
        return None

    @staticmethod
    def _extract_json_dicts(text: str) -> List[Tuple[int, int, Dict[str, Any]]]:
        """
        Extract JSON objects embedded in a larger text blob.

        This is more robust than slicing from first '{' to last '}' because the text may
        contain multiple JSON objects (e.g., tool outputs + a final answer payload).
        """
        if not text:
            return []
        s = str(text)
        out: List[Tuple[int, int, Dict[str, Any]]] = []
        in_str = False
        esc = False
        depth = 0
        start: Optional[int] = None
        for i, ch in enumerate(s):
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                elif ch == "\"":
                    in_str = False
                continue

            # not in string
            if ch == "\"":
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
                continue
            if ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = s[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            out.append((start, i + 1, obj))
                    except Exception:
                        pass
                    start = None
        return out

    @classmethod
    def _extract_result_payload(cls, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract the final result payload from an LLM response.

        Strategy:
        1) Try tolerant JSON parse (whole blob / first-to-last).
        2) Extract all embedded JSON dicts; pick the last one that contains 'verdict' (preferred),
           otherwise the last dict that contains 'confidence'.
        3) Fall back to regex extraction for Verdict/Confidence lines.
        """
        parsed = cls._safe_json_loads(text)
        if isinstance(parsed, dict):
            return parsed

        dicts = cls._extract_json_dicts(text)
        if dicts:
            # Prefer the last dict containing 'verdict'
            for _, _, obj in reversed(dicts):
                if "verdict" in obj:
                    return obj
            # Otherwise, accept the last dict that at least contains confidence
            for _, _, obj in reversed(dicts):
                if "confidence" in obj:
                    return obj

        # Regex fallback (covers pure-markdown answers)
        t = (text or "")
        verdict_match = re.search(
            r"(?i)\\bverdict\\b\\s*[:\\-]\\s*(real|fake|uncertain|inconclusive)\\b", t
        )
        conf_match = re.search(r"(?i)\\bconfidence\\b[^0-9]{0,30}([01](?:\\.\\d+)?)\\b", t)
        if verdict_match or conf_match:
            out: Dict[str, Any] = {}
            if verdict_match:
                out["verdict"] = verdict_match.group(1).strip().lower()
            if conf_match:
                try:
                    out["confidence"] = float(conf_match.group(1))
                except Exception:
                    pass
            return out or None

        return None
    
    @staticmethod
    def _normalize_verdict(verdict: Optional[str]) -> str:
        """Normalize verdict to real/fake/uncertain."""
        if not verdict:
            return "uncertain"
        v = str(verdict).strip().lower()
        # Normalize punctuation and whitespace for robustness (e.g., "Inconclusive.", "can't determine")
        v = re.sub(r"[\s_]+", " ", v)
        v = re.sub(r"[^a-z0-9\-\s]", "", v)
        v = " ".join(v.split())

        # Exact/near-exact matches only to avoid substring hits like "uncertain" -> "ai"
        fake_tokens = {
            "fake",
            "ai-generated",
            "ai generated",
            "synthetic",
            "manipulated",
            "tampered",
            "tamper",
            "deepfake",
            "ai",
        }
        real_tokens = {"real", "authentic", "natural", "genuine"}
        uncertain_tokens = {
            "uncertain",
            "inconclusive",
            "unknown",
            "undetermined",
            "unsure",
            "not sure",
            "cannot determine",
            "cant determine",
            "impossible to tell",
        }
        if v in fake_tokens:
            return "fake"
        if v in real_tokens:
            return "real"
        if v in uncertain_tokens:
            return "uncertain"
        return "uncertain"
    
    def _build_result(
        self,
        raw_text: str,
        parsed: Optional[Dict[str, Any]],
        tool_usage: List[str],
        image_path: str,
        prompts: Dict[str, str],
    ) -> Dict[str, Any]:
        """Build a structured result dictionary."""
        verdict = self._normalize_verdict((parsed or {}).get("verdict"))
        try:
            confidence_val = float((parsed or {}).get("confidence", 0.0))
        except Exception:
            confidence_val = 0.0
        confidence_val = max(0.0, min(1.0, confidence_val))
        
        return {
            "verdict": verdict,
            "confidence": confidence_val,
            "rationale": (parsed or {}).get("rationale") or "",
            "visual_description": (parsed or {}).get("visual_description")
            or (parsed or {}).get("visual_summary")
            or "",
            "forensic_summary": (parsed or {}).get("forensic_summary", ""),
            "sections": (parsed or {}).get("sections") or {},
            "raw_text": raw_text,
            "conclusion": raw_text,
            "tool_usage": tool_usage,
            "image_path": image_path,
            "prompts": prompts,
            "raw_parsed": parsed or {},
        }
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 for vision API with caching.
        
        Uses a module-level LRU cache keyed by file path and file hash
        to avoid redundant encoding of the same image across multiple calls
        (e.g., vision-only analysis, agent analysis, and retry attempts).
        Same content = same hash, ensuring cache works even if files are moved/copied.
        """
        image_hash = _get_image_hash(image_path)
        return _cached_encode_image(image_path, image_hash)
    
    @staticmethod
    def _get_mime_type(image_path: str) -> str:
        """Determine MIME type from image file extension."""
        image_ext = Path(image_path).suffix.lower()
        if image_ext in [".jpg", ".jpeg"]:
            return "image/jpeg"
        elif image_ext == ".png":
            return "image/png"
        elif image_ext == ".webp":
            return "image/webp"
        elif image_ext == ".gif":
            return "image/gif"
        else:
            # Default to JPEG for unknown extensions
            return "image/jpeg"

    
    def analyze(self, image_path: str, user_query: Optional[str] = None, use_tools: bool = True, pass_image_to_agent: bool = False) -> Dict:
        """
        Analyze an image using the forensic agent. Returns a structured verdict.
        
        Args:
            image_path: Path to the image file
            user_query: Optional specific question about the image
            use_tools: If False, run a simple vision-only prompt with no tools
            pass_image_to_agent: If True, include the image directly in the agent message.
                                If False, agent only receives text description. Default True.
        
        Returns:
            Dictionary with analysis results:
            {
                'verdict': 'real' | 'fake' | 'uncertain',
                'confidence': float 0-1,
                'rationale': str,
                'visual_description': str,
                'forensic_summary': str,
                'raw_text': str,
                'tool_usage': list,
                'prompts': dict
            }
        """
        # Verify image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.info(f"Starting analysis (non-streaming) for image: {image_path}, use_tools: {use_tools}")

        timings: Dict[str, Any] = {}
        t_total0 = time.perf_counter()

        # Shared simple vision prompt (used for both modes)
        vision_system_prompt = get_vision_system_prompt()
        vision_user_prompt = get_vision_user_prompt()
        
        prompts: Dict[str, str] = {
            "vision_system": vision_system_prompt,
            "vision_user": vision_user_prompt,
            "agent_system": "",
            "agent_user": "",
        }
        
        # Phase 0: always get a vision-only description first
        # IMPORTANT: we intentionally use the BAML vision function even in tool mode.
        # This makes the initial vision summary far more reliable than ad-hoc JSON parsing
        # (especially for smaller models), improving downstream tool selection and reasoning.
        logger.info(f"Using BAML for vision-only analysis (model: {self.vision_model})")
        t0 = time.perf_counter()
        vision_result_dict = asyncio.run(
            analyze_vision_only_structured_baml(
                image_path,
                model=self.vision_model,
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=self.default_headers,
            )
        )
        timings["vision_llm_seconds"] = float(time.perf_counter() - t0)
        vision_raw = vision_result_dict.get("full_text", "")
        vision_parsed = vision_result_dict
        
        if not use_tools:
            tool_usage: List[str] = []
            built = self._build_result(
                raw_text=vision_raw,
                parsed=vision_parsed,
                tool_usage=tool_usage,
                image_path=image_path,
                prompts=prompts,
            )
            timings["total_seconds"] = float(time.perf_counter() - t_total0)
            built["timings"] = timings
            built["models"] = {
                "agent": self.llm_model,
                "vision": self.vision_model,
                "structuring": self.structuring_model,
            }
            built["report_markdown"] = generate_image_authentication_report(built)
            return built

        # Agentic mode: LLM decides which tools to call.
        # IMPORTANT: avoid anchoring the tool-using agent on a vision-only verdict,
        # especially for weaker models. Use the description (what's visible), not the
        # vision step's final conclusion.
        visual_summary = vision_parsed.get("visual_description") or vision_raw
        agent_prompt = build_agent_prompt(visual_summary=visual_summary, image_path=image_path)
        prompts["agent_system"] = self.system_prompt
        prompts["agent_user"] = agent_prompt

        # Optionally include the image directly in the message so the agent can see it
        if pass_image_to_agent:
            base64_image = self._encode_image(image_path)
            mime_type = self._get_mime_type(image_path)
            messages = [
                HumanMessage(content=[
                    {"type": "text", "text": agent_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                    }
                ])
            ]
        else:
            messages = [
                HumanMessage(content=agent_prompt)
            ]

        # Calculate recursion_limit from max_iterations
        # Each iteration uses 2 steps (action + observation), so recursion_limit = 2 * self.max_iterations + 1
        # Use unique thread_id per invocation to avoid polluted message history from previous runs
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        if self.max_iterations is not None:
            config["recursion_limit"] = 2 * self.max_iterations + 1

        if self.agent_executor is None:
            raise RuntimeError("agent_executor is not initialized.")

        logger.info("Invoking agent executor (non-streaming mode)")
        tool_timings: List[Dict[str, Any]] = []
        self._current_tool_timings = tool_timings
        start_time = time.perf_counter()
        try:
            result = self.agent_executor.invoke(
                {"messages": messages},
                config=config
            )
        finally:
            timings["agent_graph_seconds"] = float(time.perf_counter() - start_time)
            self._current_tool_timings = None
        execution_time = timings.get("agent_graph_seconds", 0.0)
        logger.info(f"Agent executor completed in {execution_time:.2f}s")

        if isinstance(result, dict) and 'messages' in result:
            messages = result['messages']
            final_message = messages[-1] if messages else None
            if final_message:
                output = final_message.content if hasattr(final_message, 'content') else str(final_message)
            else:
                output = "No response generated"
        else:
            output = str(result)

        # Track tool usage from agent messages
        tool_usage = []
        tool_details: List[Dict[str, Any]] = []
        tool_results: List[Dict[str, Any]] = []
        timing_idx = 0
        if isinstance(result, dict) and 'messages' in result:
            for msg in result['messages']:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get('name') if isinstance(tool_call, dict) else getattr(tool_call, 'name', None)
                        if tool_name:
                            tool_usage.append(tool_name)

                if isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, "name", None) or "unknown"
                    tool_result = msg.content if hasattr(msg, "content") else str(msg)
                    parsed_tool = self._safe_json_loads(tool_result) or {}
                    seconds = None
                    if timing_idx < len(tool_timings):
                        seconds = tool_timings[timing_idx].get("seconds")
                        timing_idx += 1
                    tool_details.append(
                        {
                            "tool": tool_name,
                            "seconds": seconds,
                            "status": parsed_tool.get("status", "unknown"),
                            "error": parsed_tool.get("error"),
                        }
                    )
                    # Store full parsed tool output for downstream debugging/analysis.
                    # Tool outputs are already JSON strings; most tools do not include large blobs
                    # (we disable maps by default for TruFor/ELA), but keep a truncated raw copy anyway.
                    raw_str = tool_result if isinstance(tool_result, str) else str(tool_result)
                    tool_results.append(
                        {
                            "tool": tool_name,
                            "seconds": seconds,
                            "status": parsed_tool.get("status", "unknown"),
                            "error": parsed_tool.get("error"),
                            "parsed": parsed_tool,
                            "raw_truncated": raw_str[:8000],
                        }
                    )

        # Structure the output using BAML
        # Pass the model and API config so BAML uses the correct model instead of hardcoded gpt-5-mini
        logger.info(f"Using BAML to structure agent output (model: {self.structuring_model})")
        try:
            structured = asyncio.run(structure_analysis_baml(
                output,
                model=self.structuring_model,
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=self.default_headers,
            ))
            parsed = structured
        except Exception as e:
            logger.error(f"BAML structuring failed: {e}. BAML is required.")
            raise

        logger.info(f"Analysis complete. Tools used: {tool_usage}")

        built = self._build_result(
            raw_text=output,
            parsed=parsed,
            tool_usage=tool_usage,
            image_path=image_path,
            prompts=prompts,
        )
        built["tool_details"] = tool_details
        built["tool_results"] = tool_results
        timings["total_seconds"] = float(time.perf_counter() - t_total0)
        built["timings"] = timings
        built["models"] = {
            "agent": self.llm_model,
            "vision": self.vision_model,
            "structuring": self.structuring_model,
        }
        built["report_markdown"] = generate_image_authentication_report(built)
        return built
