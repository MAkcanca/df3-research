"""
Entry point script for analyzing images with the forensic agent.
"""

import argparse
import sys
import logging
import os
from typing import Dict, Optional
from pathlib import Path

# Load environment variables first
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import ForensicAgent  # noqa: E402

# Configure logfire after imports - completely disabled (no tracing, no sending to LangSmith)
logfire.configure(scrubbing=False, console=False, send_to_logfire=False)

def build_headers(referer: Optional[str], title: Optional[str]) -> Optional[Dict[str, str]]:
    """Optional headers for OpenRouter."""
    headers: Dict[str, str] = {}
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers or None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze an image using the forensic agent"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file to analyze"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional specific question about the image"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="LLM model to use (default: gpt-5.1)"
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default=None,
        help="Optional model override for the vision-only BAML step (defaults to --model).",
    )
    parser.add_argument(
        "--structuring-model",
        type=str,
        default=None,
        help="Optional model override for the BAML structuring step (defaults to --model).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature (default: 0.2)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "openrouter"],
        help="Provider for LLM calls (default: openai)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key. Defaults to environment variable (provider specific).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional API base URL (e.g., https://openrouter.ai/api/v1).",
    )
    parser.add_argument(
        "--referer",
        type=str,
        default=None,
        help="Optional HTTP-Referer header for OpenRouter.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional X-Title header for OpenRouter.",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Run vision-only analysis without forensic tools"
    )
    
    args = parser.parse_args()

    # Provider defaults (match evaluate_llms.py behavior)
    base_url = args.base_url
    if args.provider == "openrouter" and not base_url:
        base_url = "https://openrouter.ai/api/v1"

    default_headers = build_headers(args.referer, args.title)
    if args.provider == "openai":
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    else:
        api_key = args.api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "API key is required. Set --api-key or the appropriate environment variable "
            "(OPENAI_API_KEY for OpenAI, OPENROUTER_API_KEY for OpenRouter)."
        )

    # Initialize agent
    print(f"Initializing forensic agent with model: {args.model}...")
    agent = ForensicAgent(
        llm_model=args.model,
        vision_model=args.vision_model,
        structuring_model=args.structuring_model,
        temperature=args.temperature,
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
    )
    
    # Analyze image
    print(f"\nAnalyzing image: {args.image}")
    if args.query:
        print(f"Query: {args.query}")
    print("-" * 60)
    
    try:
        result = agent.analyze(args.image, args.query, use_tools=not args.no_tools)
        
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        
        # Show structured results if available
        if result.get('verdict'):
            print(f"\nVerdict: {result['verdict'].upper()}")
            print(f"Confidence: {result.get('confidence', 0.0):.2f}")
            if result.get('rationale'):
                print(f"\nRationale: {result['rationale']}")
            if result.get('visual_description'):
                print(f"\nVisual Description: {result['visual_description'][:200]}...")
        
        print(f"\nConclusion:\n{result['conclusion']}")
        
        if result['tool_usage']:
            print(f"\nTools used: {', '.join(result['tool_usage'])}")
        else:
            print("\nNo tools were used.")
        
        # SWGDE-style form/report (DF3-adapted)
        report_md = result.get("report_markdown")
        if isinstance(report_md, str) and report_md.strip():
            print("\n" + "=" * 60)
            print("IMAGE AUTHENTICATION REPORT (SWGDE-STYLE)")
            print("=" * 60)
            print(report_md)

        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
