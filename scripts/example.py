"""
Simple example of using the forensic agent.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import ForensicAgent


def main():
    # Example: Analyze an image
    # Replace with your image path
    # Get project root
    use_tools = "--tools" in sys.argv  # default to vision-only simple prompt
    # Image path is the first non keyword argument, should not take --tools as argument
    
    # Filter out keyword arguments to find the image path
    keyword_args = {"--tools"}
    non_keyword_args = [arg for arg in sys.argv[1:] if arg not in keyword_args]
    
    if not non_keyword_args:
        print("Error: Please provide an image path as an argument.")
        print("Usage: python example.py [--tools] <image_path>")
        sys.exit(1)
    
    image_path = non_keyword_args[0]
    
    print("Initializing forensic agent...")
    agent = ForensicAgent(
        llm_model="gpt-5.1",
        vision_model=None,
        structuring_model=None,
        temperature=None,
        reasoning_effort="medium",
    )
    
    mode = "full (with tools)" if use_tools else "vision-only (no tools)"
    print(f"\nAnalyzing image: {image_path} [{mode}]")
    print("-" * 60)
    
    try:
        # Run analysis
        result = agent.analyze(image_path, use_tools=use_tools)
        
        print("\nAnalysis Results:")
        print("=" * 60)
        print(result['conclusion'])
        print("\nTools used:", result['tool_usage'] if result['tool_usage'] else "None")

        report_md = result.get("report_markdown")
        if isinstance(report_md, str) and report_md.strip():
            print("\n" + "=" * 60)
            print("IMAGE AUTHENTICATION REPORT (SWGDE-STYLE)")
            print("=" * 60)
            print(report_md)
        
    except FileNotFoundError:
        print(f"\nError: Image file not found: {image_path}")
        print("Please update the image_path variable with a valid image path.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
