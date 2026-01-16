# BAML Integration for Forensic Analysis

This document explains how to use BAML functions for forensic image analysis, following the multi-step approach to avoid reasoning degradation as described in [Instill AI's blog post](https://www.instill-ai.com/blog/llm-structured-outputs).

## Problem: Reasoning Degradation with Structured Outputs

When LLMs are constrained to strict output formats, their reasoning performance drops. The stricter the format restrictions, the more reasoning performance degrades.

## Solution: Multi-Step Approach

We use a multi-step pipeline that separates reasoning from structuring:

1. **Step 1: Vision-only reasoning (unstructured)** - Allows the LLM to reason freely without format constraints
2. **Step 2: Agent reasoning with tools** - Uses LangChain/LangGraph for tool calling (keeps unstructured output)
3. **Step 3: Structuring** - Extracts structured data from the unstructured reasoning output

This separation ensures reasoning quality is not degraded by format constraints.

## Setup

1. Install BAML:
```bash
pip install baml-py
```

2. Generate the BAML client:
```bash
cd baml_src
baml-cli generate
```

This will generate the Python client in the project root.

## Usage

### Option 1: Use BAML Functions Directly

```python
from src.agents.baml_forensic import (
    analyze_vision_only_baml,
    structure_analysis_baml,
    analyze_vision_only_structured_baml,
)

# RECOMMENDED: Vision-only analysis (two-step approach)
# This preserves reasoning quality by separating reasoning from structuring
vision_result = analyze_vision_only_baml("path/to/image.jpg")
structured = structure_analysis_baml(vision_result["reasoning_output"])

# Alternative: Combined function (may cause reasoning degradation in complex cases)
# Only use for simple cases where reasoning complexity is low
structured = analyze_vision_only_structured_baml("path/to/image.jpg")
```

### Option 2: Use ForensicAgent (BAML Always Enabled)

```python
from src.agents.forensic_agent import ForensicAgent

# Create agent (BAML is always enabled by default)
agent = ForensicAgent(
    llm_model="gpt-5.1",
    temperature=0.0
)

# Analyze image (BAML will be used for vision-only and structuring)
result = agent.analyze("path/to/image.jpg", use_tools=True)
```

## BAML Functions

### `AnalyzeImageVisionOnly`
- **Purpose**: Vision-only reasoning (unstructured output)
- **Input**: Image
- **Output**: Unstructured markdown text with reasoning
- **Use case**: First step in multi-step approach

### `StructureForensicAnalysis`
- **Purpose**: Extract structured data from unstructured reasoning
- **Input**: Unstructured reasoning text
- **Output**: `ForensicAnalysisResult` with structured fields
- **Use case**: Final step to get structured output without degrading reasoning

### `AnalyzeImageVisionOnlyStructured`
- **Purpose**: Combined vision analysis with structured output
- **Input**: Image
- **Output**: `ForensicAnalysisResult` directly
- **Use case**: Simple cases where reasoning complexity is low
- **Note**: ⚠️ This function directly returns structured output, which may cause reasoning degradation. For best results, prefer the two-step approach (`AnalyzeImageVisionOnly` + `StructureForensicAnalysis`)

## Data Models

### `ForensicAnalysisResult`
```baml
class ForensicAnalysisResult {
  verdict Verdict              // real | fake | uncertain
  confidence float             // 0.0 to 1.0
  rationale string             // Brief justification (max 80 words)
  visual_description string    // Description of image content
  forensic_summary string       // Summary of tools used
  full_text string             // Complete formatted narrative
}
```

### `Verdict` (enum)
- `REAL` - Image appears authentic
- `FAKE` - Image appears AI-generated or manipulated
- `UNCERTAIN` - Insufficient evidence

## Benefits

1. **Better Reasoning**: Unstructured reasoning step allows LLM to think freely
2. **Reliable Structure**: Structuring step ensures consistent output format
3. **Type Safety**: BAML provides type-safe function calls
4. **Maintainability**: Prompts are defined in BAML files, easier to version control

## Migration from String Prompts

The existing `prompts.py` file still works. BAML functions are now always enabled by default for better reasoning quality.

## Notes

- Agent reasoning with tools still uses LangChain/LangGraph (BAML doesn't support tool calling directly)
- BAML is used for vision-only analysis and structuring steps
- The multi-step approach ensures reasoning quality is preserved





