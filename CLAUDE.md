# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DF3 is a forensic image analysis system using an LLM agent to detect AI-generated, manipulated, and deepfake images. The agent uses vision-capable LLMs to analyze images directly and can invoke forensic tools to gather additional evidence.

## Commands

### Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Analyze a single image
```powershell
python scripts/analyze_image.py --image path/to/image.jpg
python scripts/analyze_image.py --image path/to/image.jpg --query "Is this AI-generated?"
```

### Batch evaluation
```powershell
python scripts/evaluate_llms.py --dataset data/my_eval.jsonl --models gpt-5.1 --tools both
```

### BAML client generation (after modifying baml_src/*.baml)
```powershell
pip install baml-py
# Run from repo root (default expects ./baml_src to exist)
baml-cli generate
```

## Architecture

### Core Flow
```
Image -> Vision LLM (initial analysis) -> [Optional: Forensic Tools] -> Structured Output (BAML)
```

### Key Components

**Agent** (`src/agents/forensic_agent.py`):
- `ForensicAgent` class orchestrates analysis
- Three tool policies: `agentic` (LLM decides tools), `synthetic-first`, `manipulation-first`
- Uses LangGraph for agentic tool calling, BAML for structuring outputs
- `analyze()` method for image analysis

**Forensic Tools** (`src/tools/forensic/`):
- `trufor_tools.py`: TruFor neural forgery detection (primary manipulation detector)
- `ela_tools.py`: Error Level Analysis for compression anomalies
- `jpeg_tools.py`: JPEG quantization table analysis, double-compression detection
- `frequency_tools.py`: DCT/FFT frequency domain analysis
- `noise_tools.py`: DRUNet residual extraction
- `cfa_tools.py`: Color Filter Array consistency for splice detection
- `code_execution_tool.py`: Dynamic Python code execution for custom analysis

**BAML Integration** (`baml_src/`, `src/agents/baml_forensic.py`):
- Multi-step approach to avoid reasoning degradation from structured outputs
- Step 1: Vision analysis (unstructured reasoning)
- Step 2: Structure extraction (BAML converts reasoning to structured data)
- Functions: `AnalyzeImageVisionOnly`, `StructureForensicAnalysis`

**Prompts** (`src/agents/prompts.py`):
- System prompts and user prompts for vision analysis and agent reasoning
- Prompts are separate from BAML to allow both paths

### Data Flow

1. Image encoded to base64
2. Vision LLM generates visual description
3. If tools enabled: agent (LangGraph) decides which tools to call based on visual analysis
4. Tool results fed back to agent for reasoning
5. Final output structured via BAML's `StructureForensicAnalysis`
6. Result contains: verdict (real/fake/uncertain), confidence, rationale, tool_usage

### Model Weights

TruFor and DRUNet weights auto-download on first use via `src/utils/weight_downloader.py`. Stored in `weights/trufor/` and `src/tools/forensic/drunet/weights/`.

## BAML Guidelines

When modifying BAML files in `baml_src/`:
- Always include `{{ ctx.output_format }}` in prompts for structured output
- Use `{{ _.role("user") }}` before user inputs/images
- Prefer enums over numeric confidence intervals
- Do not repeat output schema fields in prompts
- After changes, run `baml-cli generate` to regenerate the Python client
