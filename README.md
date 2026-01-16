# DF3: Agent-Focused Forensic Image Analysis

A forensic image analysis system that uses an LLM agent to detect AI-generated, manipulated, and deepfake images. The agent uses vision-capable LLMs to analyze images directly and can invoke forensic tools to gather additional evidence.

## Overview

This project focuses on **agent flows and tool usage** rather than model-based classification. The agent:

1. Receives an image directly
2. Uses vision-capable LLM (e.g., GPT-5.1) to analyze it
3. Can invoke forensic tools to gather additional evidence
4. Provides structured reasoning and conclusions via BAML

## Key Features

- **Direct image analysis**: Images go straight to the LLM (no trained classifier required)
- **Tool-based evidence gathering**: Agent can use forensic tools (TruFor, ELA, JPEG analysis, etc.)
- **Transparent reasoning**: Agent explains its analysis process with structured outputs
- **BAML integration**: Structured outputs without reasoning degradation
- **SWGDE-style reporting**: Generates forensic reports following industry standards

## Installation

### 1. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Set up environment variables

Copy the example environment file and add your API keys:

```powershell
Copy-Item .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

Or set environment variables directly:

```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

### 4. Generate BAML client (if modifying BAML files)

```powershell
baml-cli generate
```

## Usage

### Basic Analysis

Analyze an image:

```powershell
python scripts/analyze_image.py --image path/to/image.jpg
```

### Vision-Only Analysis (No Tools)

Run without forensic tools:

```powershell
python scripts/analyze_image.py --image path/to/image.jpg --no-tools
```

### With a Specific Query

Ask a specific question:

```powershell
python scripts/analyze_image.py --image path/to/image.jpg --query "Is this image AI-generated?"
```

### Custom Model

Use a different model:

```powershell
python scripts/analyze_image.py --image path/to/image.jpg --model gpt-5.1 --temperature 0.3
```

### Batch Evaluation

1. Prepare a JSONL dataset:
   ```json
   {"id": "ex1", "image": "path/to/image1.jpg", "label": "real"}
   {"id": "ex2", "image": "path/to/image2.png", "label": "fake"}
   ```

2. Run evaluation:
   ```powershell
   python scripts/evaluate_llms.py --dataset data/my_eval.jsonl --models gpt-5.1,gpt-5-mini --tools both
   ```

3. See `RUN_EVALUATE.md` for detailed evaluation options and metrics interpretation.

## Architecture

### Core Flow

```
Image -> Vision LLM (initial analysis) -> [Optional: Forensic Tools] -> Structured Output (BAML)
```

### Key Components

- **Agent** (`src/agents/forensic_agent.py`): Orchestrates analysis using LangGraph
- **BAML Integration** (`baml_src/`, `src/agents/baml_forensic.py`): Multi-step structured outputs
- **Forensic Tools** (`src/tools/forensic/`):
  - `trufor_tools.py`: TruFor neural forgery detection
  - `ela_tools.py`: Error Level Analysis
  - `jpeg_tools.py`: JPEG quantization analysis
  - `frequency_tools.py`: DCT/FFT frequency domain analysis
  - `noise_tools.py`: DRUNet residual extraction
  - `cfa_tools.py`: Color Filter Array consistency
  - `metadata_tools.py`: EXIF/metadata extraction
- **Reporting** (`src/reporting/`): SWGDE-style forensic reports

### Project Structure

```
df3/
├── baml_src/                  # BAML function definitions
│   ├── clients.baml           # LLM client configurations
│   ├── forensic_analysis.baml # Analysis functions
│   └── generators.baml        # Code generation config
├── baml_client/               # Generated BAML Python client
├── src/
│   ├── agents/
│   │   ├── forensic_agent.py  # Main agent orchestrator
│   │   ├── baml_forensic.py   # BAML integration helpers
│   │   └── prompts.py         # System/user prompts
│   ├── tools/
│   │   ├── forensic/          # Forensic tool implementations
│   │   └── forensic_tools.py  # Tool registration
│   ├── reporting/             # Report generation
│   └── utils/                 # Utilities (weight downloader, etc.)
├── scripts/
│   ├── analyze_image.py       # Single image analysis
│   ├── evaluate_llms.py       # Batch evaluation
│   └── calculate_statistics.py
├── tests/                     # Test suite
├── docs/                      # Documentation
├── weights/                   # Model weights (auto-downloaded)
├── requirements.txt
├── .env.example               # Environment template
└── README.md
```

## BAML Setup

BAML is used for structured LLM outputs without reasoning degradation. See `BAML_USAGE.md` for details.

After modifying `baml_src/*.baml` files:

```powershell
baml-cli generate
```

This regenerates the Python client in `baml_client/`.

## Documentation

- `BAML_USAGE.md`: BAML integration guide
- `RUN_EVALUATE.md`: Batch evaluation instructions
- `docs/sw.md`: SWGDE best practices reference

## Model Weights

TruFor and DRUNet weights are automatically downloaded on first use via `src/utils/weight_downloader.py`. Stored in `weights/` directory.

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.