# Quick Start

Get up and running with DF3 in under 5 minutes.

---

## Prerequisites

- **Python 3.10+** — DF3 requires Python 3.10 or later
- **API Key** — OpenAI, Anthropic, or OpenRouter API key
- **8GB+ RAM** — For loading forensic tool models (TruFor, DRUNet)
- **GPU (optional)** — CUDA GPU accelerates TruFor inference

---

## 1. Clone and Install

```powershell
# Clone the repository
git clone https://github.com/your-org/df3.git
cd df3

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Configure API Keys

Create a `.env` file in the project root:

```ini
# OpenAI (primary provider)
OPENAI_API_KEY=sk-your-key-here

# Optional: Anthropic for Claude models
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Optional: OpenRouter for multi-provider access
OPENROUTER_API_KEY=sk-or-your-key-here
```

Or set environment variables directly:

=== "Windows PowerShell"
    ```powershell
    $env:OPENAI_API_KEY = "sk-your-key-here"
    ```

=== "Linux/macOS"
    ```bash
    export OPENAI_API_KEY="sk-your-key-here"
    ```

---

## 3. Analyze Your First Image

```powershell
python scripts/analyze_image.py --image path/to/image.jpg
```

**Example output:**

```
Initializing forensic agent with model: gpt-5.1...

Analyzing image: path/to/image.jpg
------------------------------------------------------------

==============================================================
ANALYSIS RESULTS
==============================================================

Verdict: FAKE
Confidence: 0.85

Rationale: Multiple indicators suggest AI generation: the subject's 
left hand displays six fingers with unnatural joint positioning, 
skin texture appears uniformly smooth without visible pores, and 
background elements show characteristic "melting" artifacts where 
objects blend into undefined shapes.

Visual Description: Portrait of a woman in her 30s against a 
blurred outdoor background. Subject has brown hair and is smiling.

Tools used: perform_trufor, perform_ela

==============================================================
```

---

## 4. Try Different Modes

### Vision-Only (Faster)

Skip forensic tools for quick visual-only analysis:

```powershell
python scripts/analyze_image.py --image photo.jpg --no-tools
```

### Custom Query

Ask a specific question about the image:

```powershell
python scripts/analyze_image.py --image photo.jpg --query "Is this a deepfake?"
```

### Different Model

Use a different LLM:

```powershell
python scripts/analyze_image.py --image photo.jpg --model gpt-5-mini
```

---

## 5. Run a Batch Evaluation

Evaluate multiple images with ground truth labels:

```powershell
# Prepare a dataset file (JSONL format)
# Each line: {"id": "img1", "image": "path/to/img.jpg", "label": "real"}

python scripts/evaluate_llms.py \
    --dataset data/my_dataset.jsonl \
    --models gpt-5.1 \
    --tools both \
    --output results.jsonl
```

---

## What's Next?

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
### :material-book-open-variant: Full Installation Guide
Detailed setup instructions including GPU configuration.

[Installation →](installation.md)
</div>

<div class="feature-card" markdown>
### :material-magnify: Deep Dive
Understand how DF3 analyzes images.

[How It Works →](../guide/how-it-works.md)
</div>

<div class="feature-card" markdown>
### :material-tools: Forensic Tools
Learn about each forensic tool and what it detects.

[Tools Reference →](../tools/overview.md)
</div>

<div class="feature-card" markdown>
### :material-chart-bar: Evaluation
Run comprehensive benchmarks.

[Batch Evaluation →](../guide/batch-evaluation.md)
</div>

</div>

---

## Troubleshooting Quick Fixes

??? question "TruFor weights not found"
    Weights are auto-downloaded on first use. If download fails:
    
    1. Manually download from [TruFor releases](https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip)
    2. Extract to `weights/trufor/trufor.pth.tar`

??? question "Out of memory errors"
    TruFor requires significant GPU memory. Solutions:
    
    - Set `DF3_TRUFOR_DEVICE=cpu` to use CPU (slower but works)
    - Reduce image size before analysis
    - Use `--no-tools` for vision-only analysis

??? question "API rate limits"
    If hitting rate limits:
    
    - Add delays between calls in batch evaluation (`--num-workers 1`)
    - Use a different model (smaller models have higher limits)
    - Consider OpenRouter for load balancing

See [Troubleshooting](../reference/troubleshooting.md) for more solutions.
