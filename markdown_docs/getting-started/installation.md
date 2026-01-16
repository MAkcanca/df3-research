# Installation

Complete installation guide for DF3 including all dependencies and optional components.

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **Python** | 3.10 or later |
| **RAM** | 8 GB |
| **Storage** | 2 GB (including model weights) |
| **OS** | Windows 10+, macOS 12+, Linux (Ubuntu 20.04+) |

### Recommended for Full Performance

| Component | Recommendation |
|-----------|----------------|
| **Python** | 3.11 or 3.12 |
| **RAM** | 16 GB |
| **GPU** | NVIDIA GPU with 8GB+ VRAM (CUDA 11.8+) |
| **Storage** | SSD for faster model loading |

---

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/df3.git
cd df3
```

### 2. Create Virtual Environment

=== "Windows (PowerShell)"
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

=== "Windows (Command Prompt)"
    ```cmd
    python -m venv venv
    venv\Scripts\activate.bat
    ```

=== "Linux/macOS"
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

- **LangChain/LangGraph** — Agent orchestration
- **BAML** — Structured LLM outputs
- **PyTorch** — Neural network backend
- **PIL/OpenCV** — Image processing
- **NumPy/SciPy** — Numerical computation

### 4. Configure API Keys

Create a `.env` file in the project root:

```ini
# Primary: OpenAI API key (required for default configuration)
OPENAI_API_KEY=sk-your-openai-key

# Optional: Anthropic API key (for Claude models)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# Optional: OpenRouter API key (for multi-provider access)
OPENROUTER_API_KEY=sk-or-your-openrouter-key
```

!!! tip "API Key Security"
    Never commit `.env` files to version control. The `.gitignore` file already excludes it.

### 5. Generate BAML Client (if needed)

If you modify BAML definitions in `baml_src/`, regenerate the client:

```bash
pip install baml-py
baml-cli generate
```

### 6. Verify Installation

```bash
python scripts/analyze_image.py --image example_images/example1.jpg
```

---

## GPU Setup (Optional but Recommended)

TruFor runs significantly faster on GPU. Follow these steps for CUDA support:

### NVIDIA GPU (CUDA)

1. **Install CUDA Toolkit** (11.8 or later)
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

2. **Install cuDNN**
   - Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

3. **Install PyTorch with CUDA**
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Verify CUDA is available**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print: True
   print(torch.cuda.get_device_name(0))  # Should print your GPU name
   ```

### Force CPU Mode

If you encounter GPU issues, force CPU mode:

```bash
# Environment variable
$env:DF3_TRUFOR_DEVICE = "cpu"  # PowerShell
export DF3_TRUFOR_DEVICE=cpu    # Linux/macOS
```

---

## Model Weights

### Automatic Download

TruFor and DRUNet weights are **automatically downloaded** on first use. No manual action required.

### Manual Download (if automatic fails)

1. **TruFor weights**
   - Download: [TruFor_weights.zip](https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip)
   - Extract to: `weights/trufor/trufor.pth.tar`

2. **DRUNet weights**
   - Automatically downloaded to: `src/tools/forensic/drunet/weights/`

### Weight Storage Location

```
df3/
├── weights/
│   └── trufor/
│       └── trufor.pth.tar      # ~180 MB
├── src/
│   └── tools/
│       └── forensic/
│           └── drunet/
│               └── weights/
│                   └── drunet_gray.pth  # ~32 MB
```

---

## Provider Configuration

### OpenAI (Default)

```ini
OPENAI_API_KEY=sk-your-key-here
```

Supported models:

- `gpt-5.1`, `gpt-5.2` — Latest GPT-5 family
- `gpt-5-mini` — Faster, lower cost
- `gpt-5` — Standard GPT-5

### Anthropic

```ini
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Supported models:

- `claude-opus-4` — Highest capability
- `claude-sonnet-4` — Balanced performance
- `claude-3-5-haiku` — Fast and efficient

### OpenRouter

Access 100+ models through a single API:

```ini
OPENROUTER_API_KEY=sk-or-your-key-here
```

Usage:

```bash
python scripts/analyze_image.py \
    --image photo.jpg \
    --model google/gemini-3-flash-preview \
    --provider openrouter \
    --base-url https://openrouter.ai/api/v1
```

---

## Development Setup

For contributing to DF3:

### Install Development Dependencies

```bash
pip install -r requirements.txt
pip install pytest pytest-cov ruff mypy
```

### Run Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
ruff check src/ scripts/
ruff format src/ scripts/
```

### Type Checking

```bash
mypy src/
```

---

## Docker Installation (Alternative)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Generate BAML client
RUN baml-cli generate

ENTRYPOINT ["python", "scripts/analyze_image.py"]
```

Build and run:

```bash
docker build -t df3 .
docker run -v /path/to/images:/images df3 --image /images/photo.jpg
```

---

## Troubleshooting Installation

??? question "ModuleNotFoundError: No module named 'baml_client'"
    The BAML client needs to be generated:
    ```bash
    pip install baml-py
    baml-cli generate
    ```

??? question "CUDA out of memory"
    TruFor requires ~4GB VRAM. Solutions:
    
    1. Force CPU mode: `$env:DF3_TRUFOR_DEVICE = "cpu"`
    2. Close other GPU applications
    3. Use a GPU with more VRAM

??? question "SSL Certificate errors"
    Update certificates:
    ```bash
    pip install --upgrade certifi
    ```
    
    On macOS, also run:
    ```bash
    /Applications/Python\ 3.x/Install\ Certificates.command
    ```

??? question "Pillow/OpenCV import errors"
    Reinstall with binary packages:
    ```bash
    pip uninstall pillow opencv-python
    pip install pillow opencv-python-headless
    ```

See [Troubleshooting](../reference/troubleshooting.md) for more solutions.

---

## Next Steps

- [Quick Start](quickstart.md) — Analyze your first image
- [Your First Analysis](first-analysis.md) — Detailed walkthrough
- [Configuration](../reference/configuration.md) — Customize DF3 behavior
