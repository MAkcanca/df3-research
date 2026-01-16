# Environment Variables

API keys and configuration via environment.

---

## Quick Setup

```powershell
# Copy template
Copy-Item .env.example .env

# Edit with your keys
notepad .env
```

---

## Required Variables

### LLM Provider Keys

You need at least one API key for an LLM provider:

| Variable | Provider | Required |
|----------|----------|----------|
| `OPENAI_API_KEY` | OpenAI | Yes (if using OpenAI) |
| `OPENROUTER_API_KEY` | OpenRouter | Yes (if using OpenRouter) |
| `ANTHROPIC_API_KEY` | Anthropic | Optional |
| `GOOGLE_API_KEY` | Google AI | Optional |

### Example `.env`

```ini
# Primary LLM provider
OPENAI_API_KEY=sk-proj-...

# Alternative: OpenRouter for multiple providers
OPENROUTER_API_KEY=sk-or-v1-...

# Optional providers
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

---

## Variable Reference

### OPENAI_API_KEY

**OpenAI API access key**

- **Format**: `sk-proj-...` or `sk-...`
- **Source**: [OpenAI Platform](https://platform.openai.com/api-keys)
- **Required for**: GPT models, default analysis

### OPENROUTER_API_KEY

**OpenRouter unified API key**

- **Format**: `sk-or-v1-...`
- **Source**: [OpenRouter](https://openrouter.ai/)
- **Required for**: Multi-provider access, some models

### ANTHROPIC_API_KEY

**Anthropic API key for Claude models**

- **Format**: `sk-ant-...`
- **Source**: [Anthropic Console](https://console.anthropic.com/)
- **Required for**: Claude models

### GOOGLE_API_KEY

**Google AI Studio key for Gemini**

- **Format**: `AIza...`
- **Source**: [Google AI Studio](https://aistudio.google.com/)
- **Required for**: Gemini models

---

## Loading Variables

### Automatic Loading

DF3 automatically loads `.env` using `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env from project root
```

### Manual Setting (PowerShell)

```powershell
# Session only
$env:OPENAI_API_KEY = "sk-..."

# Or use setx for persistence (new sessions only)
setx OPENAI_API_KEY "sk-..."
```

### Manual Setting (Bash/Zsh)

```bash
export OPENAI_API_KEY="sk-..."

# Or add to ~/.bashrc / ~/.zshrc
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
```

---

---

## Provider Selection

### How DF3 selects a provider

For the **CLI** (`scripts/analyze_image.py` / `scripts/evaluate_llms.py`), the provider is selected via:

- `--provider` (`openai` or `openrouter`)
- `--base-url` (optional override; defaults to `https://openrouter.ai/api/v1` when `--provider openrouter`)

**Important:** the provider is **not inferred from the model name**. For example, `openai/gpt-5.1` is an OpenRouter model identifier, but you still need `--provider openrouter` (or an OpenRouter `--base-url`) to route requests to OpenRouter.

### OpenAI (default)

- Set `OPENAI_API_KEY`
- Use OpenAI model names like `gpt-5.1`

```powershell
python scripts/analyze_image.py --image photo.jpg --model gpt-5.1
```

### OpenRouter (multi-provider)

- Set `OPENROUTER_API_KEY`
- Use OpenRouter model IDs like `google/gemini-3-flash-preview`, `anthropic/claude-sonnet-4-20250514`, `openai/gpt-5.1`, etc.

```powershell
python scripts/analyze_image.py --image photo.jpg `
    --provider openrouter `
    --model google/gemini-3-flash-preview
```

### Optional OpenRouter headers

Some OpenRouter setups recommend passing identification headers:

- `--referer` → `HTTP-Referer`
- `--title` → `X-Title`

---

## Troubleshooting

### "API key not found"

**Cause**: Environment variable not set or `.env` not loaded.

**Fix**:

```powershell
# Verify variable is set
$env:OPENAI_API_KEY

# If empty, set it
$env:OPENAI_API_KEY = "sk-..."
```

### "Invalid API key"

**Cause**: Malformed or revoked key.

**Fix**:

1. Check key format (no extra spaces)
2. Verify key is active in provider dashboard
3. Generate new key if needed

### "Rate limit exceeded"

**Cause**: Too many API calls.

**Fix**:

1. Reduce `--num-workers`
2. Add delays between requests
3. Upgrade API plan

### "Model not found"

**Cause**: Model name incorrect or not available to your key.

**Fix**:

1. Check model name spelling
2. Verify model access in provider dashboard
3. Try alternative model

---

## Template .env File

Create `.env.example` for team reference:

```ini
# DF3 Environment Configuration
# Copy this file to .env and fill in your keys

# Required: At least one LLM provider
OPENAI_API_KEY=your-openai-key-here

# Optional: Alternative providers
# OPENROUTER_API_KEY=your-openrouter-key-here
# ANTHROPIC_API_KEY=your-anthropic-key-here
# GOOGLE_API_KEY=your-google-key-here

# Optional: Custom settings
# DF3_CACHE_DIR=.tool_cache
# DF3_LOG_LEVEL=INFO
```

---

## See Also

- [Configuration Guide](configuration.md) — Full configuration reference
- [Troubleshooting](troubleshooting.md) — Common issues
- [Installation](../getting-started/installation.md) — Setup guide
