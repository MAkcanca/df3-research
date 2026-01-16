# Troubleshooting

---

## Installation

### "ModuleNotFoundError: No module named 'X'"

Ensure venv is activated and dependencies installed:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### "baml-cli: command not found"

```powershell
pip install baml-py
baml-cli generate
```

### TruFor weights download fails

1. Check internet connection
2. Verify disk space (~500MB needed)
3. Manual download to `weights/trufor/`

---

## API

### "API key not found"

```powershell
# Check
$env:OPENAI_API_KEY

# Set
$env:OPENAI_API_KEY = "sk-..."
```

### "Rate limit exceeded"

Reduce parallel workers:

```powershell
python scripts/evaluate_llms.py --num-workers 1
```

### "Model not found"

Verify model name spelling and availability for your API key.

---

## Analysis

### "Image file not found"

```powershell
# Verify path
Test-Path "path/to/image.jpg"

# Use absolute path
python scripts/analyze_image.py --image "C:\full\path\image.jpg"
```

### Analysis hangs

Possible causes:
- Large image → resize or use smaller test image
- Slow model → try `gpt-5-mini`
- Network issues → check connection

### "CUDA out of memory"

Force CPU mode:

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
```

---

## Evaluation

### Results show all errors

1. Verify API key
2. Test single image first: `python scripts/analyze_image.py --image test.jpg`
3. Check error messages in results JSONL

### Metrics show NaN

- Check raw results for errors
- Ensure dataset has both classes
- Verify model is producing verdicts

---

## BAML

### "BAML generation failed"

```powershell
baml-cli generate --verbose
```

Check syntax in `baml_src/*.baml`.

### Type mismatch in output

1. Verify BAML function definition
2. Ensure prompt includes `{{ ctx.output_format }}`
3. Try different temperature

---

## Tool-Specific

### ELA returns "skipped"

Expected for non-JPEG images. ELA only works on JPEG.

### TruFor returns unexpected values

```powershell
# Re-download weights
Remove-Item -Recurse weights/trufor
# Weights auto-download on next run
```

---

## Performance

| Issue | Solution |
|-------|----------|
| Slow analysis | Resize images, use GPU, enable cache |
| High memory | Reduce `--num-workers` |
| High latency | Use faster model, enable tool cache |

---

## Debug Mode

```powershell
$env:DF3_LOG_LEVEL = "DEBUG"
python scripts/analyze_image.py --image photo.jpg
```

---

## See Also

- [Installation](../getting-started/installation.md)
- [Environment Variables](environment.md)
- [FAQ](faq.md)
