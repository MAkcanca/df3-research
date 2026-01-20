# Benchmark Results (Complete)

This page summarizes **all evaluated configurations present in `results/`**, derived from per-sample outputs (`results/*.jsonl`) and computed summaries (`artifacts/eval_summary.json`).


!!! warning "Vision model vs agent model"
    DF3 uses a *vision model* (for the initial vision step) and an *agent model* (for tool orchestration and synthesis). Some runs use a different vision model than the agent model.
    
    Where the per-sample `models.vision` field is missing, vision provenance is inferred from the filename conventions and run policy (documented in `artifacts/eval_summary.json`).

!!! warning "Latency can be confounded by cache"
    Some runs have near-zero vision latency due to cache hits. Treat latency comparisons as valid only when cache state is controlled.

---

## Example dataset sample (image + model output)

Below is a **single real dataset example** from a tools-enabled run, showing the input image, ground-truth label, and the model's full narrative output.

![Dataset example: sample-281](../assets/examples/sample-281.jpeg)

**Sample metadata**

- `id`: `sample-281`
- `image`: `data2/sample-281.jpeg`
- `label` (ground truth): `real`
- `model`: `google/gemini-3-flash-preview` (tools mode)
- `prediction`: `real`
- `confidence`: `0.95`
- `tools used`: `metadata`, `perform_trufor`, `perform_ela`

**Structured output (parsed)**

```json
{
  "verdict": "real",
  "confidence": 0.95,
  "rationale": "Low manipulation probabilities (TruFor 0.143, ELA below threshold), consistent lighting, textures, and physical details (wood grain, rivets, stain) support authenticity. The visible iStockphoto watermark explains missing original metadata; low resolution is a noted limitation but does not outweigh the forensic indicators.",
  "visual_description": "A wine cellar/storage area with several oak barrels; two barrels rest on white metal racks in the foreground with a reddish wine stain on one. Additional barrels are stacked in the background. Soft directional indoor lighting; an iStockphoto watermark is overlaid across the center.",
  "forensic_summary": "Tools used: Metadata analysis (EXIF/C2PA), TruFor manipulation score, Error Level Analysis (ELA), and image specification inspection."
}
```

??? example "Full model output (raw_analysis)"

    ### Observations
    The image depicts a wine cellar or storage area with several oak barrels. Two barrels are in the foreground, resting on white metal racks. The wood grain, metal hoops, and rivets appear physically consistent. A reddish wine stain is visible on the top of one barrel, which is a common real-world occurrence. In the background, other barrels are stacked vertically and horizontally. An "iStockphoto" watermark is prominently overlaid across the center. The lighting is soft and directional, consistent with indoor cellar lighting, and shadows align with the objects.

    ### Tool Outputs
    *   **Metadata**: No EXIF data or C2PA manifests found. Contains an sRGB ICC profile.
    *   **TruFor**: Manipulation probability of **0.143**, which is low, suggesting no significant splicing or localized editing.
    *   **ELA**: Anomaly score of **1.867**. This is below the typical threshold of 2.0 for localized anomalies, indicating uniform compression levels across the image.
    *   **Image Specs**: Small resolution (380x252), which is typical for thumbnails or web previews.

    ### Interpretation
    The visual evidence is highly consistent with an authentic photograph. The textures of the wood, the specific wear on the metal hoops, and the background depth of field appear natural. While the "iStockphoto" watermark is technically an "addition" to the original capture, it is a standard commercial overlay and not a "forgery" in the forensic sense. The low TruFor and ELA scores suggest the image structure is consistent and hasn't undergone malicious manipulation. There are no anatomical or physical "AI artifacts" (like garbled wood grain or impossible geometry) visible even upon close inspection.

    ### Limitations
    *   **Low Resolution**: The small image size (380x252) can mask fine-grained AI artifacts or subtle editing traces.
    *   **Watermark**: The presence of a watermark confirms the image has been processed/saved by a third party (iStock), which explains the lack of original camera metadata.
    *   **Synthetic Quality**: High-quality AI can occasionally produce simple scenes like this without obvious errors, though the specific "stock photo" look and consistent physics point toward a real source.

    ### Conclusion
    The image shows no signs of AI generation or forensic manipulation. The presence of a known commercial watermark and the consistent physical details (stains, rivets, wood grain) strongly support its authenticity as a standard stock photograph.

    **Verdict: real**

    **Confidence (0-1): 0.95**

## Key findings (high level)

- **Synthetic mixed dataset (n=500, digest `f987165daff0de70`)**: highest overall triage performance from Gemini 3 Flash Preview in vision-only mode (high accuracy with high coverage).
- **Synthetic mixed dataset (n=200, digest `1f78e35118013ed4`)**: same dataset pool with limit=200; evaluated only in tools mode for available runs.
- **FaceForensics++ frames dataset (n=500, digest `c02071eee1ee544a`)**: deepfake-oriented benchmark derived from FF++ video frames (PNG). Results are not directly comparable to the synthetic mix due to domain shift and heavy class imbalance (444 fake / 56 real in this sample).
- **Abstention dominates some configurations**: for some models, "accuracy when answered" can be high even though overall accuracy is low due to abstaining most of the time.
- **Tool usage statistics are descriptive**: differences between "tool used" vs "not used" are confounded by difficulty and are **not causal**.

---

## Complete model inventory (all configs found)

The tables below include **every model/configuration found in `results/*.jsonl`**, including agent model, vision model, dataset, and mode.

## Complete Model Inventory (derived from results/*)

This table is generated from `artifacts/eval_summary.json` (itself computed from `results/*.jsonl`).

### Sample limit: n=200 (digest `1f78e35118013ed4`)

| Agent Model | Vision Model | Mode | JSONL | n | Acc | Cov | Acc(ans) | MCC(ans) | Abstain | FakeSlip | RealFalseFlag | Lat_p50(s) | Lat_p95(s) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xiaomi/mimo-v2-flash:free | google/gemini-3-flash-preview | tools | G3vision_agent_mimov2flash.jsonl | 200 | 0.650 | 0.860 | 0.756 | 0.558 | 0.140 | 0.408 | 0.020 | 44.373 | 72.020 |
| x-ai/grok-4.1-fast | google/gemini-3-flash-preview | tools | G3vision_agent_grok41fast.jsonl | 200 | 0.505 | 0.605 | 0.835 | 0.611 | 0.390 | 0.112 | 0.088 | 40.101 | 55.719 |
| anthropic/claude-sonnet-4.5 | anthropic/claude-sonnet-4.5 | tools | B_or_sonnet45.jsonl | 200 | 0.475 | 0.600 | 0.792 | 0.578 | 0.400 | 0.112 | 0.137 | 54.979 | 70.626 |
| deepseek/deepseek-v3.2 | google/gemini-3-flash-preview | tools | G3vision_agent_deepseekv32.jsonl | 200 | 0.470 | 0.605 | 0.777 | 0.555 | 0.390 | 0.184 | 0.088 | 59.138 | 289.982 |
| moonshotai/kimi-k2-thinking:nitro | google/gemini-3-flash-preview | tools | G3vision_agent_kimi_k2.jsonl | 200 | 0.295 | 0.355 | 0.831 | 0.652 | 0.630 | 0.051 | 0.069 | 32.391 | 52.559 |

### FaceForensics++ frames: n=500 (digest `c02071eee1ee544a`)

These runs correspond to the `_del` artifacts in `results/` and evaluate three models in **tools mode** on FF++ frames.

| Agent Model | Vision Model | Mode | JSONL | n | Acc | Cov | Acc(ans) | MCC(ans) | Abstain | FakeSlip | RealFalseFlag | Lat_p50(s) | Lat_p95(s) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| z-ai/glm-4.6:nitro | z-ai/glm-4.6v:nitro | tools | Glm_del.jsonl | 500 | 0.504 | 0.600 | 0.840 | 0.084 | 0.398 | 0.056 | 0.411 | 28.984 | 43.728 |
| moonshotai/kimi-k2-thinking:nitro | z-ai/glm-4.6v:nitro | tools | Kimi_k2_del.jsonl | 500 | 0.416 | 0.484 | 0.860 | -0.005 | 0.510 | 0.020 | 0.446 | 36.205 | 63.647 |
| xiaomi/mimo-v2-flash:free | z-ai/glm-4.6v:nitro | tools | Mimo_del.jsonl | 500 | 0.190 | 0.596 | 0.319 | -0.050 | 0.402 | 0.432 | 0.196 | 32.273 | 52.175 |

### Sample limit: n=500 (digest `f987165daff0de70`)

| Agent Model | Vision Model | Mode | JSONL | n | Acc | Cov | Acc(ans) | MCC(ans) | Abstain | FakeSlip | RealFalseFlag | Lat_p50(s) | Lat_p95(s) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| google/gemini-3-flash-preview | google/gemini-3-flash-preview | no-tools | G3visiononly_notools.jsonl | 500 | 0.928 | 0.984 | 0.943 | 0.890 | 0.016 | 0.101 | 0.012 | 6.049 | 8.349 |
| z-ai/glm-4.7:nitro | google/gemini-3-flash-preview | no-tools | G3vision_agent_glm47.jsonl | 500 | 0.914 | 0.980 | 0.933 | 0.869 | 0.020 | 0.117 | 0.016 | 0.029 | 0.039 |
| z-ai/glm-4.6:nitro | z-ai/glm-4.6:nitro | no-tools | Glm46v-vision_agent_glm46.jsonl | 500 | 0.418 | 0.636 | 0.657 | 0.222 | 0.364 | 0.441 | 0.000 | 0.012 | 0.026 |
| gpt-5-mini | gpt-5.2 | no-tools | A_openai_gpt5mini_v52.jsonl | 500 | 0.074 | 0.078 | 0.949 | 0.000 | 0.922 | 0.000 | 0.008 | 0.027 | 0.038 |
| gpt-5.2 | gpt-5.2 | no-tools | A_openai_gpt52_visiononly.jsonl | 500 | 0.074 | 0.078 | 0.949 | 0.000 | 0.922 | 0.000 | 0.008 | 0.037 | 0.048 |
| google/gemini-3-flash-preview | google/gemini-3-flash-preview | tools | B_or_g3flashprev.jsonl | 500 | 0.782 | 0.934 | 0.837 | 0.674 | 0.062 | 0.166 | 0.138 | 33.320 | 47.370 |
| z-ai/glm-4.7:nitro | google/gemini-3-flash-preview | tools | G3vision_agent_glm47.jsonl | 500 | 0.456 | 0.508 | 0.898 | 0.793 | 0.490 | 0.040 | 0.063 | 34.514 | 55.146 |
| z-ai/glm-4.6:nitro | z-ai/glm-4.6:nitro | tools | Glm46v-vision_agent_glm46.jsonl | 500 | 0.448 | 0.750 | 0.597 | 0.142 | 0.232 | 0.587 | 0.024 | 32.253 | 49.041 |
| gpt-5-mini | gpt-5.2 | tools | A_openai_gpt5mini_v52.jsonl | 500 | 0.240 | 0.294 | 0.816 | 0.659 | 0.700 | 0.008 | 0.099 | 88.935 | 128.741 |
| gpt-5.2 | gpt-5.2 | tools | A_openai_gpt52_e2e_tools.jsonl | 500 | 0.080 | 0.112 | 0.714 | 0.332 | 0.888 | 0.000 | 0.063 | 35.935 | 57.543 |

## Tool usage (all tools-runs; descriptive, non-causal)

_Tool selection is confounded by image difficulty; do not interpret deltas as causal._

| Tool | Calls | UsedRate | Acc(ans)Used | Acc(ans)NotUsed | DeltaAcc(ans) |
|---|---:|---:|---:|---:|---:|
| perform_trufor | 2760 | 0.789 | 0.827 | 0.664 | 0.164 |
| metadata | 2229 | 0.637 | 0.842 | 0.694 | 0.148 |
| perform_ela | 2066 | 0.590 | 0.823 | 0.746 | 0.077 |
| extract_residuals | 928 | 0.265 | 0.830 | 0.760 | 0.069 |
| execute_python_code | 298 | 0.085 | 0.808 | 0.774 | 0.034 |
| detect_jpeg_quantization | 45 | 0.013 | 0.800 | 0.778 | 0.022 |
| analyze_frequency_domain | 29 | 0.008 | 0.500 | 0.779 | -0.279 |
| analyze_jpeg_compression | 14 | 0.004 | 1.000 | 0.778 | 0.222 |

---

## Reproducibility

To recompute the derived JSON summary from `results/`:

```powershell
python scripts/summarize_results.py --results-dir results --out artifacts/eval_summary.json --out-md markdown_docs/evaluation_report.generated.md
```

---

## See also

- [Methodology](methodology.md)
- [Metrics reference](metrics.md)
- [Dataset provenance](dataset-provenance.md)
