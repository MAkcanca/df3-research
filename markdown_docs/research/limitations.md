# Limitations

This page documents known limitations of the DF3 system based on empirical observations during evaluation.

---

## Detection Limitations

### Manipulation vs. Synthesis

| Scenario | Detection Capability | Reason |
|----------|---------------------|--------|
| **Post-hoc manipulation** (splicing, inpainting) | Strong | TruFor/ELA designed for this |
| **Fully AI-generated images** | Variable | Relies on visual artifact recognition |
| **Adversarial examples** | Weak | Not evaluated systematically |

TruFor's `manipulation_probability` near 0 indicates absence of *editing*, not authenticity. AI-generated images can score low because they were never manipulated—they were created whole.

### Visual Artifact Dependence

Vision-only mode depends on recognizable generation artifacts:

- Anatomical errors (extra fingers, asymmetric features)
- Texture inconsistencies (overly smooth skin, repeating patterns)
- Semantic impossibilities (reflections that don't match)
- Lighting/shadow inconsistencies

Images without obvious artifacts may pass undetected.

### Format-Specific Tools

| Tool | Format Requirement | Behavior on Other Formats |
|------|-------------------|--------------------------|
| ELA | JPEG only | Skipped |
| JPEG Quantization | JPEG only | Skipped |
| Frequency Analysis | Any | Results vary by format |

---

## Model Behavior

### High Abstention Rates

Some models exhibit very high abstention (UNCERTAIN) rates:

| Model | Abstention Rate | Notes |
|-------|-----------------|-------|
| GPT-5.2 | 88-92% | Even with `reasoning_effort: high` |
| GPT-5-mini | 70-92% | Depends on tools mode |
| Gemini 3 Flash | 1.6-6% | Much lower abstention |

High abstention produces low overall accuracy but can have high accuracy-when-answered.

### Non-determinism

With `temperature=0`, outputs are near-deterministic but not guaranteed identical across:

- Different API calls
- Different model versions/checkpoints
- Provider-side updates

### Prompt Sensitivity

The agent's behavior depends on prompt wording. The current prompts are tuned for the evaluation workflow. Significant prompt changes may alter performance characteristics.

---

## Tool Limitations

### TruFor

- Trained on manipulation datasets; may not generalize to novel generators
- Computationally expensive (~2-5s per image on GPU)
- Weights auto-download on first use (~500MB)

### ELA

- Only meaningful for JPEG images
- Heavy recompression can produce false positives
- Legitimate watermarks/overlays appear as anomalies

### Metadata

- Easily stripped or forged
- Absence of metadata is not evidence of manipulation
- C2PA adoption is limited; most images lack it

### Residual Extraction

- DRUNet inference adds latency
- Statistical properties overlap between real and synthetic classes
- No hard thresholds exist

---

## Evaluation Limitations

### Single Dataset

All results are from one evaluation dataset with different sample limits (n=200 or n=500). Results may not generalize to:

- Different image sources
- Different generator versions
- Different content domains

### Single Trial

Most runs are single-trial. No variance estimates are available for metrics.

### Confounded Tool Comparisons

Tool usage correlates with image difficulty. "Tool used" vs. "not used" accuracy deltas are **descriptive, not causal**.

### Cache Effects

Some latency measurements are confounded by cache state. Vision cache and tool cache can reduce latency to near-zero.

---

## Confidence Score

The `confidence` field is the LLM's self-reported certainty.

**Properties:**

- Range: 0.0 to 1.0
- Not calibrated to empirical accuracy
- May exhibit overconfidence
- Varies by model

**Use for:**

- Triage ranking (higher → more certain)
- Routing low-confidence to human review

**Do not use for:**

- Probability statements
- Likelihood ratios

---

## Operational Constraints

### Latency

| Mode | Typical Range |
|------|---------------|
| Vision-only | 2-10s |
| Tools mode | 30-90s |
| Cached | <1s |

### API Dependencies

- Cloud LLM analysis requires internet and valid API keys
- Rate limits apply per provider
- Provider outages affect availability

### GPU Recommendation

TruFor and DRUNet run on CPU but are significantly faster on CUDA-capable GPU.

---

## See Also

- [Benchmark Results](../evaluation/results.md) — Empirical performance data
- [Methodology](../evaluation/methodology.md) — Evaluation setup
