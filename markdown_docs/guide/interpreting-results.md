# Interpreting Results

How to read and understand DF3 analysis outputs.

---

## Verdict

| Verdict | Meaning | Typical Action |
|---------|---------|----------------|
| **REAL** | No manipulation/generation indicators | Accept |
| **FAKE** | Manipulation or AI-generation detected | Flag for review |
| **UNCERTAIN** | Insufficient or conflicting evidence | Human review |

### What FAKE Covers

- AI-generated images (DALL-E, Midjourney, etc.)
- Manipulated images (splicing, inpainting)
- Deepfakes (face swaps)

The rationale often indicates which type was detected.

---

## Confidence Score

| Range | Interpretation |
|-------|----------------|
| 0.8 - 1.0 | High confidence |
| 0.6 - 0.8 | Medium confidence |
| 0.4 - 0.6 | Low confidence |
| 0.0 - 0.4 | Very low confidence |

The confidence score is the LLM's self-assessment. Use it for relative ranking, not as a probability.

---

## Tool Scores

### TruFor

`manipulation_probability` (0.0 - 1.0):

| Range | Interpretation |
|-------|----------------|
| 0.0 - 0.2 | Low manipulation evidence |
| 0.2 - 0.5 | Some indicators |
| 0.5 - 0.8 | Moderate evidence |
| 0.8 - 1.0 | Strong manipulation evidence |

**Note:** Low TruFor ≠ authentic. AI-generated images score low because they weren't *edited*.

### ELA

`ela_anomaly_score` (z-score):

| Range | Interpretation |
|-------|----------------|
| 0.0 - 1.5 | Normal |
| 1.5 - 2.5 | Slight anomalies |
| 2.5 - 4.0 | Moderate anomalies |
| 4.0+ | Strong anomalies |

ELA detects regions with different compression. High scores suggest localized edits.

### JPEG Quality

| Quality | Interpretation |
|---------|----------------|
| 90-100 | High quality, minimal compression |
| 70-90 | Good quality |
| 50-70 | Moderate compression |
| < 50 | Heavy compression |

### Frequency Analysis

Frequency domain results are corroborating evidence:
- Periodic peaks may indicate AI-generated regularities
- Unusual distributions can suggest processing

### Residual Analysis

Noise statistics from DRUNet:
- Very low residuals may indicate AI generation
- Unusual distributions can suggest processing

---

## Common Scenarios

### High Visual Confidence, Low TruFor Score

**Likely interpretation:** AI-generated image

TruFor detects editing, not generation. AI images are internally consistent.

### Low Visual Confidence, High TruFor Score

**Likely interpretation:** Subtle manipulation

TruFor detects things humans miss.

### Everything Uncertain

**Interpretation:** Genuine ambiguity

- Edge case
- New generation technique
- Unusual but legitimate image

---

## Reading the Rationale

A good rationale includes:

1. Visual observations
2. Tool evidence
3. How evidence was weighed
4. Why the verdict was chosen

**Strong example:**

> The image shows a portrait with anatomical anomalies: 6 fingers on left hand, uniformly smooth skin without visible pores. TruFor returned low manipulation probability (0.12), consistent with AI-generation rather than editing. Visual synthesis artifacts with absence of manipulation signatures supports verdict of FAKE (AI-generated).

---

## Batch Analysis

For dataset evaluations, focus on:

- **Accuracy when answered**: Quality of committed predictions
- **Coverage**: How often it gives an answer vs uncertain
- **Fake slip rate**: Fakes incorrectly passed as real
- **Real false flag rate**: Reals incorrectly flagged as fake

---

## See Also

- [Tools Overview](../tools/overview.md)
- [Metrics Reference](../evaluation/metrics.md)
- [SWGDE Best Practices](../reference/swgde.md)
