# Research Findings

---

## Summary

DF3 demonstrates that **LLM-based forensic analysis can provide explainable, auditable image authentication** suitable for triage workflows and human-in-the-loop forensic processes. While raw accuracy metrics favor traditional ML approaches, the system's ability to articulate reasoning in natural language represents a distinct capability valuable for forensic applications.

**Key Findings**:

1. Vision-capable LLMs can detect AI-generated images with reasonable accuracy (up to 92.8%), but this performance may be confounded by training data overlap
2. Adding forensic tools did not improve detection accuracy in our evaluation, likely due to tool-task mismatch and the challenge of zero-shot tool interpretation
3. The primary value proposition is **explainability**: LLM-generated reasoning provides transparent, auditable analysis suitable for forensic reporting
4. The system enables effective **triage workflows** where uncertain cases route to human experts

---

## Research Context

### The Two-Phase Approach: DF2 and DF3

This research builds on **DF2**, a traditional machine learning classifier developed in the first phase:

| Aspect | DF2 (Random Forest/Logistic) | DF3 (LLM Agent) |
|--------|------------------------------|-----------------|
| Architecture | Logistic regression on extracted features | Vision LLM + LangGraph ReAct agent |
| Training | 166,000+ images | No task-specific training |
| Features | 21 scalar forensic features | Same tools, different interface |
| Primary output | Probability score P(fake) | Verdict + natural language reasoning |
| Accuracy (AUROC) | 0.9996 | Not directly comparable |
| Explainability | Template-based evidence citing | Free-form reasoning with evidence |

**Critical insight from DF2**: The forensic tools (Noiseprint, DRUNet residuals, frequency analysis) **do provide discriminative signal**. DF2's ablation study demonstrates:
- Removing Noiseprint: AUROC drops from 0.9966 to 0.9777 (-1.89%)
- Removing frequency features: AUROC drops to 0.9829 (-1.37%)

The tools work. The question is whether LLMs can interpret them effectively without training.

### Research Questions

1. **Can LLMs provide explainable forensic image analysis?**
2. **Can LLMs effectively interpret forensic tool outputs without task-specific training?**
3. **What is the appropriate role for LLM-based detection in forensic workflows?**

---

## Experimental Design

### Dataset

| Property | Value |
|----------|-------|
| Total samples | 500 |
| Class balance | 247 fake (49.4%), 253 real (50.6%) |
| Fake sources | GenImage, DRAGON, Nano-banana-150k |
| Real sources | ImageNet (via GenImage), Nano-banana |
| Fake type | Primarily AI-generated (not manipulated) |

### Evaluation Configurations

1. **Vision-only**: LLM analyzes image directly, no forensic tools
2. **Tool-augmented**: LLM uses initial vision analysis, then invokes forensic tools via LangGraph ReAct agent, synthesizes findings

### Models Evaluated

- Google Gemini 3 Flash Preview
- OpenAI GPT-5.2, GPT-5-mini
- Anthropic Claude Sonnet 4.5
- Zhipu AI GLM-4.6, GLM-4.7
- DeepSeek, Grok, Kimi, MIMO

---

## Results

### Quantitative Performance

#### Vision-Only Mode (n=500)

| Model | Accuracy | Coverage | Acc(answered) | Abstention |
|-------|----------|----------|---------------|------------|
| Gemini 3 Flash Preview | 0.928 | 0.984 | 0.943 | 1.6% |
| GLM-4.7 | 0.914 | 0.980 | 0.933 | 2.0% |
| GPT-5.2 | 0.074 | 0.078 | 0.949 | 92.2% |

#### Tool-Augmented Mode (n=500)

| Model | Accuracy | Coverage | Acc(answered) | Abstention |
|-------|----------|----------|---------------|------------|
| Gemini 3 Flash Preview | 0.782 | 0.934 | 0.837 | 6.2% |
| GLM-4.7 | 0.456 | 0.508 | 0.898 | 49.0% |
| GPT-5.2 | 0.080 | 0.112 | 0.714 | 88.8% |

### Critical Caveats on Performance Numbers

#### Caveat 1: Training Data Contamination

The high vision-only accuracy (92.8% for Gemini 3 Flash) must be interpreted cautiously:

- **GenImage**, **DRAGON**, and **Nano-banana** are publicly available datasets
- Modern vision LLMs may have been trained on these or similar synthetic images
- If a model has "seen" these images (or images from the same generators) during training, it may recognize them without possessing generalizable detection capability

**Implication**: Vision-only accuracy on this dataset does not necessarily indicate robust detection capability on novel generators or unseen image distributions.

#### Caveat 2: Tool-Task Mismatch

The forensic tools available to DF3 were designed primarily for **manipulation detection** (splicing, copy-move, inpainting), not **AI-generation detection**:

| Tool | Primary Design Purpose | Signal for AI-Generated |
|------|------------------------|-------------------------|
| TruFor | Manipulation/forgery detection | Weak (outputs low scores) |
| ELA | JPEG editing detection | Weak (no editing occurred) |
| JPEG quantization | Double-compression detection | Neutral |
| Frequency analysis | Upsampling/resampling artifacts | Moderate |
| Residual extraction | Noise pattern anomalies | Moderate |

AI-generated images are internally consistent—they were never "edited." Tools designed to detect editing artifacts find little signal. This is not a failure of the LLM; it's a mismatch between available tools and the detection task.

#### Caveat 3: Zero-Shot Tool Interpretation

DF2 achieved 99.96% AUROC because it **learned** how to combine forensic features from 166,000 labeled examples. The logistic regression discovered decision boundaries like:

```
If noiseprint_std > X AND fft_low_energy > Y → likely fake
```

DF3 asks the LLM to discover these patterns from first principles, with only prompt instructions. This is fundamentally harder:

- DF2: Supervised learning on 166k examples
- DF3: Zero-shot reasoning with tool descriptions

The LLM must interpret numeric outputs (e.g., "TruFor: 0.14") without having learned what constitutes a meaningful threshold for this specific dataset.

---

## The Core Value Proposition: Explainability

### Why Explainability Matters for Forensics

In forensic contexts, a probability score is often insufficient. Investigators, attorneys, and courts need to understand:

1. **What evidence** supports the conclusion
2. **What analysis** was performed
3. **What limitations** apply to the finding
4. **What alternative explanations** were considered

LLMs provide this naturally through their reasoning output.

### Example: DF2 vs DF3 Output Comparison

**DF2 Output** (template-based):
```
Prediction: FAKE (confidence: 0.94)

Evidence:
- Noiseprint mismatch: 3.21 (threshold: 2.50) - SUPPORTS FAKE
- Residual energy p95: 0.087 (elevated) - SUPPORTS FAKE
- FFT peakiness: 1.42 (normal range) - NEUTRAL

Uncertainty: LOW (Mahalanobis distance within expected range)
```

**DF3 Output** (LLM reasoning):
```
### Visual Analysis
This portrait shows a young woman with long dark hair against a neutral
background. Several synthesis indicators are present:

- The hair strands on the left shoulder dissolve into the background
  rather than possessing physical depth
- The skin texture is hyper-smooth, lacking natural pores or
  micro-imperfections typical of photographs
- The transition between ear and hair shows unnatural blending

### Tool Evidence
- TruFor manipulation_probability: 0.14 (low)
- ELA anomaly_score: 1.2 (within normal range)

### Interpretation
The low manipulation scores are expected for AI-generated images,
as they are internally consistent and show no editing artifacts.
The visual anomalies—particularly the impossible hair physics and
unnaturally smooth skin—are characteristic of diffusion model outputs.

### Verdict: FAKE
### Confidence: 0.85

The image exhibits multiple visual hallmarks of AI generation. While
forensic tools show no manipulation (expected for synthetic images),
the anatomical and textural anomalies strongly indicate synthesis.
```

The DF3 output is suitable for:
- Forensic reports that will be reviewed by non-technical stakeholders
- Legal proceedings where reasoning must be articulated
- Training materials for forensic examiners
- Cases where the conclusion may be challenged

### Transparency of Reasoning

Unlike black-box classifiers, LLM reasoning can be:

1. **Audited**: Reviewers can check if the reasoning is sound
2. **Challenged**: Errors in reasoning can be identified and corrected
3. **Documented**: The analysis becomes part of the case record
4. **Educational**: Shows what indicators are relevant for detection

---

## Practical Use Cases

### Use Case 1: High-Volume Triage

**Scenario**: A forensic agency receives 100,000 images requiring authentication assessment.

**Workflow**:
```
100,000 images
    ↓
DF3 Vision-Only Analysis (fast, ~5s/image)
    ↓
├── High confidence REAL (60%) → Auto-clear
├── High confidence FAKE (25%) → Auto-flag for review
└── UNCERTAIN (15%) → Human expert review
```

**Value**:
- 85% of images handled automatically with reasoning documented
- Human experts focus on genuinely ambiguous cases
- Each decision includes explanation for audit trail

### Use Case 2: Forensic Report Generation

**Scenario**: An examiner needs to document their analysis of a suspected deepfake.

**Workflow**:
```
Suspicious image
    ↓
DF3 Analysis (tools + vision)
    ↓
Generate SWGDE-compliant report
    ↓
Expert reviews/validates reasoning
    ↓
Final report for case file
```

**Value**:
- Structured analysis following forensic standards
- Natural language suitable for reports
- Expert retains final judgment authority

### Use Case 3: Educational and Training

**Scenario**: Training new forensic examiners on AI-generated image detection.

**Workflow**:
```
Training images (known ground truth)
    ↓
DF3 Analysis with detailed reasoning
    ↓
Compare LLM reasoning to expert analysis
    ↓
Identify what indicators matter
```

**Value**:
- Demonstrates analytical reasoning process
- Shows what visual/forensic cues are relevant
- Provides worked examples for learning

### Use Case 4: Preliminary Assessment

**Scenario**: Quick assessment needed before committing to full forensic analysis.

**Workflow**:
```
Image of interest
    ↓
DF3 Vision-Only (2-10 seconds)
    ↓
├── Clear indicators → Preliminary finding documented
└── Ambiguous → Full forensic workup warranted
```

**Value**:
- Fast initial assessment
- Documented reasoning even for preliminary findings
- Helps prioritize resource allocation

---

## Insights for Future Research

### Insight 1: Tools Need Task-Specific Design

DF2 proved that forensic features can achieve near-perfect detection. But those features were designed for the same distribution they were evaluated on. For LLM agents to benefit from tools:

- **Develop AI-generation-specific tools**: Detectors trained specifically on diffusion/GAN outputs
- **Provide threshold guidance**: Include calibrated thresholds in tool descriptions
- **Return confidence intervals**: Help LLMs understand measurement uncertainty

### Insight 2: Few-Shot May Beat Zero-Shot

DF3's tool-augmented mode asks LLMs to interpret forensic outputs zero-shot. Consider:

- **Few-shot examples**: Show 3-5 examples of correct tool interpretation in the prompt
- **Calibration data**: Provide distribution statistics ("typical real images score 0.1-0.3")
- **Decision rules**: Extract DF2's learned rules and provide them explicitly

### Insight 3: Hybrid Architectures

The optimal system may combine DF2 and DF3 approaches:

```
Image
    ↓
DF2: Extract features → Trained classifier → P(fake)
    ↓
DF3: Vision analysis → Integrate DF2 score as "tool" → Reasoning
    ↓
Output: Probability + Explanation
```

This leverages DF2's discriminative power with DF3's explanatory capability.

### Insight 4: Evaluation on Novel Distributions

Future evaluation should use:
- Images from generators not in training data
- Temporal holdout (new models released after data collection)
- Cross-domain evaluation (different content types)

This avoids the contamination concern that affects our vision-only results.

### Insight 5: The Three-Way Classification Has Value

The UNCERTAIN verdict is a feature:

| Traditional Binary | DF3 Three-Way |
|-------------------|---------------|
| Force decision on ambiguous cases | Route ambiguous cases to humans |
| Hide uncertainty in low confidence | Make uncertainty explicit |
| Same output for "confident wrong" and "uncertain" | Distinguish confident vs uncertain |

For forensic applications, knowing when the system is uncertain is as valuable as knowing its prediction.

---

## Comparison: DF2 vs DF3

| Dimension | DF2 | DF3 |
|-----------|-----|-----|
| **Accuracy** | Higher (AUROC 0.9996) | Lower (varies by model) |
| **Explainability** | Template-based | Free-form reasoning |
| **Generalization** | Limited to training distribution | Unknown (contamination concern) |
| **Speed** | Fast (<1s) | Slower (5-90s) |
| **Human oversight** | Score requires interpretation | Reasoning is human-readable |
| **Adaptability** | Requires retraining | Prompt engineering |
| **Tool integration** | Fixed feature set | Dynamic tool selection |
| **Uncertainty handling** | Calibrated probability | Three-way classification |

**Recommendation**: Use DF2 for high-throughput automated screening. Use DF3 when explainability is required or for human-in-the-loop workflows.

---

## Limitations of This Research

### Single Dataset
All evaluation used one dataset (subset and combination of GenImage + DRAGON + Nano-banana + more for df2). Results may not generalize to other distributions.

### Single Trial
Most configurations were run once or twice. No variance estimates are available.

### Training Data Contamination Unknown
We cannot verify whether evaluated models were trained on our test images or similar data, because open-data models are rare.

### Tool-Task Mismatch
Forensic tools optimized for manipulation detection were applied to AI-generation detection.

### No Baseline Comparison
We did not compare against purpose-built AI-generated image detectors (e.g., DIRE, UnivFD).

---

## Conclusions

DF3 demonstrates that LLM-based forensic image analysis is technically feasible and provides unique value through explainability. The system is best understood not as a replacement for traditional detectors, but as a complementary capability for workflows requiring:

1. **Auditable reasoning** that can be reviewed and challenged
2. **Natural language output** suitable for reports and legal proceedings
3. **Human-in-the-loop triage** where uncertain cases are escalated
4. **Flexible analysis** that can be adapted through prompting

The finding that tools did not improve accuracy should not be interpreted as "tools are useless", DF2 proves they provide signal. Rather, it indicates that **zero-shot tool interpretation by LLMs is a deeper problem** that warrants further research.

Future work should focus on:
- Developing AI-generation-specific forensic tools
- Exploring few-shot tool interpretation
- Hybrid architectures combining trained classifiers with LLM reasoning
- Rigorous evaluation on temporally held-out, novel generators

---

## Artifacts and Reproducibility

### Result Files
- `results/G3visiononly_notools.jsonl` — Best vision-only results
- `results/B_or_g3flashprev.jsonl` — Best tool-augmented results
- Additional per-model results in `results/`

### Reproduction Commands

```powershell
# Vision-only
python scripts/evaluate_llms.py --dataset data2/samples.jsonl \
    --models google/gemini-3-flash-preview --tools no --limit 500

# Tool-augmented
python scripts/evaluate_llms.py --dataset data2/samples.jsonl \
    --models google/gemini-3-flash-preview --tools yes --limit 500
```

### Dataset Provenance
- **Digest (n=500)**: `f987165daff0de70`
- **Sources**: GenImage, DRAGON, Nano-banana-150k

---

## See Also

- [Evaluation Results](../evaluation/results.md) — Complete benchmark tables
- [Methodology](../evaluation/methodology.md) — Evaluation framework
- [Design Decisions](design-decisions.md) — Architectural choices
- [Limitations](limitations.md) — Known system limitations
