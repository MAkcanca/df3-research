# Design Decisions

This document captures the key architectural and design decisions made during DF3 development, including alternatives considered, rationale for choices made, and lessons learned from approaches that were tried and abandoned.

---

## Overview

DF3 evolved through several iterations. Understanding *why* the system is designed the way it is helps future researchers build on this work without repeating unsuccessful approaches.

---

## Decision 1: Agent Framework (LangGraph)

### Choice Made
**LangGraph** with the ReAct agent pattern for tool orchestration.

### Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **LangGraph ReAct** | Built-in tool calling, state management, checkpointing | Heavier dependency, learning curve |
| **LangChain AgentExecutor** | Simpler API, well-documented | Less flexible, deprecated patterns |
| **Custom loop** | Full control, minimal dependencies | More code to maintain, reinventing wheels |
| **Autogen/CrewAI** | Multi-agent support | Overkill for single-agent task |

### Rationale

LangGraph was selected because:

1. **Native tool calling**: Handles the ReAct loop (Reason → Act → Observe) cleanly
2. **State management**: Built-in checkpointing via MemorySaver
3. **Iteration control**: `recursion_limit` parameter prevents infinite loops
4. **Future flexibility**: Could extend to multi-agent workflows if needed

### Lessons Learned

The ReAct pattern works well for tool orchestration but introduces failure modes:
- Tool selection errors accumulate over iterations
- Long reasoning chains can drift from the task
- Agent may call unnecessary tools, adding latency

**Recommendation**: For simpler tasks, a single-shot approach may be preferable. Use agents only when dynamic tool selection is genuinely needed.

---

## Decision 2: Structured Output (BAML)

### Choice Made
**BAML** (Boundary AI Markup Language) for structured output extraction, using a **two-step approach**:
1. Free-form reasoning (no structure constraints)
2. Separate structuring call to extract JSON

### Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **BAML two-step** | Preserves reasoning quality, type-safe outputs | Two LLM calls, added latency |
| **OpenAI JSON mode** | Single call, native support | Reasoning degradation when constrained |
| **Pydantic + instructor** | Type-safe, popular | Still constrains during generation |
| **Regex extraction** | No extra LLM call | Fragile, fails on format variations |

### Rationale

Research shows that requiring structured output *during* reasoning can degrade LLM performance ("reasoning degradation"). By separating reasoning from structuring:

1. The reasoning phase uses full model capabilities
2. The structuring phase is a simpler extraction task
3. Both outputs are available (raw text + structured)

### Implementation

```python
# Step 1: Free reasoning
vision_output = await AnalyzeImageVisionOnly(image)  # Returns markdown

# Step 2: Structure extraction
structured = await StructureForensicAnalysis(vision_output)  # Returns typed object
```

### Lessons Learned

The two-step approach works well:
- Reasoning quality is preserved
- Structuring rarely fails (it's just extraction)
- Cost is acceptable (structuring can use cheaper model)

**Recommendation**: Use this pattern for any task where both reasoning quality and structured output matter.

---

## Decision 3: Vision Model Separation

### Choice Made
Allow **separate models** for vision, agent reasoning, and structuring:
- `vision_model`: Initial image analysis
- `llm_model`: Agent reasoning and tool orchestration
- `structuring_model`: BAML output extraction

### Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **Single model for all** | Simpler configuration | Forces trade-offs (cost vs. capability) |
| **Separate models** | Optimize each stage independently | More complex configuration |

### Rationale

Different stages have different requirements:

| Stage | Requirement | Optimal Model |
|-------|-------------|---------------|
| Vision | Strong visual understanding | Gemini, GPT-4o, Claude |
| Agent | Reasoning, tool use | Can be cheaper |
| Structuring | JSON extraction | Can be cheapest |

This allows cost optimization: use a capable vision model, then cheaper models for downstream tasks.

### Lessons Learned

In practice, **vision model choice dominates performance**:
- Gemini 3 Flash as vision model: 92.8% accuracy
- GPT-5.2 as vision model: 7.4% accuracy (with 92% abstention)

The agent and structuring models matter less. Future work should focus on vision model selection.

---

## Decision 4: Tool Architecture

### Choice Made
**Modular tool design** with each tool in a separate file, registered via `@tool` decorator.

### Tool Selection

| Tool | Included | Rationale |
|------|----------|-----------|
| TruFor | Yes | State-of-the-art manipulation detection |
| ELA | Yes | Classic forensic technique, interpretable |
| JPEG analysis | Yes | Compression artifact detection |
| Frequency analysis | Yes | DCT/FFT for pattern detection |
| Residual extraction | Yes | DRUNet noise patterns |
| Metadata | Yes | EXIF/C2PA, no ML required |
| Code execution | Yes | Flexibility for custom analysis |
| CFA | Disabled | Unreliable results, high false positive rate |

### Why CFA Was Disabled

Color Filter Array (CFA) analysis detects inconsistencies in Bayer pattern interpolation. In testing:
- High false positive rate on legitimate images
- Inconsistent results across image formats
- Limited discriminative power for AI-generated images

Rather than produce misleading evidence, we disabled it.

### Lessons Learned

**Tool selection should match the task**. Our tools were designed for manipulation detection, but our dataset was AI-generated images. This mismatch is a primary cause of the "tools hurt performance" finding.

**Recommendation**: Develop or integrate tools specifically designed for AI-generated detection:
- GAN fingerprint detectors
- Diffusion model artifact detectors
- Semantic consistency analyzers

---

## Decision 5: Prompt Engineering Strategy

### Choice Made
**Explicit prompts** with detailed instructions, including:
- Task definition
- Visual inspection guidelines
- Tool interpretation rules
- Common pitfall warnings
- Output format specification

### Evolution of Prompts

The prompts evolved significantly:

#### Version 1 (Early)
```
Analyze this image and determine if it is real or fake.
```

**Problem**: No guidance on what to look for, inconsistent outputs.

#### Version 2 (Mid)
```
Look for: anatomical errors, texture anomalies, lighting issues...
Output: JSON with verdict, confidence, rationale
```

**Problem**: Models conflated "no manipulation" with "real".

#### Version 3 (Final)
```
IMPORTANT DISTINCTION:
- "No evidence of manipulation" is NOT the same as "not synthetic"
- AI-generated images can score low on manipulation tools

When to trust tools vs. visual analysis...
```

**Key insight**: Explicit warnings about common errors are necessary. LLMs will make logical mistakes (e.g., "TruFor low → image is real") unless explicitly instructed not to.

### What Didn't Work

1. **Numeric thresholds**: Telling the model "TruFor > 0.8 means fake" led to overreliance on arbitrary cutoffs
2. **Confidence calibration instructions**: Asking for "calibrated" confidence didn't improve calibration
3. **Chain-of-thought forcing**: Requiring explicit reasoning steps didn't improve accuracy

### What Worked

1. **Distinction between manipulation and synthesis**: Explicit, repeated
2. **SWGDE best practices inclusion**: Added authoritative guidance
3. **Uncertain as valid output**: Explicitly permitting abstention reduced false confidence
4. **Concrete visual indicators**: Lists of what to look for (fingers, skin texture, etc.)

---

## Decision 6: Caching Strategy

### Choice Made
**Multi-level caching** with separate caches for:
- Vision outputs (keyed by image hash + model + prompt hash)
- Tool outputs (keyed by image hash + tool + parameters)
- Image encoding (LRU cache, 32 images)

### Rationale

Forensic tools are computationally expensive:
- TruFor: 2-5 seconds per image (GPU)
- Residual extraction: 1-3 seconds per image
- Vision LLM call: 3-10 seconds

For batch evaluation, caching eliminates redundant computation when:
- Re-running with different agent models (vision cache)
- Re-running with different configurations (tool cache)
- Processing the same image multiple times (encoding cache)

### Trade-offs

| Pro | Con |
|-----|-----|
| Faster iteration | Cache invalidation complexity |
| Reproducible results | Disk space usage |
| Cost savings (fewer LLM calls) | Stale results if prompts change |

### Implementation Detail

```python
# Cache keyed by content hash, not path
image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
cache_key = f"{tool_name}_{image_hash}_{param_hash}"
```

---

## Decision 7: Three-Way Classification

### Choice Made
**Three-way output**: `real`, `fake`, `uncertain` (instead of binary classification).

### Rationale

Forensic contexts require appropriate uncertainty:

| Scenario | Binary System | Three-Way System |
|----------|---------------|------------------|
| Clear fake | fake | fake |
| Clear real | real | real |
| Ambiguous | **forced guess** | uncertain → human review |

Forced binary classification on ambiguous images produces errors. Allowing "uncertain" enables:
- Higher accuracy on answered samples
- Intelligent triage to human reviewers
- Honest uncertainty communication

### Lessons Learned

Abstention rates vary dramatically by model:
- Gemini 3 Flash: 1.6% abstention
- GPT-5.2: 92% abstention

The three-way system exposed model-specific behaviors that would be hidden in binary classification.

---

## Decision 8: Evaluation Metrics

### Choice Made
**Selective classification metrics** that treat abstention appropriately:
- `accuracy`: Overall (abstentions count as wrong)
- `accuracy_answered`: Among answered samples only
- `coverage`: Fraction of samples answered

### Why Not Just Accuracy?

Binary accuracy conflates two types of errors:
1. Wrong answers (false positives, false negatives)
2. Non-answers (abstentions)

A system that abstains 99% of the time and gets 1% right has 1% accuracy but 100% accuracy-when-answered. Both numbers are meaningful for different decisions.

### Metrics Selected

| Metric | Purpose |
|--------|---------|
| accuracy | Overall system performance |
| accuracy_answered | Quality when confident |
| coverage | System usefulness |
| MCC | Balanced measure for imbalanced data |
| fake_slip_rate | Fakes incorrectly passed |
| real_false_flag_rate | Reals incorrectly flagged |

---

## Approaches Tried and Abandoned

### Approach 1: Confidence-Based Tool Selection

**Idea**: Only invoke tools when vision confidence is below threshold.

**Implementation**:
```python
if vision_confidence < 0.7:
    run_tools()
```

**Result**: No improvement. Low-confidence cases are often hard cases where tools also struggle.

**Lesson**: Confidence doesn't predict when tools will help.

### Approach 2: Tool Output Summarization

**Idea**: Summarize verbose tool outputs before passing to agent.

**Implementation**: LLM call to condense tool JSON to 2-3 sentences.

**Result**: Added latency, minimal accuracy change. The summarization sometimes lost critical details.

**Lesson**: Let the agent see raw outputs; it can extract what it needs.

### Approach 3: Ensemble Multiple Models

**Idea**: Run multiple vision models, vote on verdict.

**Implementation**: 3 models, majority vote.

**Result**: Slower, more expensive, marginal accuracy improvement. Best single model was nearly as good.

**Lesson**: Model selection > model ensembling for this task.

### Approach 4: Fine-Grained Verdict

**Idea**: Distinguish "AI-generated" from "manipulated" in the verdict.

**Implementation**: Four-way: `real`, `ai_generated`, `manipulated`, `uncertain`.

**Result**: Models struggled to distinguish generation from manipulation. Added complexity without improving usefulness.

**Lesson**: Keep verdicts simple. The rationale can provide nuance.

---

## Recommendations for Future Development

Based on these experiences:

1. **Start simple**: Vision-only is a strong baseline. Add complexity only if it helps.

2. **Match tools to task**: If detecting AI-generated images, use AI-generation detectors, not manipulation detectors.

3. **Test models individually**: Don't assume results transfer. Evaluate each model on your specific task.

4. **Preserve raw outputs**: Cache and log everything. You'll want to reanalyze later.

5. **Explicit > implicit**: When LLMs make logical errors, add explicit instructions to prevent them.

6. **Measure what matters**: Abstention rates, latency, and cost matter as much as accuracy.

---

## See Also

- [Research Findings](findings.md) — Empirical results and analysis
- [Architecture Overview](../architecture/overview.md) — System design
- [Agent Pipeline](../architecture/agent-pipeline.md) — Implementation details
