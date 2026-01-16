# SWGDE Alignment

How DF3 aligns with SWGDE (Scientific Working Group on Digital Evidence) best practices for image authentication.

---

## Overview

SWGDE establishes best practices for digital forensics. DF3's reporting and methodology are designed to align with the "Best Practices for Image Authentication" (2022) document.

---

## Core Principles Mapping

### Documentation

| SWGDE Requirement | DF3 Implementation |
|-------------------|-------------------|
| Examiner information | Report header |
| Evidence identification | Image path, hash, metadata |
| Methodology description | Tool usage log, prompts |
| Findings | Structured verdict + rationale |
| Conclusions | REAL/FAKE/UNCERTAIN |
| Limitations | Stated in report footer |

### Reproducibility

| SWGDE Requirement | DF3 Implementation |
|-------------------|-------------------|
| Tool versions | Model identifiers in results |
| Parameters | Temperature, max_iterations recorded |
| Intermediate results | Tool outputs preserved in raw results |

### Objectivity

| SWGDE Requirement | DF3 Implementation |
|-------------------|-------------------|
| Multiple techniques | Vision LLM + forensic tools |
| Corroborating evidence | Cross-reference in agent reasoning |
| Acknowledge uncertainty | UNCERTAIN verdict |
| Evidence-based reasoning | Prompts enforce factual grounding |

---

## Report Structure

DF3 generates reports with the following structure:

### 1. Header

```markdown
# Image Authentication Report
**Analysis Date**: 2026-01-15
**File**: sample-001.jpg
```

### 2. Evidence Information

```markdown
## Evidence
- **SHA-256**: abc123...
- **File Size**: 1.2 MB
- **Dimensions**: 1920x1080
```

### 3. Methodology

```markdown
## Methodology
- Vision Model: gpt-5.1
- Agent Model: gpt-5.1
- Tools: TruFor, ELA, Metadata
- Mode: Agentic (tools enabled)
```

### 4. Observations

```markdown
## Observations
### Visual Analysis
[Description of image content and anomalies]

### Tool Results
- TruFor: manipulation_probability = 0.85
- ELA: anomaly_score = 3.2
- Metadata: No C2PA; EXIF stripped
```

### 5. Interpretation

```markdown
## Interpretation
High TruFor score combined with ELA anomalies 
suggests localized editing. Absent metadata is 
consistent with post-processing.
```

### 6. Limitations

```markdown
## Limitations
- Confidence is model self-report, not calibrated probability
- AI-generated images may not trigger manipulation detectors
- Heavy compression affects tool accuracy
```

### 7. Conclusion

```markdown
## Conclusion
**Verdict**: FAKE
**Confidence**: 0.85
```

---

## Key Distinctions

### Manipulation vs. Synthetic Detection

| Task | Description | DF3 Capability |
|------|-------------|----------------|
| Manipulation Detection | Has image been edited? | Strong (TruFor, ELA) |
| Synthetic Detection | Is image AI-generated? | Variable (visual analysis) |
| Authentication | Is image what it claims? | Contextual |

DF3 addresses both content (visual) and structure (tools) as SWGDE recommends.

---

## Confidence Score Usage

The `confidence` score is the LLM's self-reported certainty.

**Appropriate uses:**

- Triage ranking
- Prioritizing review queue
- Identifying uncertain cases

**Not appropriate for:**

- Probability statements in reports
- Statistical likelihood ratios

---

## External References

- [SWGDE Publications](https://www.swgde.org/documents/published)
- [Best Practices for Image Authentication (2022)](https://www.swgde.org/documents/published)

---

## See Also

- [Interpreting Results](../guide/interpreting-results.md)
- [Limitations](../research/limitations.md)
