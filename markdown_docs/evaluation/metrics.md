# Metrics Reference

Complete definitions of all evaluation metrics computed by DF3.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| N | Total samples |
| TP | True Positives (fake correctly identified as fake) |
| TN | True Negatives (real correctly identified as real) |
| FP | False Positives (real incorrectly flagged as fake) |
| FN | False Negatives (fake incorrectly passed as real) |
| N_fake | Total fake samples in dataset |
| N_real | Total real samples in dataset |
| answered | Samples with prediction in {real, fake} |
| abstain | Samples with prediction = uncertain |

---

## Core Metrics

### accuracy

**Overall accuracy including abstentions as wrong.**

$$\text{accuracy} = \frac{TP + TN}{N}$$

- Abstentions and errors count as incorrect
- Use for overall triage performance
- Lower bound on system capability

### accuracy_answered

**Accuracy on samples where the system provided an answer.**

$$\text{accuracy_answered} = \frac{TP + TN}{\text{answered}}$$

- Only considers real/fake predictions
- Measures decision quality when confident
- Higher than overall accuracy

### coverage

**Fraction of samples where system provided an answer.**

$$\text{coverage} = \frac{\text{answered}}{N}$$

- Range: 0 to 1
- Higher = more answers
- Trade-off with accuracy_answered

---

## Class-Specific Metrics

### Fake Class (Positive)

#### precision_fake

**When predicting "fake", how often correct?**

$$\text{precision_fake} = \frac{TP}{TP + FP}$$

#### recall_fake

**What fraction of fakes are caught?**

$$\text{recall_fake} = \frac{TP}{TP + FN}$$

#### f1_fake

**Harmonic mean of precision and recall.**

$$\text{f1_fake} = \frac{2 \times \text{precision_fake} \times \text{recall_fake}}{\text{precision_fake} + \text{recall_fake}}$$

### Real Class (Negative)

#### precision_real

**When predicting "real", how often correct?**

$$\text{precision_real} = \frac{TN}{TN + FN}$$

#### recall_real

**What fraction of reals are correctly passed?**

$$\text{recall_real} = \frac{TN}{TN + FP}$$

#### f1_real

**Harmonic mean for real class.**

$$\text{f1_real} = \frac{2 \times \text{precision_real} \times \text{recall_real}}{\text{precision_real} + \text{recall_real}}$$

---

## Balanced Metrics

### balanced_accuracy

**Average of class-specific true positive rates.**

$$\text{balanced_accuracy} = \frac{TPR_{fake} + TPR_{real}}{2}$$

Where:
- $TPR_{fake} = \text{recall_fake}$
- $TPR_{real} = \text{recall_real}$

Useful when classes are imbalanced.

### MCC (Matthews Correlation Coefficient)

**Balanced measure that accounts for all confusion matrix elements.**

$$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

- Range: -1 to +1
- 0 = random guessing
- +1 = perfect prediction
- -1 = perfect inverse prediction

---

## Triage Metrics

### fake_slip_rate

**Fraction of fakes that slip through as "real".**

$$\text{fake_slip_rate} = \frac{FN}{N_{fake}}$$

- Critical metric for fraud detection
- Lower is better
- Does NOT include abstentions (they go to review)

### real_false_flag_rate

**Fraction of reals incorrectly flagged as "fake".**

$$\text{real_false_flag_rate} = \frac{FP}{N_{real}}$$

- Measures false alarm rate
- Lower is better
- Impacts user trust

### fake_catch_rate

**Fraction of fakes correctly identified.**

$$\text{fake_catch_rate} = \frac{TP}{N_{fake}}$$

- Same as recall_fake
- Higher is better

### real_pass_rate

**Fraction of reals correctly passed.**

$$\text{real_pass_rate} = \frac{TN}{N_{real}}$$

- Same as recall_real
- Higher is better

---

## Abstention Metrics

### abstain_rate

**Overall rate of uncertain verdicts.**

$$\text{abstain_rate} = \frac{\text{abstain}}{N}$$

### abstain_rate_fake

**Abstention rate on fake samples.**

$$\text{abstain_rate_fake} = \frac{\text{abstain_fake}}{N_{fake}}$$

### abstain_rate_real

**Abstention rate on real samples.**

$$\text{abstain_rate_real} = \frac{\text{abstain_real}}{N_{real}}$$

### coverage_fake / coverage_real

**Class-conditional coverage.**

$$\text{coverage_fake} = 1 - \text{abstain_rate_fake}$$

---

## Calibration Metrics

### ECE (Expected Calibration Error)

**How well does confidence predict accuracy?**

$$ECE = \sum_{b=1}^{B} \frac{|B_b|}{N} \times |acc(B_b) - conf(B_b)|$$

Where:
- B = number of bins (default: 10)
- $B_b$ = samples in bin b
- $acc(B_b)$ = accuracy of samples in bin
- $conf(B_b)$ = average confidence in bin

Interpretation:

| ECE | Interpretation |
|-----|----------------|
| < 0.05 | Well calibrated |
| 0.05 - 0.15 | Moderately calibrated |
| > 0.15 | Poorly calibrated |

### Brier Score

**Mean squared error of probabilistic predictions.**

$$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - o_i)^2$$

Where:
- $p_i$ = predicted probability of correct class
- $o_i$ = 1 if correct, 0 if incorrect

---

## Latency Metrics

### avg_latency_seconds

**Average end-to-end latency per sample.**

### Timing Components

| Component | Description |
|-----------|-------------|
| vision_llm_seconds | Time for vision analysis |
| agent_graph_seconds | Time for agent reasoning + tools |
| total_seconds | End-to-end time |

### Percentiles

- **p50** — Median latency
- **p95** — 95th percentile latency

---

## Tool Usage Metrics

### avg_tool_count

**Average number of tools invoked per sample.**

### avg_tool_seconds_total

**Average total time spent in tools per sample.**

---

## Statistical Measures

### Wilson 95% CI

Confidence interval for proportions:

$$p \pm z \sqrt{\frac{p(1-p)}{n}}$$

With Wilson score correction for small samples.

### Standard Deviation (Multi-Trial)

For multi-trial evaluations:

$$\sigma = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n-1}}$$

---

## Metric Selection Guide

### For Overall Performance

- **accuracy_answered** — Quality of confident predictions
- **coverage** — How often system answers
- **balanced_accuracy** — Account for class imbalance

### For Safety/Risk

- **fake_slip_rate** — Fakes that escape detection
- **real_false_flag_rate** — Reals incorrectly flagged
- **abstain_rate** — How much goes to human review

### For Model Comparison

- **accuracy** — Overall ranking
- **f1_fake** — Fake detection capability
- **MCC** — Balanced comparison

### For Deployment Decisions

- **avg_latency_seconds** — System responsiveness
- **coverage** — Automation potential
- **accuracy_answered** — Trust in answers

---

## See Also

- [Methodology](methodology.md) — Evaluation framework
- [Benchmark Results](results.md) — Current results
- [Interpreting Results](../guide/interpreting-results.md) — Understanding outputs
