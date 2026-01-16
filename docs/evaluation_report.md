# DF3 Forensic Image Analysis — Evaluation / Validation Report

> **Intended audiences:** forensic science journal reviewers; forensic laboratory validation reviewers.  
> **Status:** Draft (reproducible metrics + explicit claim boundaries + validation caveats).  
> **Period:** January 2026.  
> **Artifacts used:** `results/*.jsonl`, `results/*.metrics.json` (derived metrics in `artifacts/eval_summary.json`).  
> **Reproduction:** `docs/evaluation_report.generated.md` is auto-generated from `results/*.jsonl` by `scripts/summarize_results.py`.

---

## 1) Scope, propositions, and intended use (triage, not autonomous adjudication)

### 1.1 Examination request (evaluation analogue)

This report evaluates DF3 as a **triage decision-support system** for forensic image authentication tasks: detecting whether an image is likely **authentic** vs **AI-generated/manipulated**, with a third option to **abstain** and route for manual review.

### 1.2 Propositions (explicit)

For each questioned image \(I\), we evaluate DF3 under the following propositions:

- **\(H_{real}\)**: \(I\) is an authentic camera-captured photograph with no material post-hoc manipulation relevant to the question posed.
- **\(H_{fake}\)**: \(I\) is not authentic, including either (a) AI-generated/synthetic imagery or (b) materially manipulated imagery (splicing/compositing/retouching) relevant to the question posed.

DF3 outputs one of: **REAL**, **FAKE**, or **UNCERTAIN** (insufficient/conflicting evidence).

### 1.3 Intended use and non-intended use

- **Intended use**: triage/routing + explanatory evidence gathering (tool outputs) to support a qualified examiner.
- **Non-intended use**: fully automated source attribution, individual identification, or court-ready “probability of authenticity” statements.

---

## 2) System description (what DF3 actually does)

DF3’s evaluation outputs are produced by `scripts/evaluate_llms.py` running `src/agents/forensic_agent.py:ForensicAgent.analyze()`.

### 2.1 Per-sample inference flow (what “tools” vs “no-tools” means)

For every image, DF3 runs a **vision step first** (even in tool mode), then optionally runs an **agentic tool-calling step**, then **structures** the final free-form output:

- **Vision step (always)**: BAML `AnalyzeImageVisionOnlyStructured` via `src/agents/baml_forensic.py:analyze_vision_only_structured_baml()`
  - Returns a structured dict including `verdict`, `confidence`, and `visual_description`.
- **Tool-augmented step (only when `use_tools=True`)**: LangGraph ReAct agent over forensic tools (`create_forensic_tools`) using the **vision description** as context.
- **Structuring step (only when `use_tools=True`)**: BAML `StructureForensicAnalysis` extracts the final structured verdict from the agent’s markdown response.

**Important:** In tool mode, the agent is intentionally prompted to **not** output JSON, and the final structured verdict comes from a separate structuring model call. This means “tools mode” measures *tool orchestration + evidence synthesis + structuring*, not just vision classification.

### 2.2 Label space and abstentions

DF3 is configured for **3-way triage**:

- `real`: “authentic photograph”
- `fake`: “AI-generated or manipulated”
- `uncertain`: abstain / route to human review

In the stored evaluation artifacts, abstention is represented as `prediction == "uncertain"` (or missing), and is treated as:

- **Not answered** for answered-only metrics (`accuracy_answered`, class PR/F1, MCC, balanced accuracy).
- **Not correct** for overall accuracy over all samples (`accuracy_overall`).

This is a *selective classification* (triage) setting, not a forced-binary benchmark.

---

## 3) Artifacts, schema, and traceability

### 3.1 `results/*.jsonl` (per-sample)

Each line is a single sample result with at least:

- identifiers: `id`, `image`, `label`
- run keys: `model`, `use_tools`, `trial`, `run_config`
- outputs: `prediction`, `confidence`, `rationale`
- timing: `latency_seconds` and often a nested `timings` dict (`vision_llm_seconds`, `agent_graph_seconds`, `total_seconds`)
- tool info (tools mode): `tool_usage`, `tool_details`, `tool_results`

### 3.2 `results/*.metrics.json` (aggregates)

Aggregated metrics are produced by `compute_metrics()` inside `scripts/evaluate_llms.py`. These match the per-sample recomputation in `scripts/summarize_results.py` (validated with 0 mismatches on core fields for the existing artifacts).

### 3.3 Provenance and auditability gaps in the current `results/` folder

Many of the `results/*.jsonl` files in this repo do **not** contain a `models` field (agent vs vision vs structuring model provenance). This means:

- If a run used a **vision-model override** (e.g., “Gemini vision + different agent model”), the artifact may not prove it.
- Any report that infers “vision model” from filenames should explicitly label that as **inference**, not ground truth.

**Fix forward (in code):** newer outputs can include `models` and prompt hashes. For strict lab validation / paper claims about specific model provenance, re-run evaluations with provenance persisted.

---

## 4) Vision vs agent model provenance (explicit mapping rules used in this report)

DF3’s **agent model** and **vision model** can differ. Many existing artifacts do not persist the `models` field, so the generated tables infer the vision model using the following **explicit rules** (documented to make assumptions transparent):

1. **Any `results/*.jsonl` filename starting with `G3...`** → vision model = `google/gemini-3-flash-preview`.
2. **`B_or_g3flashprev.jsonl`** → agent model = `google/gemini-3-flash-preview`; vision model = `google/gemini-3-flash-preview`.
3. **OpenAI agent models (`gpt-5-mini`, `gpt-5.2`)** → vision model = `gpt-5.2`.
4. **Sonnet 4.5 agent model** → vision model = `anthropic/claude-sonnet-4.5`.
5. **One explicit experiment**: `Glm46v-vision_agent_glm46.jsonl` → vision model = `z-ai/glm-4.6:nitro` (native GLM-4.6 vision).
6. **Default** (when none of the above apply): vision model = `google/gemini-3-flash-preview`.

These rules reflect the known run configuration in this project and are encoded in `scripts/summarize_results.py`. If you re-run evaluations with full provenance stored in `results/*.jsonl`, the inference rules are bypassed in favor of explicit per-sample `models.vision`.

---

## 5) Interpretation of “confidence” (avoid overstated probability language)

DF3 records a numeric `confidence` field. In the current implementation this value is **a model-generated self-report**, not a calibrated likelihood ratio or a validated probability of authenticity.

Accordingly:

- Use `confidence` as an **internal triage signal** (e.g., to prioritize manual review), not as a court-facing probabilistic statement.
- Calibration diagnostics (ECE/Brier/log loss on answered samples) are included to characterize reliability, but **do not** transform model self-reported confidence into a validated probabilistic measure of evidential strength.

---

## 6) Caching and why it matters for scientific validity and laboratory reproducibility

DF3 includes a tool+vision cache (`src/tools/forensic/cache.py`), enabled by default in `scripts/evaluate_llms.py`.

### 5.1 Latency confounding

Many stored artifacts show **vision-step timing < 250ms for ~100% of samples**, which is consistent with **cache hits** (not real model calls). Therefore:

- Latency comparisons in this repo are **not scientifically comparable** unless cache state is controlled and recorded.

### 5.2 Reproducibility confounding (the bigger issue)

If a cache key does not change when prompts change, old outputs can be silently reused after prompt edits.

**Fix forward (implemented):** vision-cache entries are now additionally keyed by a `cache_tag` derived from the hash of `baml_src/forensic_analysis.baml` (or `DF3_VISION_CACHE_TAG`), preventing silent reuse across prompt edits.

**Recommendation for journal/lab runs:**
- Run once with caching **disabled** (or empty cache) when reporting latency.
- Or record cache stats + cache tag in every artifact and treat cached latency separately.

---

## 7) Datasets, ground truth, and comparability constraints

The repo contains at least two distinct evaluated datasets (fingerprinted by sorted sample IDs):

- **Dataset A (n=500)**: ID digest `f987165daff0de70` (images predominantly under `data2/`; mix of `.jpeg/.png/.jpg`)
- **Dataset B (n=200)**: ID digest `1f78e35118013ed4`

**Scientifically valid rankings must not mix different datasets/sizes.**

**Ground truth note:** The evaluation treats labels as binary {real,fake}. In this project the dataset is **constructed from labeled synthetic datasets** (not random internet files), but the exact source list should be explicitly documented in the final submission. For laboratory validation, ground truth provenance should be auditable (e.g., dataset citation, generation logs, or source acquisition records).

---

## 8) Metrics (complete definitions for DF3’s triage/selective-classification setting)

Let \(N\) be total samples, and define:

- **Answered** = predictions in `{real,fake}` and no error.
- **Abstained** = prediction is `uncertain` (or missing), no error.
- **Errored** = `error` present.

### 8.1 Core DF3 metrics

- **Overall accuracy**: \((TP + TN) / N\) where abstentions+errors count as not correct.
- **Answered accuracy**: \((TP + TN) / N_{answered}\).
- **Coverage**: \(N_{answered}/N\).
- **Class PR/F1 (answered-only)**: computed on answered-only confusion matrix.
- **Balanced accuracy (answered-only)**: \((TPR_{fake} + TPR_{real})/2\).
- **MCC (answered-only)**:
  \[
  MCC = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
  \]
- **Triage rates (class-conditional over all items)**:
  - **Fake slip rate**: \(FN / N_{fake}\) (fake items incorrectly passed as real; abstentions are not slips)
  - **Real false-flag rate**: \(FP / N_{real}\) (real items incorrectly flagged as fake; abstentions are not false-flags)
  - **Fake catch rate**: \(TP / N_{fake}\)
  - **Real pass rate**: \(TN / N_{real}\)

### 8.2 Derived metrics included for paper readiness

Computed in `scripts/summarize_results.py` and reported in `docs/evaluation_report.generated.md`:

- **Wilson 95% CI** for `accuracy_overall` and `coverage` (binomial on \(N\)).
- **Calibration-ish diagnostics on answered samples**:
  - ECE (10 bins) on “confidence of predicted label” vs empirical accuracy.
  - Brier and log loss using \(p(fake)\) derived from confidence + predicted label.
- **Selective-risk proxy (AURC)**: area under risk–coverage curve induced by confidence-based rejection among answered items.
- **Latency percentiles**: p50/p95 for end-to-end `latency_seconds`.

---

## 9) Results (use the generated tables for complete coverage)

All numeric results should be sourced from the generated tables:

- **Primary table**: `docs/evaluation_report.generated.md` → “Key Metrics (per artifact/config)”
- **Machine-readable**: `artifacts/eval_summary.json`

### 9.1 Headline (dataset A, n=500)

- **Best vision-only (no-tools) overall triage performance**: `google/gemini-3-flash-preview|no-tools`
  - High accuracy and high coverage simultaneously (see generated table).
- **Best tool-augmented (tools) overall triage performance**: `google/gemini-3-flash-preview|tools`

### 9.2 Tools vs no-tools (paired on same samples)

Paired McNemar exact tests are reported in `docs/evaluation_report.generated.md` under “Paired Comparisons”.

Interpretation guidance:

- This test evaluates correctness on *all* samples, where abstentions count as incorrect (triage framing).
- It does **not** answer: “which system is better at the same review rate?” For that, use risk–coverage analysis and compare at matched coverage/abstention.

### 9.3 Tool usage analysis (descriptive only)

The generated tables include a **Tool Usage Analysis** section summarizing:

- which tools were invoked most frequently,
- conditional accuracy/confidence when a given tool was used vs not used.

**Important:** these are **descriptive** correlations. Tool selection is not random and is confounded by image difficulty; therefore these statistics must **not** be interpreted as causal “tool improves accuracy” claims.

---

## 10) Limitations (journal + lab validation)

- **Single trial per configuration** (no run-to-run variance estimates).
- **Different datasets / sample sizes** (n=200 vs n=500) prevent direct cross-ranking.
- **Provenance gaps** in existing artifacts (`models` missing) limit claims about “vision model used” for some runs.
- **Caching confounds**: latency is not comparable unless cache state is controlled; and without prompt-versioned caching, outputs may be non-reproducible across prompt edits (fixed forward).
- **Dataset stratification missing**: no breakdown by manipulation type, compression level, source, etc.
- **Human factors / cognitive bias controls not evaluated**: DF3 prompts include procedural guidance, but this report does not validate human-in-the-loop workflows, contextual bias mitigation, or independent review processes.

---

## 11) Recommendations

1. **Persist full provenance per sample**: agent/vision/structuring model IDs, prompt hashes, cache tag, cache hit/miss flags.
2. **Disable caching for latency studies**, or report cache-hit vs cache-miss latencies separately.
3. **Run multi-trial evaluations** (e.g., 3–10) to estimate variance; keep temperature at 0 for determinism unless explicitly studying stochasticity.
4. **Add forced-binary baselines** (no abstain) and compare to triage mode explicitly.
5. **Stratify results** by metadata: format, resolution, compression, manipulation type, source model family, and difficulty bins.
6. **Define and document operational policies**: what triggers manual review, what tool outputs are retained, and how conflicting evidence is resolved by examiners.
7. **Independent verification**: implement second-examiner review / blind verification procedures when used in casework-like contexts.

---

## Appendix: Reproduction and supporting artifacts

Generate the paper tables from the committed artifacts:

```bash
python scripts/summarize_results.py --results-dir results --out artifacts/eval_summary.json --out-md docs/evaluation_report.generated.md
```

