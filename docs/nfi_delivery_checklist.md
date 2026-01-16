# DF3 Research Delivery Checklist

This checklist is designed for a forensic laboratory / research unit handoff where reviewers prioritize **traceability**, **reproducibility**, **claim boundaries**, and **quality assurance**.

---

## 1) What to deliver (minimum package)

- **Code + commit hash**
  - Provide a repository snapshot (or archive) with the exact commit used for the reported results.
- **Evaluation artifacts**
  - `results/*.jsonl` (per-sample outputs)
  - `results/*.metrics.json` (aggregated metrics)
  - `artifacts/eval_summary.json` (derived metrics; generated)
  - `docs/evaluation_report.md` (narrative)
  - `docs/evaluation_report.generated.md` (tables; generated)
- **Datasets (or dataset manifest)**
  - Prefer providing a *manifest* with stable identifiers if images cannot be shared.
  - At minimum, provide dataset JSONL(s) used (`data2/samples.jsonl`, `data/*.jsonl`) and a description of label provenance.
  - Document the dataset sources (the project uses labeled synthetic datasets; list exact sources in the handoff package).

---

## 2) Reproducibility (exact commands)

### 2.1 Rebuild derived tables from existing `results/`

```bash
python scripts/summarize_results.py --results-dir results --out artifacts/eval_summary.json --out-md docs/evaluation_report.generated.md
```

### 2.2 Re-run evaluation (requires dataset images + API access)

Key script: `scripts/evaluate_llms.py`

**Important knobs to document for any rerun:**
- `--models` (agent model)
- `--vision-model` (vision step model override, if any)
- `--structuring-model` (BAML structuring override, if any)
- `--tools` (`tools`, `no-tools`, or `both`)
- `--temperature` (should be `0.0` for determinism)
- `--max-iterations` (tool budget)
- caching flags (`--disable-tool-cache` recommended for latency studies)

---

## 3) Audit trail / provenance (what the lab will ask for)

For each per-sample record in `results/*.jsonl`, ensure the following are present for *future* runs:

- **Model provenance**: agent vs vision vs structuring models (a `models` field).
- **Prompt provenance**: prompt hashes and prompt versioning.
- **Cache provenance**:
  - cache enabled/disabled
  - cache directory + stats
  - cache tag / prompt version discriminator
  - ideally per-sample cache hit/miss flags (not currently logged; recommended)

**Why:** without these, a reviewer cannot distinguish a real model call from a cache hit, or know whether prompt edits changed outputs.

---

## 4) “Confidence” and reporting language (avoid overclaiming)

DF3 stores a numeric `confidence` in outputs. For forensic reporting:

- Treat this as **model self-reported confidence**, not a calibrated probability or likelihood ratio.
- Do **not** express results as “\(P(\mathrm{real})\)” without a validated calibration and a defensible probabilistic framework.
- Prefer a triage framing:
  - “DF3 flagged as FAKE / REAL / UNCERTAIN under the project’s operating definitions”
  - “Recommended for manual review” (for uncertain or high-risk cases)

---

## 5) Validation expectations (what to plan next)

To be publication- and lab-ready, plan a follow-on validation that includes:

- **Multi-trial runs** for variance (same dataset; fixed prompts; cache controlled).
- **Stratified performance** by:
  - file format, resolution, compression level
  - manipulation type (splice, copy-move, retouch, full synthetic)
  - image source/device provenance where available
- **Independent review / bias mitigation plan** for any human-in-the-loop use.

---

## 6) Practical notes for evaluation

- Keep a clean separation between:
  - **research evaluation** (benchmark numbers),
  - **casework-like operation** (controlled inputs, documented workflow, independent review).
- Provide a short “Known limitations” section that explicitly includes:
  - tool limitations for fully synthetic images,
  - limitations under heavy compression / low resolution,
  - provenance uncertainty when metadata is absent or altered.

