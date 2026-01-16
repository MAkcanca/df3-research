#!/usr/bin/env python3
"""
Summarize DF3 evaluation artifacts in results/ into a compact, paper-friendly form.

Why this exists:
- `results/*.jsonl` lines can be extremely large (raw markdown, prompts, tool outputs).
- The existing `*.metrics.json` files contain useful metrics but omit many derived metrics
  needed for a rigorous report (latency percentiles, calibration-ish scores, selective-risk curves, etc.).

This script parses `results/*.jsonl` and (optionally) validates against `results/*.metrics.json`,
then emits:
- A compact JSON summary (machine-readable; reproducible).
- Optional markdown tables suitable for inclusion in docs/papers.

Run:
  python scripts/summarize_results.py --results-dir results --out artifacts/eval_summary.json
  python scripts/summarize_results.py --results-dir results --out artifacts/eval_summary.json --out-md docs/evaluation_report.generated.md
"""

from __future__ import annotations

import argparse
import json
import math
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _percentiles(xs: List[float], ps: Iterable[float]) -> Dict[str, float]:
    if not xs:
        return {}
    arr = np.asarray(xs, dtype=float)
    out: Dict[str, float] = {}
    for p in ps:
        out[f"p{int(round(100*p))}"] = float(np.percentile(arr, 100 * p))
    out["mean"] = float(arr.mean())
    out["median"] = float(np.median(arr))
    return out


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score CI for a proportion."""
    if n <= 0:
        return 0.0, 0.0
    p_hat = float(p_hat)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p_hat + z2 / (2.0 * n)
    rad = z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
    lo = (center - rad) / denom
    hi = (center + rad) / denom
    return float(lo), float(hi)


def mcnemar_exact(n10: int, n01: int) -> float:
    """
    Exact two-sided McNemar test p-value using Binomial(b, 0.5) on discordant pairs.
    """
    b = n10 + n01
    if b == 0:
        return 1.0
    k = min(n10, n01)
    p = 0.0
    for i in range(0, k + 1):
        p += math.comb(b, i) * (0.5**b)
    return float(min(1.0, 2.0 * p))


@dataclass(frozen=True)
class ConfigKey:
    model: str
    mode: str  # "tools" | "no-tools"


def _mode_from_use_tools(use_tools: bool) -> str:
    return "tools" if bool(use_tools) else "no-tools"


def _confusion_from_answered(
    rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], int, int]:
    """
    Compute confusion matrix on answered rows only (pred in {real,fake}), excluding errors.

    Returns:
      (confusion, support_by_label, abstain_by_label, abstain_count, error_count)
    """
    tp = tn = fp = fn = 0
    support = {"real": 0, "fake": 0}
    abstain_by = {"real": 0, "fake": 0}
    abstain = 0
    errors = 0
    for r in rows:
        gold = (r.get("label") or "").strip().lower()
        pred = r.get("prediction")
        if gold in support:
            support[gold] += 1
        if r.get("error"):
            errors += 1
            continue
        if pred is None or pred == "uncertain":
            abstain += 1
            if gold in abstain_by:
                abstain_by[gold] += 1
            continue
        pred = str(pred).strip().lower()
        if gold == "fake" and pred == "fake":
            tp += 1
        elif gold == "real" and pred == "real":
            tn += 1
        elif gold == "real" and pred == "fake":
            fp += 1
        elif gold == "fake" and pred == "real":
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}, support, abstain_by, abstain, errors


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return float(prec), float(rec), float(f1)


def _mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Matthews Correlation Coefficient on answered-only confusion matrix.
    """
    num = tp * tn - fp * fn
    a = tp + fp
    b = tp + fn
    c = tn + fp
    d = tn + fn
    if a == 0 or b == 0 or c == 0 or d == 0:
        return 0.0
    den = math.sqrt(a * b * c * d)
    if den == 0:
        return 0.0
    return float(num / den)


def _ece_from_pred_conf(conf: List[float], correct: List[int], n_bins: int = 10) -> float:
    """
    ECE for 'confidence of predicted label' vs empirical accuracy within bins.
    """
    if not conf:
        return 0.0
    c = np.asarray(conf, dtype=float)
    y = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (c >= lo) & (c < hi if i < n_bins - 1 else c <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        acc = float(y[mask].mean())
        avg_conf = float(c[mask].mean())
        ece += (n / len(c)) * abs(acc - avg_conf)
    return float(ece)


def _brier_and_logloss(
    p_fake: List[float],
    y_fake: List[int],
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """
    Brier and log loss for binary labels, using p(fake).
    """
    if not p_fake:
        return 0.0, 0.0
    p = np.asarray(p_fake, dtype=float)
    y = np.asarray(y_fake, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    brier = float(np.mean((p - y) ** 2))
    ll = float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))
    return brier, ll


def _mean_conf_answered(rows: List[Dict[str, Any]]) -> float:
    vals: List[float] = []
    for r in rows:
        if r.get("error"):
            continue
        pred = r.get("prediction")
        if pred not in ("real", "fake"):
            continue
        c = _safe_float(r.get("confidence"))
        if c is None:
            continue
        vals.append(_clamp(c, 0.0, 1.0))
    return float(np.mean(vals)) if vals else 0.0


def _basic_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    confusion, support, abstain_by, abstain_count, error_count = _confusion_from_answered(rows)
    tp, tn, fp, fn = confusion["tp"], confusion["tn"], confusion["fp"], confusion["fn"]
    correct = tp + tn
    answered = total - abstain_count - error_count
    accuracy = correct / total if total else 0.0
    accuracy_answered = correct / answered if answered else 0.0
    coverage = answered / total if total else 0.0
    abstain_rate = abstain_count / total if total else 0.0
    error_rate = error_count / total if total else 0.0
    return {
        "n_total": int(total),
        "n_answered": int(answered),
        "n_abstain": int(abstain_count),
        "n_error": int(error_count),
        "accuracy_overall": float(accuracy),
        "accuracy_answered": float(accuracy_answered),
        "coverage": float(coverage),
        "abstain_rate": float(abstain_rate),
        "error_rate": float(error_rate),
        "mean_conf_answered": _mean_conf_answered(rows),
        "confusion": confusion,
        "support": support,
    }


def _aurc_from_confidence(conf: List[float], correct: List[int]) -> Dict[str, float]:
    """
    Compute a simple risk-coverage curve and AURC by rejecting the lowest-confidence answered items.

    Notes:
    - Uses the model's self-reported confidence on answered items.
    - This is *not* AUROC; it measures how well confidence can triage errors.
    """
    if not conf:
        return {"aurc": 0.0}
    c = np.asarray(conf, dtype=float)
    ok = np.asarray(correct, dtype=int)
    order = np.argsort(c)  # low confidence first
    ok_sorted = ok[order]
    n = len(ok_sorted)
    # As we reject k lowest-confidence, keep remaining n-k items.
    risks = []
    coverages = []
    for k in range(0, n + 1):
        remain = ok_sorted[k:]
        cov = len(remain) / n if n else 0.0
        if len(remain) == 0:
            risk = 0.0
        else:
            risk = 1.0 - float(remain.mean())
        coverages.append(cov)
        risks.append(risk)
    # Trapezoid integration over coverage in [0,1]
    aurc = 0.0
    for i in range(1, len(coverages)):
        dx = coverages[i - 1] - coverages[i]  # decreasing coverage as k increases
        aurc += dx * (risks[i - 1] + risks[i]) / 2.0
    return {"aurc": float(aurc)}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _index_by_config(rows: List[Dict[str, Any]]) -> Dict[ConfigKey, List[Dict[str, Any]]]:
    out: Dict[ConfigKey, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        model = r.get("model")
        use_tools = r.get("use_tools")
        if not isinstance(model, str) or use_tools is None:
            continue
        key = ConfigKey(model=model, mode=_mode_from_use_tools(bool(use_tools)))
        out[key].append(r)
    return dict(out)


def _extract_answered_confidence_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract confidence + correctness arrays for answered rows (no errors).
    """
    conf: List[float] = []
    correct: List[int] = []
    p_fake: List[float] = []
    y_fake: List[int] = []
    for r in rows:
        if r.get("error"):
            continue
        pred = r.get("prediction")
        if pred is None or pred == "uncertain":
            continue
        gold = (r.get("label") or "").strip().lower()
        pred = str(pred).strip().lower()
        c = _safe_float(r.get("confidence"))
        if c is None:
            continue
        c = _clamp(c, 0.0, 1.0)
        is_ok = 1 if pred == gold else 0
        conf.append(c)
        correct.append(is_ok)
        # Convert "confidence of predicted label" into p(fake)
        pf = c if pred == "fake" else (1.0 - c)
        pf = _clamp(pf, 0.0, 1.0)
        p_fake.append(pf)
        y_fake.append(1 if gold == "fake" else 0)

    ece = _ece_from_pred_conf(conf, correct, n_bins=10)
    brier, logloss = _brier_and_logloss(p_fake, y_fake)
    aurc = _aurc_from_confidence(conf, correct)
    return {
        "answered_confidence": _percentiles(conf, ps=[0.05, 0.1, 0.25, 0.75, 0.9, 0.95]),
        "ece_predlabel_10bins": float(ece),
        "brier_p_fake": float(brier),
        "logloss_p_fake": float(logloss),
        **aurc,
    }


def _timing_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    lat: List[float] = []
    v: List[float] = []
    a: List[float] = []
    tot: List[float] = []
    tool_sum: List[float] = []
    tool_count: List[int] = []
    vision_fast: int = 0
    vision_seen: int = 0

    for r in rows:
        ls = _safe_float(r.get("latency_seconds"))
        if ls is not None:
            lat.append(ls)

        tm = r.get("timings") or {}
        if isinstance(tm, dict):
            vv = _safe_float(tm.get("vision_llm_seconds"))
            if vv is not None:
                v.append(vv)
                vision_seen += 1
                if vv < 0.25:
                    vision_fast += 1
            aa = _safe_float(tm.get("agent_graph_seconds"))
            if aa is not None:
                a.append(aa)
            tt = _safe_float(tm.get("total_seconds"))
            if tt is not None:
                tot.append(tt)

        td = r.get("tool_details") or []
        if isinstance(td, list) and td:
            secs = 0.0
            cnt = 0
            for d in td:
                if not isinstance(d, dict):
                    continue
                s = _safe_float(d.get("seconds"))
                if s is None:
                    continue
                secs += s
                cnt += 1
            if cnt > 0:
                tool_sum.append(secs)
                tool_count.append(cnt)

    out = {
        "latency_seconds": _percentiles(lat, ps=[0.5, 0.9, 0.95, 0.99]),
        "timings": {
            "vision_llm_seconds": _percentiles(v, ps=[0.5, 0.9, 0.95]),
            "agent_graph_seconds": _percentiles(a, ps=[0.5, 0.9, 0.95]),
            "total_seconds": _percentiles(tot, ps=[0.5, 0.9, 0.95]),
            "vision_llm_fast_fraction_lt_250ms": float(vision_fast / vision_seen) if vision_seen else 0.0,
            "vision_llm_samples": int(vision_seen),
        },
        "tools": {
            "tool_seconds_total": _percentiles(tool_sum, ps=[0.5, 0.9, 0.95]),
            "tool_count": _percentiles([float(x) for x in tool_count], ps=[0.5, 0.9, 0.95]),
        },
    }
    return out


def _tool_usage_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter()
    per_row_counts: List[int] = []
    for r in rows:
        tu = r.get("tool_usage") or []
        if not isinstance(tu, list):
            continue
        names = [x for x in tu if isinstance(x, str)]
        per_row_counts.append(len(names))
        for n in names:
            counts[n] += 1
    top = counts.most_common(15)
    return {
        "tool_usage_count": _percentiles([float(x) for x in per_row_counts], ps=[0.5, 0.9, 0.95]) if per_row_counts else {},
        "top_tools": [{"tool": k, "calls": int(v)} for k, v in top],
        "unique_tools": int(len(counts)),
    }


def _tool_conditional_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    For each tool, compute conditional metrics for rows where the tool was used vs not used.
    NOTE: This is descriptive only; tool selection is not random and is confounded by image difficulty.
    """
    tool_set: Counter[str] = Counter()
    tool_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    tool_to_not_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # Normalize tool usage per row
    for r in rows:
        tu = r.get("tool_usage") or []
        tools = [t for t in tu if isinstance(t, str)]
        tools_set = set(tools)
        for t in tools_set:
            tool_set[t] += 1
            tool_to_rows[t].append(r)

    # Build "not used" rows per tool (on demand)
    all_rows = rows
    out_tools: List[Dict[str, Any]] = []
    for tool, n_used in tool_set.most_common():
        used_rows = tool_to_rows.get(tool, [])
        if not used_rows:
            continue
        # rows where tool not used
        if tool not in tool_to_not_rows:
            tool_to_not_rows[tool] = [r for r in all_rows if tool not in set(r.get("tool_usage") or [])]
        not_rows = tool_to_not_rows[tool]

        used = _basic_metrics(used_rows)
        not_used = _basic_metrics(not_rows)
        delta_acc_ans = used["accuracy_answered"] - not_used["accuracy_answered"]
        delta_conf = used["mean_conf_answered"] - not_used["mean_conf_answered"]

        out_tools.append(
            {
                "tool": tool,
                "n_used": int(n_used),
                "used_rate": float(n_used / len(all_rows)) if all_rows else 0.0,
                "used": used,
                "not_used": not_used,
                "delta_accuracy_answered": float(delta_acc_ans),
                "delta_mean_conf_answered": float(delta_conf),
            }
        )

    return {
        "note": "Conditional statistics only; tool selection is confounded by image content/difficulty and is not causal.",
        "tools": out_tools,
    }


def _infer_vision_model(cfg: ConfigKey, jsonl_name: str, rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Infer vision model with explicit rules when models field is missing.
    Returns (vision_model, source).
    """
    # Prefer explicit per-row provenance if available
    counts: Counter[str] = Counter()
    for r in rows:
        m = r.get("models")
        if isinstance(m, dict) and isinstance(m.get("vision"), str) and m.get("vision"):
            counts[m["vision"]] += 1
    if counts:
        vision_model = counts.most_common(1)[0][0]
        return vision_model, "models_field"

    name = jsonl_name.lower()
    agent = cfg.model
    if "glm46v-vision_agent_glm46" in name:
        return "z-ai/glm-4.6:nitro", "filename_glm46_vision"
    if name.startswith("g3") or "g3vision" in name:
        return "google/gemini-3-flash-preview", "filename_g3_prefix"
    if name == "b_or_g3flashprev.jsonl":
        return "google/gemini-3-flash-preview", "filename_b_or_g3"
    if isinstance(agent, str) and "claude-sonnet-4.5" in agent:
        return "anthropic/claude-sonnet-4.5", "agent_rule_sonnet"
    if isinstance(agent, str) and "gpt-5" in agent:
        return "gpt-5.2", "agent_rule_openai"

    # Default: Gemini vision for G3-prefixed experiments and other OpenRouter runs
    return "google/gemini-3-flash-preview", "default_gemini"


def summarize_config(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    # Dataset fingerprint (critical for scientific comparability)
    ids: List[str] = []
    exts: Counter[str] = Counter()
    img_roots: Counter[str] = Counter()
    for r in rows:
        sid = r.get("id")
        if sid is not None:
            ids.append(str(sid))
        img = r.get("image")
        if isinstance(img, str) and img:
            p = Path(img)
            exts[p.suffix.lower() or ""] += 1
            # root directory name (e.g., data2)
            try:
                parts = [x for x in p.parts if x]
                if parts:
                    img_roots[parts[-2] if len(parts) >= 2 else parts[-1]] += 1
            except Exception:
                pass
    ids_sorted = sorted(ids)
    ids_digest = hashlib.sha256(("\n".join(ids_sorted)).encode("utf-8")).hexdigest()[:16] if ids_sorted else ""
    confusion, support, abstain_by, abstain_count, error_count = _confusion_from_answered(rows)
    tp, tn, fp, fn = confusion["tp"], confusion["tn"], confusion["fp"], confusion["fn"]
    correct = tp + tn
    answered = total - abstain_count - error_count

    accuracy = correct / total if total else 0.0
    accuracy_answered = correct / answered if answered else 0.0
    coverage = answered / total if total else 0.0

    prec_f, rec_f, f1_f = _prf(tp, fp, fn)
    # for "real" class, treat real as positive: TP_real=tn, FP_real=fn, FN_real=fp
    prec_r, rec_r, f1_r = _prf(tn, fn, fp)
    bal_acc = (rec_f + rec_r) / 2.0 if total else 0.0
    mcc = _mcc(tp, tn, fp, fn)

    support_fake = int(support.get("fake", 0))
    support_real = int(support.get("real", 0))
    abstain_fake = int(abstain_by.get("fake", 0))
    abstain_real = int(abstain_by.get("real", 0))
    answered_fake = max(support_fake - abstain_fake, 0)
    answered_real = max(support_real - abstain_real, 0)
    coverage_fake = answered_fake / support_fake if support_fake else 0.0
    coverage_real = answered_real / support_real if support_real else 0.0

    fake_slip_rate = fn / support_fake if support_fake else 0.0
    real_false_flag_rate = fp / support_real if support_real else 0.0
    fake_catch_rate = tp / support_fake if support_fake else 0.0
    real_pass_rate = tn / support_real if support_real else 0.0

    abstain_rate = abstain_count / total if total else 0.0
    error_rate = error_count / total if total else 0.0

    # Confidence/calibration-ish metrics are computed on answered items only.
    conf_stats = _extract_answered_confidence_stats(rows)

    # Timing/tool usage stats
    timing = _timing_stats(rows)
    tool_usage = _tool_usage_stats(rows)

    # Model provenance coverage
    models_present = 0
    vision_models: Counter[str] = Counter()
    agent_models: Counter[str] = Counter()
    struct_models: Counter[str] = Counter()
    for r in rows:
        m = r.get("models")
        if isinstance(m, dict) and m:
            models_present += 1
            if isinstance(m.get("vision"), str):
                vision_models[m["vision"]] += 1
            if isinstance(m.get("agent"), str):
                agent_models[m["agent"]] += 1
            if isinstance(m.get("structuring"), str):
                struct_models[m["structuring"]] += 1

    return {
        "dataset_fingerprint": {
            "n_records": int(total),
            "n_unique_ids": int(len(set(ids))),
            "id_sha256_16": ids_digest,
            "sample_id_preview": ids_sorted[:10],
            "image_ext_counts": [{"ext": k, "count": int(v)} for k, v in exts.most_common()],
            "image_root_dir_guess": [{"dir": k, "count": int(v)} for k, v in img_roots.most_common(5)],
        },
        "n_total": int(total),
        "n_answered": int(answered),
        "n_abstain": int(abstain_count),
        "n_error": int(error_count),
        "support": {"fake": support_fake, "real": support_real},
        "confusion_answered_only": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "metrics": {
            "accuracy_overall": float(accuracy),
            "accuracy_answered": float(accuracy_answered),
            "coverage": float(coverage),
            "balanced_accuracy_answered": float(bal_acc),
            "mcc_answered": float(mcc),
            "precision_fake_answered": float(prec_f),
            "recall_fake_answered": float(rec_f),
            "f1_fake_answered": float(f1_f),
            "precision_real_answered": float(prec_r),
            "recall_real_answered": float(rec_r),
            "f1_real_answered": float(f1_r),
            "abstain_rate": float(abstain_rate),
            "abstain_rate_fake": float(abstain_fake / support_fake) if support_fake else 0.0,
            "abstain_rate_real": float(abstain_real / support_real) if support_real else 0.0,
            "coverage_fake": float(coverage_fake),
            "coverage_real": float(coverage_real),
            "fake_slip_rate": float(fake_slip_rate),
            "real_false_flag_rate": float(real_false_flag_rate),
            "fake_catch_rate": float(fake_catch_rate),
            "real_pass_rate": float(real_pass_rate),
            "error_rate": float(error_rate),
        },
        "ci_95_wilson": {
            "accuracy_overall": wilson_ci(accuracy, total),
            "coverage": wilson_ci(coverage, total),
            # "accuracy_answered" CI is conditional on answered; treat as binomial over answered.
            "accuracy_answered": wilson_ci(accuracy_answered, answered) if answered else (0.0, 0.0),
        },
        "confidence_diagnostics": conf_stats,
        "timing": timing,
        "tool_usage": tool_usage,
        "tool_effects": _tool_conditional_stats(rows) if rows and any(r.get("use_tools") for r in rows) else {},
        "provenance": {
            "models_field_present_fraction": float(models_present / total) if total else 0.0,
            "vision_models": [{"model": k, "count": int(v)} for k, v in vision_models.most_common(5)],
            "agent_models": [{"model": k, "count": int(v)} for k, v in agent_models.most_common(5)],
            "structuring_models": [{"model": k, "count": int(v)} for k, v in struct_models.most_common(5)],
        },
    }


def _load_metrics_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _maybe_validate_against_metrics_json(
    config: ConfigKey,
    summary: Dict[str, Any],
    metrics_json: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Cross-check core counts/metrics vs existing *.metrics.json if present.
    This is best-effort; we do not fail the run on mismatch.
    """
    key = f"{config.model}|{config.mode}"
    trial0 = (((metrics_json.get(key) or {}).get("per_trial") or {}).get("0")) if metrics_json else None
    if not isinstance(trial0, dict):
        return {"validated": False, "note": "no_matching_metrics_key"}

    # Compare a small set of stable fields.
    mismatches: List[str] = []
    def chk(name: str, got: Any, exp: Any) -> None:
        if got != exp:
            mismatches.append(f"{name}: got={got} exp={exp}")

    chk("total", summary.get("n_total"), trial0.get("total"))
    chk("answered", summary.get("n_answered"), trial0.get("answered"))
    chk("abstain_count", summary.get("n_abstain"), trial0.get("abstain_count"))
    chk("error_count", summary.get("n_error"), trial0.get("error_count"))

    # floats: compare with tolerance
    def chkf(name: str, got: float, exp: float, tol: float = 1e-9) -> None:
        try:
            if abs(float(got) - float(exp)) > tol:
                mismatches.append(f"{name}: got={got:.6g} exp={float(exp):.6g}")
        except Exception:
            mismatches.append(f"{name}: got={got} exp={exp}")

    chkf("accuracy", summary["metrics"]["accuracy_overall"], float(trial0.get("accuracy", 0.0)))
    chkf("accuracy_answered", summary["metrics"]["accuracy_answered"], float(trial0.get("accuracy_answered", 0.0)))
    chkf("coverage", summary["metrics"]["coverage"], float(trial0.get("coverage", 0.0)))

    return {
        "validated": True,
        "mismatch_count": int(len(mismatches)),
        "mismatches": mismatches[:25],
    }


def _escape_md_cell(value: Any) -> str:
    if value is None:
        return "—"
    text = str(value)
    # Avoid breaking tables: escape pipes and collapse newlines.
    text = text.replace("|", "\\|").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    return text


def _md_table(rows: List[Dict[str, Any]], headers: List[Tuple[str, str]]) -> str:
    """
    headers: [(col_key, title)]
    """
    lines: List[str] = []
    lines.append("| " + " | ".join(t for _, t in headers) + " |")
    lines.append("|" + "|".join([":---:" if i == 0 else "---" for i in range(len(headers))]) + "|")
    for r in rows:
        vals = []
        for k, _ in headers:
            v = r.get(k)
            if isinstance(v, float):
                vals.append(_escape_md_cell(f"{v:.3f}"))
            else:
                vals.append(_escape_md_cell(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path (summary).")
    ap.add_argument("--out-md", type=Path, default=None, help="Optional markdown report output.")
    ap.add_argument("--validate-metrics-json", action="store_true", default=True)
    ap.add_argument("--no-validate-metrics-json", dest="validate_metrics_json", action="store_false")
    args = ap.parse_args()

    results_dir: Path = args.results_dir
    jsonl_files = sorted(results_dir.glob("*.jsonl"))
    metrics_files = {p.stem.replace(".metrics", ""): p for p in results_dir.glob("*.metrics.json")}

    all_configs: Dict[str, Any] = {}
    inventory: List[Dict[str, Any]] = []
    # For paired tests: accumulate { (model,mode) -> {id -> correct_bool} }
    correctness_by_config: Dict[Tuple[str, str], Dict[str, bool]] = defaultdict(dict)
    # For global tool analysis: accumulate rows from tool runs
    all_tool_rows: List[Dict[str, Any]] = []

    for jf in jsonl_files:
        rows = _load_jsonl(jf)
        by_cfg = _index_by_config(rows)
        # Most files are single-config; keep it general anyway.
        for cfg, cfg_rows in by_cfg.items():
            if cfg.mode == "tools":
                all_tool_rows.extend(cfg_rows)

            # Record per-sample correctness for paired tests (best-effort).
            for r in cfg_rows:
                sid = r.get("id")
                if sid is None:
                    continue
                sid_s = str(sid)
                gold = (r.get("label") or "").strip().lower()
                pred = r.get("prediction")
                is_err = bool(r.get("error"))
                ok = (not is_err) and (pred in ("real", "fake")) and (str(pred).strip().lower() == gold)
                correctness_by_config[(cfg.model, cfg.mode)][sid_s] = bool(ok)

            cfg_id = f"{cfg.model}|{cfg.mode}|{jf.name}"
            vision_model, vision_src = _infer_vision_model(cfg, jf.name, cfg_rows)
            summ = summarize_config(cfg_rows)

            validation = {"validated": False, "note": "disabled"}
            if args.validate_metrics_json:
                base = jf.stem
                mf = metrics_files.get(base)
                if mf is not None:
                    mjson = _load_metrics_json(mf)
                    validation = _maybe_validate_against_metrics_json(cfg, summ, mjson)
                else:
                    validation = {"validated": False, "note": "no_metrics_file_pair"}
            summ["validation_vs_metrics_json"] = validation

            # Lightweight metadata
            inventory.append(
                {
                    "jsonl": jf.name,
                    "agent_model": cfg.model,
                    "vision_model": vision_model,
                    "vision_model_source": vision_src,
                    "mode": cfg.mode,
                    "config": f"{cfg.model}|{cfg.mode}",
                    "dataset": summ.get("dataset_fingerprint", {}).get("id_sha256_16", ""),
                    "n_total": summ["n_total"],
                    "accuracy_overall": summ["metrics"]["accuracy_overall"],
                    "accuracy_answered": summ["metrics"]["accuracy_answered"],
                    "coverage": summ["metrics"]["coverage"],
                    "abstain_rate": summ["metrics"]["abstain_rate"],
                    "error_rate": summ["metrics"]["error_rate"],
                    "vision_fast_frac_lt_250ms": summ["timing"]["timings"]["vision_llm_fast_fraction_lt_250ms"],
                    "models_present_frac": summ["provenance"]["models_field_present_fraction"],
                }
            )

            all_configs[cfg_id] = {
                "jsonl_file": jf.name,
                "config": {"model": cfg.model, "mode": cfg.mode},
                "model_roles": {
                    "agent_model": cfg.model,
                    "vision_model": vision_model,
                    "vision_model_source": vision_src,
                },
                "summary": summ,
            }

    out_obj = {
        "results_dir": str(results_dir),
        "inventory": sorted(inventory, key=lambda r: (r["config"], r["jsonl"])),
        "configs": all_configs,
        "paired_tests": [],
        "tool_analysis_global": _tool_conditional_stats(all_tool_rows) if all_tool_rows else {},
    }

    # Paired McNemar tests: tools vs no-tools for the same model where overlapping IDs exist.
    paired: List[Dict[str, Any]] = []
    all_models = sorted({m for (m, _mode) in correctness_by_config.keys()})
    for model in all_models:
        corr_v = correctness_by_config.get((model, "no-tools"))
        corr_t = correctness_by_config.get((model, "tools"))
        if not corr_v or not corr_t:
            continue
        ids = sorted(set(corr_v.keys()) & set(corr_t.keys()))
        if not ids:
            continue
        n10 = 0  # no-tools correct, tools wrong
        n01 = 0  # no-tools wrong, tools correct
        both_ok = 0
        both_bad = 0
        for sid in ids:
            v_ok = bool(corr_v[sid])
            t_ok = bool(corr_t[sid])
            if v_ok and (not t_ok):
                n10 += 1
            elif (not v_ok) and t_ok:
                n01 += 1
            elif v_ok and t_ok:
                both_ok += 1
            else:
                both_bad += 1
        p = mcnemar_exact(n10, n01)
        paired.append(
            {
                "model": model,
                "n_common": int(len(ids)),
                "both_correct": int(both_ok),
                "vision_correct_tools_wrong": int(n10),
                "vision_wrong_tools_correct": int(n01),
                "both_wrong": int(both_bad),
                "mcnemar_p_two_sided": float(p),
            }
        )
    out_obj["paired_tests"] = sorted(paired, key=lambda r: r["mcnemar_p_two_sided"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")

    if args.out_md is not None:
        # Split by mode for scientifically comparable tables
        inv = out_obj["inventory"]
        vision = [r for r in inv if r["config"].endswith("|no-tools")]
        tools = [r for r in inv if r["config"].endswith("|tools")]
        vision.sort(key=lambda r: r["accuracy_overall"], reverse=True)
        tools.sort(key=lambda r: r["accuracy_overall"], reverse=True)

        md: List[str] = []
        md.append("# DF3 Evaluation Report (Generated)")
        md.append("")
        md.append("This file is auto-generated from `results/*.jsonl` by `scripts/summarize_results.py`.")
        md.append("It is intended as a *reproducible* backbone for a paper-quality narrative report.")
        md.append("")
        md.append("## How to Reproduce")
        md.append("")
        md.append("Run:")
        md.append("")
        md.append("```bash")
        md.append("python scripts/summarize_results.py --results-dir results --out artifacts/eval_summary.json --out-md docs/evaluation_report.generated.md")
        md.append("```")
        md.append("")
        md.append("The full machine-readable output is `artifacts/eval_summary.json`.")
        md.append("")
        md.append("## Inventory")
        md.append("")
        md.append(_md_table(inv, headers=[("agent_model", "Agent Model"), ("vision_model", "Vision Model"), ("mode", "Mode"), ("jsonl", "JSONL"), ("n_total", "n"), ("accuracy_overall", "Acc"), ("coverage", "Cov"), ("accuracy_answered", "Acc(ans)"), ("abstain_rate", "Abstain"), ("error_rate", "Errors")]))
        md.append("")

        # Detailed per-config metrics (pulled from the JSON summary; one row per cfg_id)
        detail_rows: List[Dict[str, Any]] = []
        for cfg_id, obj in out_obj["configs"].items():
            summ = (obj or {}).get("summary") or {}
            m = (summ.get("metrics") or {}) if isinstance(summ, dict) else {}
            ci = (summ.get("ci_95_wilson") or {}) if isinstance(summ, dict) else {}
            timing = (summ.get("timing") or {}) if isinstance(summ, dict) else {}
            lat = ((timing.get("latency_seconds") or {}) if isinstance(timing, dict) else {})
            dset = (summ.get("dataset_fingerprint") or {}) if isinstance(summ, dict) else {}
            confd = (summ.get("confidence_diagnostics") or {}) if isinstance(summ, dict) else {}
            roles = (obj.get("model_roles") or {}) if isinstance(obj, dict) else {}
            detail_rows.append(
                {
                    "cfg_id": cfg_id,
                    "agent_model": roles.get("agent_model", obj.get("config", {}).get("model")),
                    "vision_model": roles.get("vision_model", ""),
                    "vision_src": roles.get("vision_model_source", ""),
                    "config": f"{obj['config']['model']}|{obj['config']['mode']}",
                    "n": summ.get("n_total"),
                    "dataset": dset.get("id_sha256_16", ""),
                    "acc": float(m.get("accuracy_overall", 0.0)),
                    "acc_ci": "[{:.3f},{:.3f}]".format(*(ci.get("accuracy_overall") or (0.0, 0.0))),
                    "cov": float(m.get("coverage", 0.0)),
                    "cov_ci": "[{:.3f},{:.3f}]".format(*(ci.get("coverage") or (0.0, 0.0))),
                    "acc_ans": float(m.get("accuracy_answered", 0.0)),
                    "mcc_ans": float(m.get("mcc_answered", 0.0)),
                    "bal_acc_ans": float(m.get("balanced_accuracy_answered", 0.0)),
                    "fake_slip": float(m.get("fake_slip_rate", 0.0)),
                    "real_false_flag": float(m.get("real_false_flag_rate", 0.0)),
                    "abstain": float(m.get("abstain_rate", 0.0)),
                    "lat_p50": float(lat.get("p50", 0.0)) if isinstance(lat, dict) else 0.0,
                    "lat_p95": float(lat.get("p95", 0.0)) if isinstance(lat, dict) else 0.0,
                    "ece": float(confd.get("ece_predlabel_10bins", 0.0)),
                    "brier": float(confd.get("brier_p_fake", 0.0)),
                    "aurc": float(confd.get("aurc", 0.0)),
                }
            )
        # One row per cfg_id can include duplicates across files; keep as-is for traceability.
        detail_rows.sort(key=lambda r: (r["config"], r["cfg_id"]))

        md.append("## Key Metrics (per artifact/config)")
        md.append("")
        md.append("_Note: `acc` and `cov` are computed over all samples; `acc_ans`/`mcc_ans`/`bal_acc_ans` are computed on answered samples only (abstentions removed). `fake_slip` and `real_false_flag` are triage-style rates over all items in the class._")
        md.append("")
        md.append(
            _md_table(
                detail_rows,
                headers=[
                    ("cfg_id", "Artifact"),
                    ("agent_model", "Agent Model"),
                    ("vision_model", "Vision Model"),
                    ("vision_src", "Vision Src"),
                    ("dataset", "Dataset"),
                    ("n", "n"),
                    ("acc", "Acc"),
                    ("acc_ci", "Acc 95% CI"),
                    ("cov", "Cov"),
                    ("cov_ci", "Cov 95% CI"),
                    ("acc_ans", "Acc(ans)"),
                    ("mcc_ans", "MCC(ans)"),
                    ("bal_acc_ans", "BalAcc(ans)"),
                    ("fake_slip", "FakeSlip"),
                    ("real_false_flag", "RealFalseFlag"),
                    ("abstain", "Abstain"),
                    ("lat_p50", "Lat p50(s)"),
                    ("lat_p95", "Lat p95(s)"),
                    ("ece", "ECE"),
                    ("brier", "Brier"),
                    ("aurc", "AURC"),
                ],
            )
        )
        md.append("")

        # Rankings by vision model (within dataset)
        md.append("## Rankings by Vision Model (within dataset)")
        md.append("")
        inv_by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in inv:
            inv_by_dataset[str(r.get("dataset", ""))].append(r)
        for dataset_key, subset in sorted(inv_by_dataset.items()):
            if not dataset_key:
                dataset_key = "unknown"
            md.append(f"### Dataset {dataset_key}")
            md.append("")
            subset_sorted = sorted(subset, key=lambda r: (r.get("vision_model", ""), -float(r.get("accuracy_overall", 0.0))))
            md.append(
                _md_table(
                    subset_sorted,
                    headers=[
                        ("vision_model", "Vision Model"),
                        ("agent_model", "Agent Model"),
                        ("mode", "Mode"),
                        ("accuracy_overall", "Acc"),
                        ("coverage", "Cov"),
                        ("accuracy_answered", "Acc(ans)"),
                        ("abstain_rate", "Abstain"),
                    ],
                )
            )
            md.append("")
        # Tool analysis (global)
        md.append("## Tool Usage Analysis (All Tools Runs; Descriptive)")
        md.append("")
        md.append("_Note: Conditional statistics only; tool selection is confounded by image difficulty and is not causal._")
        md.append("")
        tool_global = out_obj.get("tool_analysis_global") or {}
        tool_stats = (tool_global.get("tools") or []) if isinstance(tool_global, dict) else []
        tool_rows: List[Dict[str, Any]] = []
        for t in tool_stats:
            used = t.get("used") or {}
            not_used = t.get("not_used") or {}
            tool_rows.append(
                {
                    "tool": t.get("tool"),
                    "n_used": t.get("n_used"),
                    "used_rate": t.get("used_rate", 0.0),
                    "acc_ans_used": used.get("accuracy_answered", 0.0),
                    "acc_ans_not": not_used.get("accuracy_answered", 0.0),
                    "delta_acc_ans": t.get("delta_accuracy_answered", 0.0),
                    "conf_used": used.get("mean_conf_answered", 0.0),
                    "conf_not": not_used.get("mean_conf_answered", 0.0),
                    "delta_conf": t.get("delta_mean_conf_answered", 0.0),
                }
            )
        if tool_rows:
            md.append(
                _md_table(
                    tool_rows,
                    headers=[
                        ("tool", "Tool"),
                        ("n_used", "Calls"),
                        ("used_rate", "Used Rate"),
                        ("acc_ans_used", "Acc(ans) Used"),
                        ("acc_ans_not", "Acc(ans) Not Used"),
                        ("delta_acc_ans", "DeltaAcc(ans)"),
                        ("conf_used", "Conf Used"),
                        ("conf_not", "Conf Not Used"),
                        ("delta_conf", "DeltaConf"),
                    ],
                )
            )
        else:
            md.append("_No tool-usage data available in artifacts._")
        md.append("")
        md.append("## Paired Comparisons (McNemar exact; tools vs no-tools)")
        md.append("")
        if out_obj.get("paired_tests"):
            md.append("| Model | n_common | both_correct | vision_correct_tools_wrong | vision_wrong_tools_correct | both_wrong | McNemar p (2-sided) |")
            md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for r in out_obj["paired_tests"]:
                md.append(
                    "| {model} | {n_common} | {both_correct} | {vision_correct_tools_wrong} | {vision_wrong_tools_correct} | {both_wrong} | {p:.3g} |".format(
                        model=r["model"],
                        n_common=r["n_common"],
                        both_correct=r["both_correct"],
                        vision_correct_tools_wrong=r["vision_correct_tools_wrong"],
                        vision_wrong_tools_correct=r["vision_wrong_tools_correct"],
                        both_wrong=r["both_wrong"],
                        p=r["mcnemar_p_two_sided"],
                    )
                )
        else:
            md.append("_No paired comparisons available (missing matching tools/no-tools runs with overlapping IDs)._")
        md.append("")
        md.append("## Rankings (Vision-only / no-tools)")
        md.append("")
        md.append(_md_table(vision, headers=[("agent_model", "Agent Model"), ("vision_model", "Vision Model"), ("mode", "Mode"), ("accuracy_overall", "Acc"), ("coverage", "Cov"), ("accuracy_answered", "Acc(ans)"), ("abstain_rate", "Abstain"), ("vision_fast_frac_lt_250ms", "Vision<250ms")]))
        md.append("")
        md.append("## Rankings (Tool-augmented / tools)")
        md.append("")
        md.append(_md_table(tools, headers=[("agent_model", "Agent Model"), ("vision_model", "Vision Model"), ("mode", "Mode"), ("accuracy_overall", "Acc"), ("coverage", "Cov"), ("accuracy_answered", "Acc(ans)"), ("abstain_rate", "Abstain"), ("vision_fast_frac_lt_250ms", "Vision<250ms")]))
        md.append("")
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text("\n".join(md).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

