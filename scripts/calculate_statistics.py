#!/usr/bin/env python3
"""
Calculate statistics from evaluation metrics files.

This script reads all *.metrics.json files from the results directory and computes:
- Matthews Correlation Coefficient (MCC)
- Wilson score confidence intervals
- Paired McNemar exact tests (when per-sample JSONL is available)
- (Optional) Two-proportion z-tests + Cohen's h (for independent comparisons only)

Usage:
    python scripts/calculate_statistics.py
    python scripts/calculate_statistics.py --results-dir results/
"""

import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
import argparse
from typing import Optional, Dict, Tuple, List


@dataclass
class ConfusionMatrix:
    """Confusion matrix with computed metrics."""
    tp: int  # True Positive (fake correctly identified as fake)
    tn: int  # True Negative (real correctly identified as real)
    fp: int  # False Positive (real incorrectly identified as fake)
    fn: int  # False Negative (fake incorrectly identified as real)
    
    @property
    def total_answered(self) -> int:
        return self.tp + self.tn + self.fp + self.fn
    
    @property
    def correct(self) -> int:
        return self.tp + self.tn


@dataclass
class ModelResult:
    """Results for a single model configuration."""
    name: str
    mode: str  # 'tools' or 'no-tools'
    n: int
    confusion: ConfusionMatrix
    accuracy: float
    accuracy_answered: float
    coverage: float
    balanced_accuracy: float
    abstain_count: int
    latency: float
    source_metrics_file: str
    config_key: str
    
    
def calculate_mcc(cm: ConfusionMatrix) -> tuple[float, str]:
    """
    Calculate Matthews Correlation Coefficient.
    
    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    
    Returns: (mcc_value, calculation_string)
    """
    tp, tn, fp, fn = cm.tp, cm.tn, cm.fp, cm.fn
    
    numerator = tp * tn - fp * fn
    
    # Check for zero denominators
    denom_parts = (tp + fp), (tp + fn), (tn + fp), (tn + fn)
    if any(p == 0 for p in denom_parts):
        return 0.0, "undefined (zero in denominator)"
    
    denominator = math.sqrt(denom_parts[0] * denom_parts[1] * denom_parts[2] * denom_parts[3])
    
    if denominator == 0:
        return 0.0, "undefined (zero denominator)"
    
    mcc = numerator / denominator
    
    calc = (
        f"MCC = ({tp}*{tn} - {fp}*{fn}) / sqrt[({tp}+{fp})({tp}+{fn})({tn}+{fp})({tn}+{fn})]\n"
        f"    = ({tp*tn} - {fp*fn}) / sqrt[{denom_parts[0]}*{denom_parts[1]}*{denom_parts[2]}*{denom_parts[3]}]\n"
        f"    = {numerator} / sqrt[{denom_parts[0] * denom_parts[1] * denom_parts[2] * denom_parts[3]}]\n"
        f"    = {numerator} / {denominator:.2f}\n"
        f"    = {mcc:.4f}"
    )
    
    return mcc, calc


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float, str]:
    """
    Calculate Wilson score confidence interval for a proportion.
    
    CI = (p + z^2/2n +/- z*sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n)
    
    Returns: (lower, upper, calculation_string)
    """
    z_sq = z ** 2
    
    # Components
    z_sq_over_n = z_sq / n
    z_sq_over_2n = z_sq / (2 * n)
    z_sq_over_4n_sq = z_sq / (4 * n * n)
    p_variance = p * (1 - p) / n
    
    # Standard error term
    se_term = math.sqrt(p_variance + z_sq_over_4n_sq)
    margin = z * se_term
    
    # Denominator
    denom = 1 + z_sq_over_n
    
    # CI bounds
    lower = (p + z_sq_over_2n - margin) / denom
    upper = (p + z_sq_over_2n + margin) / denom
    
    calc = (
        f"Wilson CI (95%, z={z}):\n"
        f"  p = {p:.4f}, n = {n}\n"
        f"  z^2/n = {z_sq_over_n:.6f}\n"
        f"  z^2/2n = {z_sq_over_2n:.6f}\n"
        f"  p(1-p)/n = {p_variance:.6f}\n"
        f"  z^2/4n^2 = {z_sq_over_4n_sq:.8f}\n"
        f"  SE term = sqrt({p_variance:.6f} + {z_sq_over_4n_sq:.8f}) = {se_term:.6f}\n"
        f"  Margin = {z} * {se_term:.6f} = {margin:.6f}\n"
        f"  Denominator = 1 + {z_sq_over_n:.6f} = {denom:.6f}\n"
        f"  Lower = ({p:.4f} + {z_sq_over_2n:.6f} - {margin:.6f}) / {denom:.6f} = {lower:.4f}\n"
        f"  Upper = ({p:.4f} + {z_sq_over_2n:.6f} + {margin:.6f}) / {denom:.6f} = {upper:.4f}\n"
        f"  95% CI = [{lower:.3f}, {upper:.3f}]"
    )
    
    return lower, upper, calc


def two_proportion_ztest(p1: float, n1: int, p2: float, n2: int) -> tuple[float, float, str]:
    """
    Two-proportion z-test.
    
    z = (p1 - p2) / sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    Returns: (z_statistic, p_value, calculation_string)
    """
    # Pooled proportion
    x1 = int(round(p1 * n1))
    x2 = int(round(p2 * n2))
    p_pooled = (x1 + x2) / (n1 + n2)
    
    # Standard error
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    if se == 0:
        return 0.0, 1.0, "SE = 0, cannot compute"
    
    # Z statistic
    z = (p1 - p2) / se
    
    # Two-tailed p-value (approximation using normal CDF)
    # For |z| > 3.5, p < 0.0005
    p_value = 2 * (1 - normal_cdf(abs(z)))
    
    calc = (
        f"Two-proportion z-test:\n"
        f"  p1 = {p1:.4f} (n1 = {n1}), p2 = {p2:.4f} (n2 = {n2})\n"
        f"  x1 = {x1}, x2 = {x2}\n"
        f"  Pooled p = ({x1} + {x2}) / ({n1} + {n2}) = {p_pooled:.4f}\n"
        f"  SE = sqrt[{p_pooled:.4f} * {1-p_pooled:.4f} * (1/{n1} + 1/{n2})]\n"
        f"     = sqrt[{p_pooled * (1-p_pooled):.6f} * {1/n1 + 1/n2:.6f}]\n"
        f"     = sqrt[{p_pooled * (1-p_pooled) * (1/n1 + 1/n2):.8f}]\n"
        f"     = {se:.6f}\n"
        f"  z = ({p1:.4f} - {p2:.4f}) / {se:.6f} = {z:.2f}\n"
        f"  p-value (two-tailed) = {p_value:.6f}" + (" (< 0.0001)" if p_value < 0.0001 else "")
    )
    
    return z, p_value, calc


def normal_cdf(x: float) -> float:
    """Approximate normal CDF using error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def cohens_h(p1: float, p2: float) -> tuple[float, str, str]:
    """
    Calculate Cohen's h effect size for two proportions.
    
    h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2))
    
    Returns: (h_value, interpretation, calculation_string)
    """
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    h = phi1 - phi2
    
    # Interpretation
    abs_h = abs(h)
    if abs_h < 0.2:
        interp = "Small"
    elif abs_h < 0.5:
        interp = "Small-Medium"
    elif abs_h < 0.8:
        interp = "Medium"
    else:
        interp = "Large"
    
    calc = (
        f"Cohen's h:\n"
        f"  phi1 = 2*arcsin(sqrt({p1:.4f})) = 2*arcsin({math.sqrt(p1):.4f}) = {phi1:.4f} rad\n"
        f"  phi2 = 2*arcsin(sqrt({p2:.4f})) = 2*arcsin({math.sqrt(p2):.4f}) = {phi2:.4f} rad\n"
        f"  h = {phi1:.4f} - {phi2:.4f} = {h:.4f}\n"
        f"  Interpretation: {interp} effect"
    )
    
    return h, interp, calc


def load_metrics(results_dir: Path) -> list[ModelResult]:
    """Load all metrics files from results directory."""
    results = []
    
    for metrics_file in sorted(results_dir.glob("*.metrics.json")):
        with open(metrics_file, encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract model results (skip _meta)
        for key, value in data.items():
            if key == "_meta":
                continue
            
            # Parse model name and mode
            if "|" in key:
                model_name, mode = key.rsplit("|", 1)
            else:
                model_name = key
                mode = "unknown"
            
            # Get trial 0 data
            trial = value["per_trial"]["0"]
            
            cm = ConfusionMatrix(
                tp=trial["confusion"]["tp"],
                tn=trial["confusion"]["tn"],
                fp=trial["confusion"]["fp"],
                fn=trial["confusion"]["fn"]
            )
            
            result = ModelResult(
                name=model_name,
                mode=mode,
                n=trial["total"],
                confusion=cm,
                accuracy=trial["accuracy"],
                accuracy_answered=trial["accuracy_answered"],
                coverage=trial["coverage"],
                balanced_accuracy=trial["balanced_accuracy"],
                abstain_count=trial["abstain_count"],
                latency=trial.get("avg_latency_seconds", float("nan")),
                source_metrics_file=metrics_file.name,
                config_key=key,
            )
            results.append(result)
    
    return results


def _find_jsonl_candidates(results_dir: Path) -> List[Path]:
    return sorted(results_dir.glob("*.jsonl"))


def _jsonl_contains_config(jsonl_path: Path, model: str, use_tools: bool, sample_lines: int = 200) -> bool:
    """
    Heuristic: scan first N lines looking for a line with both model and use_tools.
    This is fast and sufficient for mapping results/*.metrics.json -> results/*.jsonl in this repo.
    """
    needle_model = f"\"model\": \"{model}\""
    needle_tools = "\"use_tools\": true" if use_tools else "\"use_tools\": false"
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= sample_lines:
                    break
                if needle_model in line and needle_tools in line:
                    return True
    except OSError:
        return False
    return False


def find_jsonl_for_config(results_dir: Path, model: str, mode: str) -> Optional[Path]:
    """
    Locate a JSONL file that contains per-sample records for (model, mode).
    Prefer the metrics file base name when available (common case).
    """
    use_tools = (mode.strip() == "tools")

    # Common convention: metrics file is paired with a JSONL of the same base name.
    # e.g., results/A_openai_gpt52_visiononly.metrics.json -> results/A_openai_gpt52_visiononly.jsonl
    # We can't always infer the base name from here, so we scan results/*.jsonl.
    for cand in _find_jsonl_candidates(results_dir):
        if _jsonl_contains_config(cand, model=model, use_tools=use_tools):
            return cand
    return None


def load_correctness_from_jsonl(jsonl_path: Path, model: str, mode: str) -> Dict[str, bool]:
    """
    Build {sample_id -> is_correct} from a results JSONL file.
    """
    use_tools = (mode.strip() == "tools")
    out: Dict[str, bool] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            if j.get("model") != model:
                continue
            if j.get("use_tools") is not use_tools:
                continue
            sid = j.get("id")
            if not sid:
                continue
            pred = j.get("prediction")
            label = j.get("label")
            if pred is None or label is None:
                continue
            out[str(sid)] = (str(pred) == str(label))
    return out


def mcnemar_exact(n10: int, n01: int) -> float:
    """
    Exact two-sided McNemar test p-value using binomial distribution on discordant pairs.
    """
    b = n10 + n01
    if b == 0:
        return 1.0
    k = min(n10, n01)
    # two-sided exact p = 2 * sum_{i=0..k} Binom(b,0.5)[i]
    p = 0.0
    for i in range(0, k + 1):
        p += math.comb(b, i) * (0.5 ** b)
    return min(1.0, 2 * p)


def main():
    parser = argparse.ArgumentParser(description="Calculate statistics from evaluation results")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Directory containing .metrics.json files")
    parser.add_argument("--paired-tests", action="store_true", default=True,
                        help="Compute paired McNemar tests for models that have both tools and no-tools runs (requires JSONL).")
    parser.add_argument("--no-paired-tests", dest="paired_tests", action="store_false",
                        help="Disable paired McNemar tests.")
    parser.add_argument("--include-ztests", action="store_true", default=False,
                        help="Include independent two-proportion z-tests vs best model within each mode (not paired).")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed calculations")
    args = parser.parse_args()
    
    results = load_metrics(args.results_dir)
    
    if not results:
        print(f"No metrics files found in {args.results_dir}")
        return
    
    # Split by mode for scientifically comparable rankings
    vision_results = [r for r in results if r.mode.strip() == "no-tools"]
    tools_results = [r for r in results if r.mode.strip() == "tools"]
    other_results = [r for r in results if r.mode.strip() not in ("no-tools", "tools")]
    vision_results.sort(key=lambda r: r.accuracy, reverse=True)
    tools_results.sort(key=lambda r: r.accuracy, reverse=True)
    other_results.sort(key=lambda r: r.accuracy, reverse=True)
    
    print("=" * 80)
    print("DF3 EVALUATION STATISTICS")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Section 1: Summary Table
    # -------------------------------------------------------------------------
    print("\n## 1. MODEL PERFORMANCE SUMMARY\n")
    print("### Vision-only (no-tools)\n")
    print(f"{'Rank':<5} {'Model':<35} {'n':<6} {'Acc':<8} {'Cov':<8} {'MCC':<8} {'Latency':<10} {'Source':<30}")
    print("-" * 115)
    
    mcc_values = {}
    for i, r in enumerate(vision_results, 1):
        mcc, _ = calculate_mcc(r.confusion)
        mcc_values[(r.name, r.mode)] = mcc
        latency_str = f"{r.latency:.1f}s" if r.latency > 1 else f"{r.latency*1000:.0f}ms"
        print(f"{i:<5} {r.name:<35} {r.n:<6} {r.accuracy:<8.3f} {r.coverage:<8.3f} {mcc:<8.3f} {latency_str:<10} {r.source_metrics_file:<30}")

    print("\n### Tool-augmented (tools)\n")
    print(f"{'Rank':<5} {'Model':<35} {'n':<6} {'Acc':<8} {'Cov':<8} {'MCC':<8} {'Latency':<10} {'Source':<30}")
    print("-" * 115)
    for i, r in enumerate(tools_results, 1):
        mcc, _ = calculate_mcc(r.confusion)
        mcc_values[(r.name, r.mode)] = mcc
        latency_str = f"{r.latency:.1f}s" if r.latency > 1 else f"{r.latency*1000:.0f}ms"
        print(f"{i:<5} {r.name:<35} {r.n:<6} {r.accuracy:<8.3f} {r.coverage:<8.3f} {mcc:<8.3f} {latency_str:<10} {r.source_metrics_file:<30}")

    if other_results:
        print("\n### Other/unknown modes\n")
        print(f"{'Rank':<5} {'Model':<35} {'Mode':<10} {'n':<6} {'Acc':<8} {'Cov':<8} {'MCC':<8} {'Latency':<10} {'Source':<30}")
        print("-" * 125)
        for i, r in enumerate(other_results, 1):
            mcc, _ = calculate_mcc(r.confusion)
            mcc_values[(r.name, r.mode)] = mcc
            latency_str = f"{r.latency:.1f}s" if r.latency > 1 else f"{r.latency*1000:.0f}ms"
            print(f"{i:<5} {r.name:<35} {r.mode:<10} {r.n:<6} {r.accuracy:<8.3f} {r.coverage:<8.3f} {mcc:<8.3f} {latency_str:<10} {r.source_metrics_file:<30}")
    
    # -------------------------------------------------------------------------
    # Section 2: MCC Calculations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("## 2. MATTHEWS CORRELATION COEFFICIENT (MCC)")
    print("=" * 80)
    print("\nFormula: MCC = (TP*TN - FP*FN) / sqrt[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]")
    
    for r in (vision_results + tools_results + other_results):
        mcc, calc = calculate_mcc(r.confusion)
        print(f"\n### {r.name} ({r.mode})")
        print(f"Confusion Matrix: TP={r.confusion.tp}, TN={r.confusion.tn}, FP={r.confusion.fp}, FN={r.confusion.fn}")
        if args.verbose:
            print(calc)
        else:
            print(f"MCC = {mcc:.4f}")
    
    # -------------------------------------------------------------------------
    # Section 3: Confidence Intervals
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("## 3. WILSON SCORE CONFIDENCE INTERVALS (95%)")
    print("=" * 80)
    
    print(f"\n{'Model':<35} {'Mode':<10} {'Accuracy':<10} {'95% CI':<20}")
    print("-" * 80)
    
    ci_values = {}
    for r in (vision_results + tools_results + other_results):
        lower, upper, calc = wilson_ci(r.accuracy, r.n)
        ci_values[(r.name, r.mode)] = (lower, upper)
        print(f"{r.name:<35} {r.mode:<10} {r.accuracy:<10.3f} [{lower:.3f}, {upper:.3f}]")
        if args.verbose:
            print(f"  {calc}\n")
    
    # -------------------------------------------------------------------------
    # Section 4: Paired comparisons (preferred when comparing tools vs no-tools)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("## 4. PAIRED COMPARISONS (McNemar exact; requires JSONL)")
    print("=" * 80)

    if args.paired_tests:
        by_model: Dict[str, Dict[str, ModelResult]] = {}
        for r in (vision_results + tools_results):
            by_model.setdefault(r.name, {})[r.mode.strip()] = r

        any_pairs = False
        print(f"\n{'Model':<35} {'n_common':<10} {'Acc(no-tools)':<14} {'Acc(tools)':<12} {'n10':<6} {'n01':<6} {'p':<12} {'JSONL(no-tools)':<25} {'JSONL(tools)':<25}")
        print("-" * 150)

        for model, modes in by_model.items():
            if "no-tools" not in modes or "tools" not in modes:
                continue
            r_v = modes["no-tools"]
            r_t = modes["tools"]

            # Find JSONL files for each mode
            jsonl_v = find_jsonl_for_config(args.results_dir, model=model, mode="no-tools")
            jsonl_t = find_jsonl_for_config(args.results_dir, model=model, mode="tools")
            if jsonl_v is None or jsonl_t is None:
                continue

            try:
                corr_v = load_correctness_from_jsonl(jsonl_v, model=model, mode="no-tools")
                corr_t = load_correctness_from_jsonl(jsonl_t, model=model, mode="tools")
            except Exception:
                continue

            ids = sorted(set(corr_v.keys()) & set(corr_t.keys()))
            if not ids:
                continue

            n10 = n01 = 0
            for sid in ids:
                v_ok = corr_v[sid]
                t_ok = corr_t[sid]
                if v_ok and (not t_ok):
                    n10 += 1
                elif (not v_ok) and t_ok:
                    n01 += 1

            p = mcnemar_exact(n10, n01)
            any_pairs = True
            print(f"{model:<35} {len(ids):<10} {r_v.accuracy:<14.3f} {r_t.accuracy:<12.3f} {n10:<6} {n01:<6} {p:<12.3g} {jsonl_v.name:<25} {jsonl_t.name:<25}")

        if not any_pairs:
            print("\nNo paired comparisons available (missing JSONL files or no overlapping IDs).")
    else:
        print("\nPaired tests disabled (--no-paired-tests).")

    # Optional: independent comparisons within each mode (not paired)
    if args.include_ztests:
        print("\n" + "=" * 80)
        print("## 4B. INDEPENDENT COMPARISONS WITHIN MODE (Two-proportion z-test; NOT paired)")
        print("=" * 80)
        for mode_name, subset in [("no-tools", vision_results), ("tools", tools_results)]:
            if len(subset) < 2:
                continue
            best = subset[0]
            print(f"\nMode: {mode_name}. Reference: {best.name}, acc={best.accuracy:.3f}, n={best.n}")
            print(f"{'Comparison':<55} {'Delta':<10} {'z':<10} {'p':<12} {'h':<10} {'Effect':<12}")
            print("-" * 115)
            for r in subset[1:]:
                if r.n != best.n:
                    continue
                z, p_val, _ = two_proportion_ztest(best.accuracy, best.n, r.accuracy, r.n)
                h, interp, _ = cohens_h(best.accuracy, r.accuracy)
                delta = best.accuracy - r.accuracy
                p_str = f"{p_val:.6f}" if p_val >= 0.0001 else "<0.0001"
                comp_name = f"{best.name[:20]} vs {r.name[:25]} ({mode_name})"
                print(f"{comp_name:<55} {delta:<10.3f} {z:<10.2f} {p_str:<12} {h:<10.3f} {interp:<12}")
    
    # -------------------------------------------------------------------------
    # Section 5: Raw Data Export
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("## 5. RAW CONFUSION MATRICES")
    print("=" * 80)
    
    print(f"\n{'Model':<35} {'Mode':<10} {'n':<6} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6} {'Abstain':<8}")
    print("-" * 95)
    
    for r in (vision_results + tools_results + other_results):
        cm = r.confusion
        print(f"{r.name:<35} {r.mode:<10} {r.n:<6} {cm.tp:<6} {cm.tn:<6} {cm.fp:<6} {cm.fn:<6} {r.abstain_count:<8}")
    
    # -------------------------------------------------------------------------
    # Section 6: Key Statistics for Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("## 6. REPORT")
    print("=" * 80)
    
    print("\n### Markdown Table\n")
    def emit_md_table(subset: List[ModelResult], title: str) -> None:
        if not subset:
            return
        print(f"#### {title}\n")
        print("| Rank | Model | Accuracy | 95% CI | Coverage | MCC | Latency | Source |")
        print("|:----:|-------|:--------:|:------:|:--------:|:---:|:-------:|--------|")
        for i, r in enumerate(subset, 1):
            mcc = mcc_values.get((r.name, r.mode))
            if mcc is None:
                mcc, _ = calculate_mcc(r.confusion)
            lower, upper = ci_values.get((r.name, r.mode), (None, None))
            latency_str = f"{r.latency*1000:.0f}ms" if r.latency < 1 else f"{r.latency:.0f}s"
            mode_str = "(Vision)" if r.mode.strip() == "no-tools" else "(Tools)" if r.mode.strip() == "tools" else f"({r.mode})"
            if lower is None or upper is None:
                ci = ""
            else:
                ci = f"[{lower*100:.1f}%, {upper*100:.1f}%]"
            if i == 1:
                print(f"| **{i}** | **{r.name} {mode_str}** | **{r.accuracy*100:.1f}%** | {ci} | **{r.coverage*100:.1f}%** | **{mcc:.2f}** | **{latency_str}** | `{r.source_metrics_file}` |")
            else:
                print(f"| {i} | {r.name} {mode_str} | {r.accuracy*100:.1f}% | {ci} | {r.coverage*100:.1f}% | {mcc:.2f} | {latency_str} | `{r.source_metrics_file}` |")
        print("")

    emit_md_table(vision_results, "Vision-only (no-tools)")
    emit_md_table(tools_results, "Tool-augmented (tools)")
    if other_results:
        emit_md_table(other_results, "Other/unknown modes")


if __name__ == "__main__":
    main()
