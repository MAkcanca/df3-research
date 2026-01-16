"""
Batch evaluator for forensic image detection across multiple LLMs.

It compares model performance with and without forensic tools, optionally
using OpenRouter as the provider. Dataset format: JSONL with fields:
{
  "id": "sample-1",
  "image": "relative/or/absolute/path/to/image.jpg",
  "label": "real" | "fake"   # fake means AI-generated or manipulated
}
"""

import argparse
import json
import logging
import os
import random
import statistics
import sys
import time
import threading
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from tqdm import tqdm

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Suppress BAML verbose logging
os.environ["BAML_LOG"] = "OFF"

# Disable logfire console output (the LangGraph trace trees)
# Must be set before any logfire.configure() call
os.environ["LOGFIRE_CONSOLE"] = "false"

# Suppress verbose logging from various libraries
logging.getLogger('opentelemetry.attributes').setLevel(logging.ERROR)
logging.getLogger('opentelemetry').setLevel(logging.ERROR)
logging.getLogger('langchain').setLevel(logging.WARNING)
logging.getLogger('langgraph').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

import logfire  # noqa: E402

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import ForensicAgent  # noqa: E402
from src.tools.forensic.code_execution_tool import clean_artifacts_dir  # noqa: E402

# Configure logfire after imports (works standalone without LangSmith)
# Suppress logfire console output during evaluation (the LangGraph trace trees)
logfire.configure(scrubbing=False, console=False, send_to_logfire=False)
from src.tools.forensic import prewarm_trufor_model, prewarm_residual_extractor  # noqa: E402
from src.tools.forensic.cache import ToolCache, set_cache  # noqa: E402

# Reduce noise from internal modules that configure their own INFO loggers
logging.getLogger().setLevel(logging.WARNING)
for _name in [
    "src.agents.forensic_agent",
    "src.agents.baml_forensic",
    "src.tools.forensic",
]:
    _l = logging.getLogger(_name)
    _l.setLevel(logging.WARNING)
    _l.propagate = False

# Set up logging - reduce verbosity for cleaner output
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)  # Only show warnings and errors


@contextmanager
def _suppress_stdout():
    """Suppress noisy third-party stdout without affecting tqdm (stderr)."""
    # Save original stdout to ensure we can restore it even if something goes wrong
    original_stdout = sys.stdout
    try:
        with open(os.devnull, "w") as _devnull:
            with redirect_stdout(_devnull):
                yield
    finally:
        # Ensure stdout is always restored, even if there's an exception
        if sys.stdout is not original_stdout:
            sys.stdout = original_stdout


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL dataset."""
    items: List[Dict[str, Any]] = []
    base_dir = path.parent
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Support both "image" and "image_path" field names
            image_key = record.get("image") or record.get("image_path")
            if not image_key:
                raise ValueError(f"Record missing 'image' or 'image_path' field: {record}")
            image_path = Path(image_key)
            if not image_path.is_absolute():
                image_path = (base_dir / image_path).resolve()
            # Support both "label" and "ground_truth" field names
            label_key = record.get("label") or record.get("ground_truth")
            if not label_key:
                raise ValueError(f"Record missing 'label' or 'ground_truth' field: {record}")
            items.append(
                {
                    "id": record.get("id") or image_path.stem,
                    "image": str(image_path),
                    "label": label_key.lower(),
                    "meta": record.get("meta", {}),
                }
            )
    return items


def preflight_dataset(
    dataset: List[Dict[str, Any]],
    *,
    skip_missing_images: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Validate dataset items and optionally drop missing images.

    Returns:
        (filtered_dataset, issues)
    """
    filtered: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    for item in dataset:
        image_path = item.get("image")
        if not image_path or not os.path.exists(image_path):
            issues.append(
                {
                    "id": item.get("id"),
                    "image": image_path,
                    "error_type": "missing_image",
                    "error": f"Image not found: {image_path}",
                }
            )
            if not skip_missing_images:
                filtered.append(item)
            continue
        filtered.append(item)
    return filtered, issues


def build_headers(referer: Optional[str], title: Optional[str]) -> Optional[Dict[str, str]]:
    """Optional headers for OpenRouter."""
    headers: Dict[str, str] = {}
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers or None


def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _compute_running_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute quick running metrics for live display."""
    if not records:
        return {"n": 0, "acc": 0.0, "correct": 0, "wrong": 0, "uncertain": 0, "errors": 0}
    
    correct = wrong = uncertain = errors = 0
    for rec in records:
        if rec.get("error"):
            errors += 1
            continue
        pred = rec.get("prediction")
        label = rec.get("label")
        if pred == "uncertain" or pred is None:
            uncertain += 1
        elif pred == label:
            correct += 1
        else:
            wrong += 1
    
    n = len(records)
    answered = correct + wrong
    acc = correct / answered if answered > 0 else 0.0
    
    return {
        "n": n,
        "acc": acc,
        "correct": correct,
        "wrong": wrong,
        "uncertain": uncertain,
        "errors": errors,
    }


def compute_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute accuracy and balanced metrics, tracking abstains/errors explicitly."""
    tp = fp = tn = fn = 0
    abstain = error_count = 0
    confidences: List[float] = []
    label_counts = {"real": 0, "fake": 0}
    abstain_by_label = {"real": 0, "fake": 0}
    latencies: List[float] = []
    tool_counts: List[int] = []
    tool_seconds_totals: List[float] = []

    records_list = list(records)
    total = len(records_list)

    for rec in records_list:
        gold = rec.get("label")
        pred = rec.get("prediction")
        if gold in label_counts:
            label_counts[gold] += 1

        if rec.get("error"):
            error_count += 1
            continue

        if pred == "uncertain" or pred is None:
            abstain += 1
            if gold in abstain_by_label:
                abstain_by_label[gold] += 1
            continue

        if gold == "fake" and pred == "fake":
            tp += 1
        elif gold == "real" and pred == "real":
            tn += 1
        elif gold == "real" and pred == "fake":
            fp += 1
        elif gold == "fake" and pred == "real":
            fn += 1

        if pred in ("real", "fake"):
            confidences.append(rec.get("confidence", 0.0))
            if isinstance(rec.get("latency_seconds"), (int, float)):
                latencies.append(float(rec["latency_seconds"]))
            tool_usage = rec.get("tool_usage") or []
            if isinstance(tool_usage, list):
                tool_counts.append(len(tool_usage))
            tool_details = rec.get("tool_details") or []
            if isinstance(tool_details, list):
                secs = 0.0
                for d in tool_details:
                    try:
                        secs += float((d or {}).get("seconds", 0.0))
                    except Exception:
                        continue
                if secs > 0:
                    tool_seconds_totals.append(secs)

    correct = tp + tn
    accuracy = correct / total if total else 0.0
    precision_fake = tp / (tp + fp) if (tp + fp) else 0.0
    recall_fake = tp / (tp + fn) if (tp + fn) else 0.0
    f1_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if (precision_fake + recall_fake) else 0.0

    precision_real = tn / (tn + fn) if (tn + fn) else 0.0
    recall_real = tn / (tn + fp) if (tn + fp) else 0.0
    f1_real = 2 * precision_real * recall_real / (precision_real + recall_real) if (precision_real + recall_real) else 0.0

    tpr_fake = recall_fake
    tpr_real = recall_real
    balanced_accuracy = (tpr_fake + tpr_real) / 2 if total else 0.0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    answered = total - error_count - abstain
    accuracy_answered = correct / answered if answered > 0 else 0.0
    coverage = answered / total if total else 0.0
    # Class-conditional coverage / abstention (important for selective classification)
    support_fake = int(label_counts.get("fake", 0))
    support_real = int(label_counts.get("real", 0))
    abstain_fake = int(abstain_by_label.get("fake", 0))
    abstain_real = int(abstain_by_label.get("real", 0))
    answered_fake = max(support_fake - abstain_fake, 0)
    answered_real = max(support_real - abstain_real, 0)
    coverage_fake = answered_fake / support_fake if support_fake else 0.0
    coverage_real = answered_real / support_real if support_real else 0.0

    # Triage-style rates assuming "uncertain" routes to manual review:
    # - fake_slip_rate: fraction of all fake items that were incorrectly passed as real
    # - real_false_flag_rate: fraction of all real items incorrectly flagged as fake
    fake_slip_rate = fn / support_fake if support_fake else 0.0
    real_false_flag_rate = fp / support_real if support_real else 0.0
    fake_catch_rate = tp / support_fake if support_fake else 0.0
    real_pass_rate = tn / support_real if support_real else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_tool_count = sum(tool_counts) / len(tool_counts) if tool_counts else 0.0
    avg_tool_seconds_total = (
        sum(tool_seconds_totals) / len(tool_seconds_totals) if tool_seconds_totals else 0.0
    )

    return {
        "total": total,
        "correct": correct,
        "answered": answered,
        "accuracy": accuracy,
        "accuracy_answered": accuracy_answered,
        "coverage": coverage,
        "precision_fake": precision_fake,
        "recall_fake": recall_fake,
        "f1_fake": f1_fake,
        "precision_real": precision_real,
        "recall_real": recall_real,
        "f1_real": f1_real,
        "balanced_accuracy": balanced_accuracy,
        # Explicit answered-only aliases (these metrics ignore abstains by construction)
        "balanced_accuracy_answered": balanced_accuracy,
        "precision_fake_answered": precision_fake,
        "recall_fake_answered": recall_fake,
        "f1_fake_answered": f1_fake,
        "precision_real_answered": precision_real,
        "recall_real_answered": recall_real,
        "f1_real_answered": f1_real,
        "avg_confidence": avg_conf,
        "avg_latency_seconds": avg_latency,
        "avg_tool_count": avg_tool_count,
        "avg_tool_seconds_total": avg_tool_seconds_total,
        "abstain_count": abstain,
        "abstain_count_fake": abstain_fake,
        "abstain_count_real": abstain_real,
        "error_count": error_count,
        "abstain_rate": abstain / total if total else 0.0,
        "abstain_rate_fake": abstain_fake / support_fake if support_fake else 0.0,
        "abstain_rate_real": abstain_real / support_real if support_real else 0.0,
        "error_rate": error_count / total if total else 0.0,
        "coverage_fake": coverage_fake,
        "coverage_real": coverage_real,
        "fake_slip_rate": fake_slip_rate,
        "real_false_flag_rate": real_false_flag_rate,
        "fake_catch_rate": fake_catch_rate,
        "real_pass_rate": real_pass_rate,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "support": label_counts,
    }


def normalize_verdict(verdict: Optional[str]) -> str:
    """Normalize a free-form verdict to canonical labels."""
    if not verdict:
        return "uncertain"
    v = str(verdict).strip().lower()
    # Normalize punctuation and whitespace (e.g., "Inconclusive.", "can't determine")
    v = re.sub(r"[\s_]+", " ", v)
    v = re.sub(r"[^a-z0-9\-\s]", "", v)
    v = " ".join(v.split())
    fake_tokens = {
        "fake",
        "ai-generated",
        "ai generated",
        "synthetic",
        "manipulated",
        "tampered",
        "tamper",
        "deepfake",
        "ai",
    }
    real_tokens = {"real", "authentic", "natural", "genuine"}
    uncertain_tokens = {
        "uncertain",
        "inconclusive",
        "unknown",
        "undetermined",
        "unsure",
        "not sure",
        "cannot determine",
        "cant determine",
        "impossible to tell",
    }
    if v in fake_tokens:
        return "fake"
    if v in real_tokens:
        return "real"
    if v in uncertain_tokens:
        return "uncertain"
    return "uncertain"


def summarize_trials(trial_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean/std across trials for headline metrics."""
    if not trial_metrics:
        return {}
    
    headline_keys = [
        "accuracy",
        "accuracy_answered",
        "coverage",
        "balanced_accuracy",
        "balanced_accuracy_answered",
        "f1_fake",
        "f1_real",
        "precision_fake",
        "recall_fake",
        "precision_real",
        "recall_real",
        "abstain_rate",
        "abstain_rate_fake",
        "abstain_rate_real",
        "coverage_fake",
        "coverage_real",
        "fake_slip_rate",
        "real_false_flag_rate",
        "fake_catch_rate",
        "real_pass_rate",
    ]
    summary: Dict[str, Any] = {"mean": {}, "std": {}}
    for key in headline_keys:
        values = [m.get(key, 0.0) for m in trial_metrics]
        summary["mean"][key] = statistics.mean(values) if values else 0.0
        summary["std"][key] = statistics.stdev(values) if len(values) > 1 else 0.0
    
    error_rates = [m.get("error_rate", 0.0) for m in trial_metrics]
    abstain_rates = [m.get("abstain_rate", 0.0) for m in trial_metrics]
    summary["mean_error_rate"] = statistics.mean(error_rates) if error_rates else 0.0
    summary["mean_abstain_rate"] = statistics.mean(abstain_rates) if abstain_rates else 0.0
    
    summary["totals"] = {
        "avg_total": statistics.mean([m.get("total", 0) for m in trial_metrics]) if trial_metrics else 0,
        "avg_support_fake": statistics.mean([m.get("support", {}).get("fake", 0) for m in trial_metrics]) if trial_metrics else 0,
        "avg_support_real": statistics.mean([m.get("support", {}).get("real", 0) for m in trial_metrics]) if trial_metrics else 0,
    }
    return summary


def evaluate(
    dataset: List[Dict[str, Any]],
    models: List[str],
    use_tools_options: List[bool],
    api_key: str,
    base_url: Optional[str],
    default_headers: Optional[Dict[str, str]],
    temperature: float,
    max_iterations: int,
    trials: int,
    num_workers: int,
    reasoning_effort: Optional[str] = None,
    vision_model: Optional[str] = None,
    structuring_model: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run evaluation for all model + tool combinations and trials."""
    results: List[Dict[str, Any]] = []
    cancelled = False
    cancel_event = threading.Event()

    # Pre-warm forensic tool models before starting workers to avoid concurrent loading
    # This is especially important when using multiple workers, as each worker would
    # otherwise try to load models independently, causing contention and wasted time.
    if num_workers > 1 and any(use_tools_options):
        print("Pre-warming forensic tool models...", end=" ", flush=True)
        try:
            with _suppress_stdout():
                # Pre-warm TruFor (primary manipulation detector)
                # Device selection is handled internally by TruFor
                prewarm_trufor_model()

                # Pre-warm DRUNet/ResidualExtractor (used by extract_residuals)
                prewarm_residual_extractor()

            print("OK")
        except Exception as e:
            print(f"WARN (will load lazily: {e})")

    try:
        for model in models:
            for use_tools in use_tools_options:
                for trial in range(trials):
                    if cancel_event.is_set():
                        raise KeyboardInterrupt
                    thread_local = threading.local()

                    def get_agent():
                        """Create one agent per worker thread to avoid cross-thread state."""
                        if not hasattr(thread_local, "agent"):
                            thread_local.agent = ForensicAgent(
                                llm_model=model,
                                vision_model=vision_model,
                                structuring_model=structuring_model,
                                temperature=temperature,
                                reasoning_effort=reasoning_effort,
                                api_key=api_key,
                                base_url=base_url,
                                default_headers=default_headers,
                                max_iterations=max_iterations,
                                enable_checkpointer=False,
                            )
                        return thread_local.agent

                    def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
                        if cancel_event.is_set():
                            return {
                                "id": item.get("id"),
                                "model": model,
                                "use_tools": use_tools,
                                "trial": trial,
                                "label": item.get("label"),
                                "image": item.get("image"),
                                "prediction": "uncertain",
                                "confidence": 0.0,
                                "error": "cancelled",
                                "error_type": "cancelled",
                            }
                        record = {
                            "id": item["id"],
                            "model": model,
                            "use_tools": use_tools,
                            "trial": trial,
                            "label": item["label"],
                            "image": item["image"],
                            "meta": item.get("meta", {}),
                            "run_config": {
                                "temperature": temperature,
                                "max_iterations": max_iterations,
                                "reasoning_effort": reasoning_effort,
                            },
                        }
                        start = time.time()
                        try:
                            if not os.path.exists(item["image"]):
                                raise FileNotFoundError(f"Image not found: {item['image']}")
                            # These runs are often very chatty (logfire/langgraph + tool prints).
                            # Suppress per-task (important: ThreadPool workers won't inherit outer contexts).
                            # NOTE: do NOT redirect stderr here; tqdm uses stderr in the main thread.
                            with logfire.suppress_instrumentation(), _suppress_stdout():
                                analysis = get_agent().analyze(
                                    image_path=item["image"],
                                    user_query=None,
                                    use_tools=use_tools,
                                )
                            latency = time.time() - start

                            analysis_verdict = normalize_verdict(analysis.get("verdict"))
                            parsed_verdict = normalize_verdict(
                                (analysis.get("raw_parsed") or {}).get("verdict")
                            )
                            final_verdict = (
                                parsed_verdict if parsed_verdict != "uncertain" else analysis_verdict
                            )

                            record.update(
                                {
                                    "prediction": final_verdict,
                                    "analysis_verdict": analysis_verdict,
                                    "parsed_verdict": parsed_verdict,
                                    "confidence": analysis.get("confidence", 0.0),
                                    "rationale": analysis.get("rationale", ""),
                                    "raw_analysis": analysis.get("raw_text", ""),
                                    "raw_parsed": analysis.get("raw_parsed", {}),
                                    # Provenance: which models actually produced which parts.
                                    # This is critical for scientific validity when using --vision-model / --structuring-model overrides.
                                    "models": analysis.get("models", {}),
                                    "visual_description": analysis.get("visual_description", ""),
                                    "forensic_summary": analysis.get("forensic_summary", ""),
                                    "prompts": analysis.get("prompts", {}),
                                    "prompt_hashes": {
                                        k: _hash_text(v)
                                        for k, v in (analysis.get("prompts", {}) or {}).items()
                                        if isinstance(v, str)
                                    },
                                    "tool_usage": analysis.get("tool_usage", []),
                                    "tool_details": analysis.get("tool_details", []),
                                    "tool_results": analysis.get("tool_results", []),
                                    "timings": analysis.get("timings", {}),
                                    "latency_seconds": latency,
                                }
                            )
                        except Exception as exc:
                            error_msg = str(exc)
                            error_type = (
                                "missing_image" if isinstance(exc, FileNotFoundError) else "exception"
                            )
                            record.update(
                                {
                                    "prediction": "uncertain",
                                    "confidence": 0.0,
                                    "error": error_msg,
                                    "error_type": error_type,
                                }
                            )
                            # Log the error with context
                            logger.error(
                                f"[ERROR] Failed to analyze item {item['id']} "
                                f"(model={model}, use_tools={use_tools}, trial={trial}): {error_msg}",
                                exc_info=True,
                            )
                            # Check if this is a tool-related error
                            is_tool_error = any(
                                x in error_msg.lower()
                                for x in [
                                    "gpu",
                                    "tool",
                                    "aborted",
                                    "cuda",
                                    "memory",
                                    "timeout",
                                    "forensic",
                                    "jpeg",
                                    "python_code",
                                ]
                            )
                            if is_tool_error:
                                logger.warning(
                                    f"[TOOL ERROR] Tool execution error detected for item {item['id']}: {error_msg}"
                                )
                        return record

                    # Create progress bar description
                    config_desc = f"{model}|{'tools' if use_tools else 'no-tools'}|t{trial+1}"

                    # Track results for this configuration to compute running metrics
                    config_results: List[Dict[str, Any]] = []

                    def update_pbar_metrics(pbar):
                        """Update progress bar with running metrics."""
                        m = _compute_running_metrics(config_results)
                        if m["n"] > 0:
                            pbar.set_postfix_str(
                                f"acc={m['acc']:.1%} ok={m['correct']} bad={m['wrong']} unc={m['uncertain']} err={m['errors']}",
                                refresh=True
                            )

                    if num_workers <= 1:
                        try:
                            with tqdm(
                                total=len(dataset),
                                desc=config_desc,
                                leave=True,
                                ncols=120,
                                ascii=True,
                            ) as pbar:
                                pbar.refresh()
                                for item in dataset:
                                    record = process_item(item)
                                    results.append(record)
                                    config_results.append(record)
                                    pbar.update(1)
                                    update_pbar_metrics(pbar)
                        except KeyboardInterrupt:
                            cancel_event.set()
                            raise
                    else:
                        executor = ThreadPoolExecutor(max_workers=num_workers)
                        futures = [executor.submit(process_item, item) for item in dataset]
                        try:
                            with tqdm(
                                total=len(dataset),
                                desc=config_desc,
                                leave=True,
                                ncols=120,
                                ascii=True,
                            ) as pbar:
                                pbar.refresh()
                                for future in as_completed(futures):
                                    record = future.result()
                                    results.append(record)
                                    config_results.append(record)
                                    pbar.update(1)
                                    update_pbar_metrics(pbar)
                        except KeyboardInterrupt:
                            cancel_event.set()
                            for f in futures:
                                f.cancel()
                            # Best-effort: stop waiting for queued work. In-flight calls may still take time.
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise
                        finally:
                            if not cancel_event.is_set():
                                executor.shutdown(wait=True, cancel_futures=False)
    except KeyboardInterrupt:
        cancelled = True


    # Aggregate metrics per (model, use_tools)
    metrics: Dict[str, Any] = {
        "_meta": {
            "temperature": temperature,
            "max_iterations": max_iterations,
            "trials": trials,
            "num_workers": num_workers,
            "reasoning_effort": reasoning_effort,
            "generated_at_unix": time.time(),
            "cancelled": cancelled,
            "completed_records": len(results),
        }
    }
    for model in models:
        for use_tools in use_tools_options:
            key = f"{model}|{'tools' if use_tools else 'no-tools'}"
            per_trial_metrics: Dict[str, Any] = {}
            trial_metrics_list: List[Dict[str, Any]] = []
            for trial in range(trials):
                subset = [
                    r
                    for r in results
                    if r["model"] == model and r["use_tools"] == use_tools and r.get("trial") == trial
                ]
                m = compute_metrics(subset)
                per_trial_metrics[str(trial)] = m
                trial_metrics_list.append(m)

            metrics[key] = {
                "per_trial": per_trial_metrics,
                "summary": summarize_trials(trial_metrics_list),
            }

    return results, metrics


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on real/fake image detection with/without tools."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset.")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of model names (e.g., gpt-5.1,gpt-5-mini).",
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default=None,
        help="Optional model override for the vision-only BAML step (defaults to each --models entry).",
    )
    parser.add_argument(
        "--structuring-model",
        type=str,
        default=None,
        help="Optional model override for the BAML structuring step (defaults to each --models entry).",
    )
    parser.add_argument(
        "--tools",
        type=str,
        default="both",
        choices=["both", "tools", "no-tools"],
        help="Run with tools, without tools, or both.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key. Defaults to environment variable (provider specific).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "openrouter"],
        help="Provider for LLM calls.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional API base URL (e.g., https://openrouter.ai/api/v1).",
    )
    parser.add_argument(
        "--referer",
        type=str,
        default=None,
        help="Optional HTTP-Referer header for OpenRouter.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional X-Title header for OpenRouter.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature for the analysis agent (use >0 only when studying stochasticity).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        help="Maximum agent iterations (tool + reasoning steps). Reduced from 30 to improve performance.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials per model/tool configuration (use >1 for mean/std estimates).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of samples to evaluate.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for dataset shuffling/reproducibility.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel workers per model/tool configuration. Use 1 to disable threading.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        help="Reasoning effort level for the model (e.g., 'low', 'medium', 'high').",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip dataset items whose images are missing instead of recording an error for them.",
    )
    parser.add_argument(
        "--enable-tool-cache",
        action="store_true",
        default=True,
        help="Enable caching of tool outputs and vision model outputs to avoid re-running expensive operations. Default: True.",
    )
    parser.add_argument(
        "--disable-tool-cache",
        dest="enable_tool_cache",
        action="store_false",
        help="Disable caching (tool outputs and vision model outputs).",
    )
    parser.add_argument(
        "--tool-cache-dir",
        type=str,
        default=None,
        help="Directory for cache files (tools and vision outputs). Default: .tool_cache in project root.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.jsonl",
        help="Path to write per-sample results.",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="eval_metrics.json",
        help="Path to write aggregated metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure tool cache (also caches vision model outputs)
    cache = ToolCache(
        cache_dir=args.tool_cache_dir,
        enabled=args.enable_tool_cache,
    )
    set_cache(cache)
    if args.enable_tool_cache:
        cache_stats = cache.get_stats()
        print(
            f"Cache (tools + vision): {cache_stats.get('cache_dir')} "
            f"({cache_stats.get('entry_count', 0)} entries, "
            f"{cache_stats.get('total_size_mb', 0):.1f} MB)"
        )
    else:
        print("Cache: disabled")

    # Clean up artifacts directory before starting evaluation
    print("Cleaning up artifacts directory...", end=" ", flush=True)
    clean_artifacts_dir()
    print("OK")

    if args.seed is not None:
        random.seed(args.seed)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = load_dataset(dataset_path)
    dataset, preflight_issues = preflight_dataset(
        dataset, skip_missing_images=args.skip_missing_images
    )
    if preflight_issues:
        print(
            f"WARN: Dataset preflight found {len(preflight_issues)} issue(s). "
            f"{'Skipping missing images.' if args.skip_missing_images else 'They will be recorded as errors.'}"
        )
    if args.shuffle:
        random.shuffle(dataset)
    if args.limit:
        dataset = dataset[: args.limit]

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ValueError("No models provided.")

    if args.tools == "both":
        use_tools_options = [True, False]
    elif args.tools == "tools":
        use_tools_options = [True]
    else:
        use_tools_options = [False]

    # Provider defaults
    base_url = args.base_url
    if args.provider == "openrouter" and not base_url:
        base_url = "https://openrouter.ai/api/v1"

    default_headers = build_headers(args.referer, args.title)
    if args.provider == "openai":
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    else:
        api_key = args.api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("API key is required. Set --api-key or the appropriate environment variable.")

    print(f"\nEvaluating {len(dataset)} samples × {len(models)} model(s) × {len(use_tools_options)} config(s) × {args.trials} trial(s)")
    print("=" * 80)
    
    results, metrics = evaluate(
        dataset=dataset,
        models=models,
        use_tools_options=use_tools_options,
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        trials=args.trials,
        num_workers=args.num_workers,
        reasoning_effort=args.reasoning_effort,
        vision_model=args.vision_model,
        structuring_model=args.structuring_model,
    )

    save_jsonl(Path(args.output), results)
    Path(args.metrics_output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Safeguard: ensure stdout is not closed (can happen with redirect_stdout in threads)
    try:
        if hasattr(sys.stdout, 'closed') and sys.stdout.closed:
            sys.stdout = sys.__stdout__
    except (AttributeError, ValueError):
        # If stdout is in a bad state, restore it
        sys.stdout = sys.__stdout__
    
    print("\n=== Evaluation Complete ===")
    print(f"Saved per-sample results to: {args.output}")
    print(f"Saved metrics to: {args.metrics_output}\n")
    for key, vals in metrics.items():
        if key.startswith("_"):
            continue
        summary = vals.get("summary", {})
        mean = summary.get("mean", {})
        mean_abstain = mean.get("abstain_rate", summary.get("mean_abstain_rate", 0.0))
        print(
            f"[{key}] overall_acc={mean.get('accuracy', 0.0):.3f} "
            f"acc_answered={mean.get('accuracy_answered', 0.0):.3f} "
            f"coverage={mean.get('coverage', 0.0):.3f} "
            f"bal_acc_ans={mean.get('balanced_accuracy_answered', mean.get('balanced_accuracy', 0.0)):.3f} "
            f"f1_fake_ans={mean.get('f1_fake_answered', mean.get('f1_fake', 0.0)):.3f} "
            f"f1_real_ans={mean.get('f1_real_answered', mean.get('f1_real', 0.0)):.3f} "
            f"review_rate={mean_abstain:.3f} "
            f"fake_slip={mean.get('fake_slip_rate', 0.0):.3f} "
            f"real_false_flag={mean.get('real_false_flag_rate', 0.0):.3f} "
            f"error_rate={summary.get('mean_error_rate', 0.0):.3f}"
        )


if __name__ == "__main__":
    main()
