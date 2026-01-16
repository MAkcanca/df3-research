"""
Simple bare LLM evaluation - no agentic flows, no complex prompts.

This script makes a simple single LLM call with just the image and
"Is this image fake or real?" prompt, similar to ChatGPT behavior.
"""

import argparse
import json
import logging
import os
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

import base64  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


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


def encode_image(image_path: str) -> Tuple[str, str]:
    """Encode image to base64 and determine MIME type."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Determine MIME type from extension
    image_ext = Path(image_path).suffix.lower()
    if image_ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif image_ext == ".png":
        mime_type = "image/png"
    elif image_ext == ".webp":
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"  # default
    
    return encoded_string, mime_type


def extract_verdict(response_text: str) -> str:
    """Extract verdict from response text, looking for structured format first, then keywords."""
    if not response_text:
        return "uncertain"
    
    text_lower = response_text.lower()
    
    # First, look for structured VERDICT: format
    verdict_match = re.search(r'verdict\s*:\s*(fake|real|uncertain)', text_lower, re.IGNORECASE)
    if verdict_match:
        verdict_str = verdict_match.group(1).strip().lower()
        if verdict_str in ("fake", "real", "uncertain"):
            return verdict_str
    
    # Fallback: search for keywords in the text (not just exact matches)
    fake_phrases = [
        r'\bai[-\s]?generated\b',
        r'\bsynthetic\b',
        r'\bdeepfake\b',
        r'\bfake\b',
        r'\bmanipulated\b',
        r'\btampered\b',
        r'\bcomposited\b',
        r'\bnot real\b',
        r'\bnot genuine\b',
        r'\bnot authentic\b',
    ]
    
    real_phrases = [
        r'\bgenuine\b',
        r'\bauthentic\b',
        r'\breal\b',
        r'\bunmanipulated\b',
        r'\bnot fake\b',
        r'\bnot ai[-\s]?generated\b',
        r'\bnot synthetic\b',
    ]
    
    # Check for fake indicators (stronger weight if near the end)
    fake_score = 0
    for pattern in fake_phrases:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            # Give more weight to matches in the last 30% of the text
            pos_ratio = match.start() / len(text_lower) if text_lower else 0
            weight = 2.0 if pos_ratio > 0.7 else 1.0
            fake_score += weight
    
    # Check for real indicators
    real_score = 0
    for pattern in real_phrases:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            pos_ratio = match.start() / len(text_lower) if text_lower else 0
            weight = 2.0 if pos_ratio > 0.7 else 1.0
            real_score += weight
    
    # Also check for phrases that indicate uncertainty
    uncertain_phrases = [
        r'\buncertain\b',
        r'\binconclusive\b',
        r'\bunclear\b',
        r'\bcannot determine\b',
        r'\bcannot be certain\b',
        r'\bimpossible to tell\b',
    ]
    uncertain_score = sum(1 for pattern in uncertain_phrases if re.search(pattern, text_lower, re.IGNORECASE))
    
    # Decide based on scores
    if uncertain_score > 0 and fake_score == 0 and real_score == 0:
        return "uncertain"
    
    if fake_score > real_score and fake_score > 0:
        return "fake"
    elif real_score > fake_score and real_score > 0:
        return "real"
    
    # If scores are equal or both zero, return uncertain
    return "uncertain"


def compute_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute accuracy and balanced metrics, tracking abstains/errors explicitly."""
    tp = fp = tn = fn = 0
    abstain = error_count = 0
    confidences: List[float] = []
    label_counts = {"real": 0, "fake": 0}
    abstain_by_label = {"real": 0, "fake": 0}
    latencies: List[float] = []
    prompt_tokens: List[int] = []
    completion_tokens: List[int] = []
    total_tokens: List[int] = []

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
            # Track token usage
            token_usage = rec.get("token_usage", {})
            if isinstance(token_usage, dict):
                if isinstance(token_usage.get("prompt_tokens"), int):
                    prompt_tokens.append(token_usage["prompt_tokens"])
                if isinstance(token_usage.get("completion_tokens"), int):
                    completion_tokens.append(token_usage["completion_tokens"])
                if isinstance(token_usage.get("total_tokens"), int):
                    total_tokens.append(token_usage["total_tokens"])

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
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

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
    fake_slip_rate = fn / support_fake if support_fake else 0.0
    real_false_flag_rate = fp / support_real if support_real else 0.0
    fake_catch_rate = tp / support_fake if support_fake else 0.0
    real_pass_rate = tn / support_real if support_real else 0.0
    
    # Token usage statistics
    avg_prompt_tokens = sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else 0.0
    avg_completion_tokens = sum(completion_tokens) / len(completion_tokens) if completion_tokens else 0.0
    avg_total_tokens = sum(total_tokens) / len(total_tokens) if total_tokens else 0.0
    total_prompt_tokens = sum(prompt_tokens) if prompt_tokens else 0
    total_completion_tokens = sum(completion_tokens) if completion_tokens else 0
    total_all_tokens = sum(total_tokens) if total_tokens else 0

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
        "token_usage": {
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": avg_completion_tokens,
            "avg_total_tokens": avg_total_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_all_tokens": total_all_tokens,
            "samples_with_tokens": len(total_tokens),
        },
    }


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


def build_headers(referer: Optional[str], title: Optional[str]) -> Optional[Dict[str, str]]:
    """Optional headers for OpenRouter."""
    headers: Dict[str, str] = {}
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers or None


def evaluate(
    dataset: List[Dict[str, Any]],
    models: List[str],
    api_key: str,
    base_url: Optional[str],
    default_headers: Optional[Dict[str, str]],
    temperature: float,
    trials: int,
    num_workers: int,
    reasoning_effort: str,
    max_tokens: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run evaluation for all models and trials."""
    results: List[Dict[str, Any]] = []

    for model in models:
        for trial in range(trials):
            # Create LLM client for this model
            llm_kwargs = {
                "model": model,
                "temperature": temperature,
                "api_key": api_key,
                "reasoning_effort": reasoning_effort,
            }
            if base_url:
                llm_kwargs["base_url"] = base_url
            if default_headers:
                llm_kwargs["default_headers"] = default_headers
            if max_tokens is not None:
                llm_kwargs["max_tokens"] = max_tokens
            llm = ChatOpenAI(**llm_kwargs)

            def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
                record = {
                    "id": item["id"],
                    "model": model,
                    "trial": trial,
                    "label": item["label"],
                    "image": item["image"],
                    "meta": item.get("meta", {}),
                    "run_config": {
                        "temperature": temperature,
                    },
                }
                start = time.time()
                try:
                    if not os.path.exists(item["image"]):
                        raise FileNotFoundError(f"Image not found: {item['image']}")
                    
                    # Encode image
                    base64_image, mime_type = encode_image(item["image"])
                    
                    # Simple prompt - just like ChatGPT, but with structured output request
                    prompt = """Is this image fake or real? Evaluate and reason.

After your reasoning, write your final verdict as:
VERDICT: FAKE
or
VERDICT: REAL

If you are truly uncertain, write:
VERDICT: UNCERTAIN"""
                    
                    # Make simple LLM call
                    messages = [
                        HumanMessage(content=[
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                            }
                        ])
                    ]
                    
                    response = llm.invoke(messages)
                    latency = time.time() - start
                    
                    # Extract response text
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    
                    # Extract token usage from response metadata
                    token_usage = {}
                    if hasattr(response, 'response_metadata') and response.response_metadata:
                        usage = response.response_metadata.get('token_usage', {})
                        if usage:
                            token_usage = {
                                "prompt_tokens": usage.get("prompt_tokens", 0),
                                "completion_tokens": usage.get("completion_tokens", 0),
                                "total_tokens": usage.get("total_tokens", 0),
                            }
                    
                    # Try to extract verdict from response
                    verdict = extract_verdict(response_text)
                    
                    # Try to extract confidence (look for patterns like "confidence: 0.8" or "80%")
                    confidence = 0.5  # default
                    # Look for confidence patterns
                    conf_match = re.search(r'confidence[:\s]+([0-9.]+)', response_text, re.IGNORECASE)
                    if conf_match:
                        try:
                            conf_val = float(conf_match.group(1))
                            if conf_val > 1.0:
                                conf_val = conf_val / 100.0  # Convert percentage to 0-1
                            confidence = max(0.0, min(1.0, conf_val))
                        except ValueError:
                            pass
                    else:
                        # Look for percentage patterns
                        pct_match = re.search(r'([0-9.]+)%', response_text)
                        if pct_match:
                            try:
                                confidence = max(0.0, min(1.0, float(pct_match.group(1)) / 100.0))
                            except ValueError:
                                pass

                    record.update(
                        {
                            "prediction": verdict,
                            "confidence": confidence,
                            "raw_response": response_text,
                            "latency_seconds": latency,
                            "token_usage": token_usage,
                        }
                    )
                except Exception as exc:
                    error_msg = str(exc)
                    error_type = "missing_image" if isinstance(exc, FileNotFoundError) else "exception"
                    record.update(
                        {
                            "prediction": "uncertain",
                            "confidence": 0.0,
                            "error": error_msg,
                            "error_type": error_type,
                        }
                    )
                    logger.error(
                        f"[ERROR] Failed to analyze item {item['id']} "
                        f"(model={model}, trial={trial}): {error_msg}",
                        exc_info=True
                    )
                return record

            if num_workers <= 1:
                for item in dataset:
                    results.append(process_item(item))
            else:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    future_to_item = {executor.submit(process_item, item): item for item in dataset}
                    for future in as_completed(future_to_item):
                        results.append(future.result())

    # Aggregate metrics per model
    metrics: Dict[str, Any] = {
        "_meta": {
            "temperature": temperature,
            "trials": trials,
            "num_workers": num_workers,
            "max_tokens": max_tokens,
            "generated_at_unix": time.time(),
        }
    }
    for model in models:
        key = model
        per_trial_metrics: Dict[str, Any] = {}
        trial_metrics_list: List[Dict[str, Any]] = []
        for trial in range(trials):
            subset = [
                r
                for r in results
                if r["model"] == model and r.get("trial") == trial
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
        description="Evaluate bare LLMs on real/fake image detection (no agentic flows, simple prompt)."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset.")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of model names (e.g., gpt-5.1,gpt-5-mini).",
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
        help="LLM temperature.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials per model (use >1 for mean/std estimates).",
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
        help="Parallel workers per model. Use 1 to disable threading.",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip dataset items whose images are missing instead of recording an error for them.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bare_llm_results.jsonl",
        help="Path to write per-sample results.",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="bare_llm_metrics.json",
        help="Path to write aggregated metrics.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        help="Reasoning effort level for the model (e.g., 'low', 'medium', 'high').",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate in the response.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

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
        logger.warning(
            f"Dataset preflight found {len(preflight_issues)} issue(s). "
            f"{'Skipping missing images.' if args.skip_missing_images else 'They will be recorded as errors.'}"
        )
    if args.shuffle:
        random.shuffle(dataset)
    if args.limit:
        dataset = dataset[: args.limit]

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ValueError("No models provided.")

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

    results, metrics = evaluate(
        dataset=dataset,
        models=models,
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
        temperature=args.temperature,
        trials=args.trials,
        num_workers=args.num_workers,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
    )

    save_jsonl(Path(args.output), results)
    Path(args.metrics_output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

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


