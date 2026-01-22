"""
Data loader for forensic analysis results.

Loads JSONL result files and provides filtering/parsing utilities.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_available_results_files(results_dir: Optional[Path] = None) -> List[Path]:
    """
    Get all available JSONL result files from the results directory.
    
    Args:
        results_dir: Path to results directory. Defaults to 'results/' in project root.
    
    Returns:
        List of Path objects for each .jsonl file found.
    """
    if results_dir is None:
        # Default to results/ folder relative to this file
        results_dir = Path(__file__).parent.parent.parent / "results"
    
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    
    return sorted(results_dir.glob("*.jsonl"), key=lambda p: p.name)


def load_results(filepath: Path) -> List[Dict[str, Any]]:
    """
    Load results from a JSONL file.
    
    Args:
        filepath: Path to the JSONL file.
    
    Returns:
        List of result dictionaries.
    """
    results = []
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    with filepath.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # Add a unique key for each record (combination of id, model, use_tools, trial)
                record["_key"] = f"{record.get('id', '')}_{record.get('model', '')}_{record.get('use_tools', '')}_{record.get('trial', 0)}"
                results.append(record)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    
    return results


def filter_results(
    results: List[Dict[str, Any]],
    model: Optional[str] = None,
    use_tools: Optional[bool] = None,
    prediction: Optional[str] = None,
    label: Optional[str] = None,
    correct_only: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Filter results based on various criteria.
    
    Args:
        results: List of result dictionaries.
        model: Filter by model name (partial match).
        use_tools: Filter by whether tools were used.
        prediction: Filter by prediction verdict.
        label: Filter by ground truth label.
        correct_only: If True, only correct predictions; if False, only incorrect.
    
    Returns:
        Filtered list of results.
    """
    filtered = results
    
    if model is not None:
        filtered = [r for r in filtered if model.lower() in r.get("model", "").lower()]
    
    if use_tools is not None:
        filtered = [r for r in filtered if r.get("use_tools") == use_tools]
    
    if prediction is not None:
        filtered = [r for r in filtered if r.get("prediction", "").lower() == prediction.lower()]
    
    if label is not None:
        filtered = [r for r in filtered if r.get("label", "").lower() == label.lower()]
    
    if correct_only is not None:
        if correct_only:
            filtered = [r for r in filtered if r.get("prediction") == r.get("label")]
        else:
            filtered = [r for r in filtered if r.get("prediction") != r.get("label")]
    
    return filtered


def get_unique_values(results: List[Dict[str, Any]], field: str) -> List[Any]:
    """
    Get unique values for a specific field across all results.
    
    Args:
        results: List of result dictionaries.
        field: Field name to extract unique values from.
    
    Returns:
        Sorted list of unique values.
    """
    values = set()
    for r in results:
        val = r.get(field)
        if val is not None:
            values.add(val)
    return sorted(values, key=lambda x: str(x))


def get_result_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a summary of key fields from a result record.
    
    Args:
        result: A single result dictionary.
    
    Returns:
        Dictionary with key summary fields.
    """
    return {
        "id": result.get("id", "unknown"),
        "model": result.get("model", "unknown"),
        "use_tools": result.get("use_tools", False),
        "trial": result.get("trial", 0),
        "label": result.get("label", "unknown"),
        "prediction": result.get("prediction", "unknown"),
        "confidence": result.get("confidence", 0.0),
        "is_correct": result.get("prediction") == result.get("label"),
        "tool_count": len(result.get("tool_usage", [])),
        "latency_seconds": result.get("latency_seconds", 0.0),
    }


def extract_tool_info(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract detailed tool information from a result.
    
    Args:
        result: A single result dictionary.
    
    Returns:
        List of tool info dictionaries.
    """
    tool_results = result.get("tool_results", [])
    tool_details = result.get("tool_details", [])
    
    tools = []
    for i, tr in enumerate(tool_results):
        tool_info = {
            "name": tr.get("tool", "unknown"),
            "status": tr.get("status", "unknown"),
            "seconds": tr.get("seconds", 0.0),
            "error": tr.get("error"),
            "parsed": tr.get("parsed", {}),
        }
        tools.append(tool_info)
    
    return tools
