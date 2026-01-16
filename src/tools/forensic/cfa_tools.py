"""
Popescu–Farid CFA Consistency Analyzer for the agent.

This tool analyzes Color Filter Array (CFA) demosaicing artifacts to detect
inconsistencies within an image. It is designed for SPLICE DETECTION and
SOURCE CONSISTENCY analysis, NOT for whole-image authenticity classification.

Scientific basis:
- Real camera images have CFA interpolation artifacts from Bayer demosaicing
- Spliced regions from different sources (AI, screenshots, different cameras)
  may have different or absent CFA patterns
- By analyzing the DISTRIBUTION of CFA metrics across windows, we can identify
  regions that are inconsistent with the rest of the image

What this tool DOES:
- Detects CFA pattern consistency across image regions
- Identifies outlier windows that differ from the image baseline
- Provides distribution analysis (unimodal vs bimodal)

What this tool does NOT do:
- Classify whole images as "authentic" or "fake"
- Work reliably on heavily compressed images
- Detect AI-generated images (use TruFor for that)

Supports two modes:
- analyze: run CFA consistency analysis on a single image
- calibrate: optional; build reference thresholds from a set of camera images
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from . import cfa


DEFAULT_PATTERN = "RGGB"
DEFAULT_WINDOW = 256
DEFAULT_TOP_K = 5
DEFAULT_OUTLIER_ZSCORE = 2.0  # Windows beyond this z-score are outliers
DEFAULT_TEXTURE_VAR_THRESHOLD = 1e-4  # Variance on [0,1] scale; below ~flat/no-texture


def _parse_request(input_str: str) -> Dict[str, Any]:
    """Parse JSON or treat input_str as image_path for analyze mode."""
    try:
        data = json.loads(input_str)
        if isinstance(data, dict):
            return data
        if isinstance(data, str):
            return {"mode": "analyze", "image_path": data}
    except Exception:
        pass
    return {"mode": "analyze", "image_path": input_str}


def _compute_stats(values: Sequence[float]) -> Dict[str, float]:
    """Compute basic statistics for a list of values."""
    arr = np.asarray(values, dtype=np.float64)
    # Robustness: ignore NaN/Inf (can happen if an upstream window produced non-finite M)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def _detect_bimodality(values: Sequence[float]) -> Dict[str, Any]:
    """
    Heuristic bimodality detection using the *bimodality coefficient* (BC).

    Returns:
        Dictionary with bimodality analysis results
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 10:
        return {
            "is_bimodal": False,
            "bimodality_coefficient": 0.0,
            "distribution_type": "insufficient_data",
            "note": "Need at least 10 windows for distribution analysis",
        }

    # Bimodality coefficient (Pfister et al., 2013):
    #   BC = (g1^2 + 1) / (g2 + 3 * (n-1)^2 / ((n-2)(n-3)))
    # where:
    #   g1 = skewness
    #   g2 = *excess* kurtosis (kurtosis - 3)
    # BC > ~0.555 suggests bimodality/multimodality (uniform distribution baseline).
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-10:
        return {
            "is_bimodal": False,
            "bimodality_coefficient": 0.0,
            "distribution_type": "constant",
            "note": "All values are nearly identical",
        }

    normalized = (arr - mean) / std
    skewness = float(np.mean(normalized ** 3))
    kurtosis = float(np.mean(normalized ** 4))

    # Excess kurtosis (Fisher definition)
    excess_kurtosis = float(kurtosis - 3.0)

    n = int(arr.size)
    # Finite-sample correction term; defined for n > 3.
    correction = 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3)) if n > 3 else 3.0
    denom = excess_kurtosis + float(correction)
    if abs(denom) < 1e-12:
        bc = 0.0
        note = "Degenerate kurtosis; bimodality coefficient set to 0.0"
    else:
        bc = float((skewness ** 2 + 1.0) / denom)
        note = None

    # Also report coefficient of variation (CV) for context (use abs(mean) for safety)
    cv = float(std / (abs(mean) + 1e-12))

    # Determine distribution type
    if bc > 0.6:
        dist_type = "bimodal"
        is_bimodal = True
    elif bc > 0.5:
        dist_type = "possibly_bimodal"
        is_bimodal = False
    elif cv > 0.3:
        dist_type = "high_variance"
        is_bimodal = False
    else:
        dist_type = "unimodal"
        is_bimodal = False

    return {
        "is_bimodal": is_bimodal,
        "bimodality_coefficient": bc,
        "coefficient_of_variation": float(cv),
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "distribution_type": dist_type,
        **({"note": note} if note else {}),
    }


def _robust_median_mad(arr: np.ndarray) -> Tuple[float, float]:
    """Return (median, MAD_scaled) with MAD scaled to match std for normal data."""
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    mad_scaled = mad * 1.4826 if mad > 0 else 1e-12
    return median, mad_scaled


def _compute_texture_variances(
    img: np.ndarray,
    positions: Sequence[Tuple[int, int, int, int]],
    channel_idx: int = 1,
) -> List[float]:
    """
    Compute a cheap per-window texture proxy (variance in a single channel).

    img is expected to be float RGB in [0,1] (as returned by cfa.load_rgb_image).
    """
    out: List[float] = []
    H, W, _ = img.shape
    c = int(channel_idx)
    c = 1 if c not in (0, 1, 2) else c
    for (y, x, h, w) in positions:
        y2 = min(y + h, H)
        x2 = min(x + w, W)
        patch = img[y:y2, x:x2, c]
        out.append(float(np.var(patch)))
    return out


def _log10_area_normalized_m(
    m_values: Sequence[float],
    positions: Sequence[Tuple[int, int, int, int]],
    ref_area: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Return log10(M_normalized) where:
        M_normalized = M / (area/ref_area)^3

    This makes M approximately window-size invariant for heuristic comparisons.
    """
    m = np.asarray(m_values, dtype=np.float64)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    areas = np.asarray([max(1, h * w) for (_, _, h, w) in positions], dtype=np.float64)
    scale = np.power(areas / float(ref_area), 3.0)
    return np.log10(m + eps) - np.log10(scale + eps)


def _classify_window_populations(
    m_values: Sequence[float],
    positions: Sequence[Tuple[int, int, int, int]],
    texture_vars: Sequence[float],
    texture_var_threshold: float,
) -> Dict[str, Any]:
    """
    Classify windows into populations based on M value magnitude.

    Real camera images typically show (with DC-excluded M values):
    - Low M (~0 to 1e4): Flat/uniform regions (sky, walls) - no texture to detect CFA
    - High M (>1e5): Textured regions with strong CFA signal

    This is content-dependent, not evidence of manipulation.
    Manipulation would show as textured regions WITHOUT CFA signal.
    """
    m = np.asarray(m_values, dtype=np.float64)
    if m.size == 0:
        return {"flat_regions": 0, "textured_regions": 0, "intermediate": 0, "flat_pct": 0.0, "textured_pct": 0.0}

    # Use texture proxy to identify clearly flat windows (content-driven; window-size invariant).
    t = np.asarray(texture_vars, dtype=np.float64)
    if t.size != m.size:
        t = np.zeros_like(m)
    flat_mask = t < float(texture_var_threshold)
    flat_count = int(np.sum(flat_mask))

    # Heuristic "strong CFA" gate: use area-normalized log(M) and a fixed reference threshold
    # anchored at DEFAULT_WINDOW. This avoids window-size breakage (e.g., window=128).
    #
    # NOTE: After fixing the M value computation to exclude DC component (which was
    # dominating the sum), typical M values are now:
    #   - Non-CFA synthetic: ~1e4 to 1e5
    #   - Real camera with CFA: ~1e5 to 1e7
    # The threshold is set at 1e5 to require meaningful CFA-specific signal.
    ref_area = float(DEFAULT_WINDOW * DEFAULT_WINDOW)
    log_m_norm = _log10_area_normalized_m(m_values, positions, ref_area=ref_area)
    # Equivalent to M >= 1e5 when area == DEFAULT_WINDOW^2 (DC-excluded M values)
    strong_threshold_log = float(np.log10(1e5))
    strong_mask = (~flat_mask) & (log_m_norm >= strong_threshold_log)
    textured_count = int(np.sum(strong_mask))

    intermediate_count = int(m.size) - flat_count - textured_count

    return {
        "flat_regions": flat_count,
        "textured_regions": textured_count,
        "intermediate": intermediate_count,
        "flat_pct": flat_count / int(m.size) * 100.0,
        "textured_pct": textured_count / int(m.size) * 100.0,
        # Extra diagnostics (non-breaking additions)
        "texture_var_threshold": float(texture_var_threshold),
        "texture_var_stats": _compute_stats(t.tolist()),
        "strong_cfa_threshold_log10_area_normalized": strong_threshold_log,
    }


def _window_brief(entry: Dict[str, Any], channel: str = "G") -> Dict[str, Any]:
    """Extract brief window info for a specific channel."""
    m = float(entry[channel]["M"])
    if not np.isfinite(m):
        m = 0.0
    return {
        "y": entry["y"],
        "x": entry["x"],
        "h": entry["h"],
        "w": entry["w"],
        "M_value": m,
    }


def _analyze(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run CFA consistency analysis on a single image."""
    image_path = params.get("image_path")
    if not image_path:
        return {"error": "image_path is required for analyze mode."}

    window = int(params.get("window", DEFAULT_WINDOW))
    pattern = params.get("pattern", DEFAULT_PATTERN)
    em_kwargs = params.get("em") or params.get("em_kwargs") or {}
    top_k = int(params.get("top_k", DEFAULT_TOP_K))
    channel = params.get("channel", "G").upper()  # Green channel is most reliable

    if channel not in ("R", "G", "B"):
        channel = "G"

    try:
        img = cfa.load_rgb_image(str(image_path))
    except Exception as e:
        return {"error": f"Failed to load image: {e}"}

    try:
        window_results = cfa.analyze_image_windows(
            img,
            window=window,
            pattern=pattern,
            em_kwargs=em_kwargs,
            channels=(channel,),
            return_maps=False,
        )
    except Exception as e:
        return {"error": f"CFA analysis failed: {e}"}

    if not window_results:
        return {"error": "No windows analyzed (image may be too small)."}

    # Extract M values and positions for the selected channel
    invalid_m_count = 0
    m_values: List[float] = []
    for r in window_results:
        m = float(r[channel]["M"])
        if not np.isfinite(m):
            invalid_m_count += 1
            m = 0.0
        m_values.append(m)
    positions = [(r["y"], r["x"], r["h"], r["w"]) for r in window_results]
    texture_var_threshold = float(
        params.get("texture_var_threshold", DEFAULT_TEXTURE_VAR_THRESHOLD)
    )
    texture_vars = _compute_texture_variances(img, positions, channel_idx=1)

    # Compute statistics
    stats = _compute_stats(m_values)
    ref_area = float(DEFAULT_WINDOW * DEFAULT_WINDOW)
    log_m_norm = _log10_area_normalized_m(m_values, positions, ref_area=ref_area)
    log_stats = _compute_stats(log_m_norm.tolist())

    # Analyze distribution
    # Use log10(area-normalized M) for stability across window sizes and heavy-tailed raw M.
    bimodality = _detect_bimodality(log_m_norm.tolist())

    # Classify windows into populations (flat vs textured-with-strong-CFA)
    populations = _classify_window_populations(
        m_values, positions, texture_vars, texture_var_threshold=texture_var_threshold
    )

    # Outlier detection on log10(area-normalized M), focusing on textured windows.
    outlier_z = float(params.get("outlier_zscore", DEFAULT_OUTLIER_ZSCORE))
    textured_mask = np.asarray(texture_vars, dtype=np.float64) >= float(texture_var_threshold)
    if int(np.sum(textured_mask)) >= 5:
        base_arr = log_m_norm[textured_mask]
    else:
        base_arr = log_m_norm
    med_log, mad_log = _robust_median_mad(base_arr)
    zscores = (log_m_norm - med_log) / (mad_log if mad_log > 0 else 1e-12)

    low_outliers: List[Dict[str, Any]] = []
    high_outliers: List[Dict[str, Any]] = []
    for z, val, pos, tvar in zip(zscores.tolist(), m_values, positions, texture_vars):
        if z < -outlier_z:
            interp = (
                "Textured window with unusually weak CFA response. This can happen with splices/mixed sources, "
                "heavy processing (resize/denoise/sharpen), or model artifacts. Treat as a localization hint, not proof."
                if tvar >= texture_var_threshold
                else "Low CFA response in a low-texture window (often expected for sky/walls)"
            )
            low_outliers.append(
                {
                    "y": pos[0],
                    "x": pos[1],
                    "h": pos[2],
                    "w": pos[3],
                    "M_value": float(val),
                    "z_score": float(z),
                    "texture_variance": float(tvar),
                    "interpretation": interp,
                }
            )
        elif z > outlier_z:
            high_outliers.append(
                {
                    "y": pos[0],
                    "x": pos[1],
                    "h": pos[2],
                    "w": pos[3],
                    "M_value": float(val),
                    "z_score": float(z),
                    "texture_variance": float(tvar),
                    "interpretation": (
                        "Unusually strong CFA response. This can occur with different capture pipelines, strong "
                        "sharpening/processing, or local periodic artifacts. Treat as a localization hint, not proof."
                    ),
                }
            )

    low_outliers.sort(key=lambda x: x["z_score"])
    high_outliers.sort(key=lambda x: -x["z_score"])

    # Get top windows by M value (strongest CFA signal)
    sorted_indices = np.argsort(m_values)[::-1]
    top_windows = [
        _window_brief(window_results[i], channel)
        for i in sorted_indices[:top_k]
    ]

    # Get bottom windows by M value (weakest CFA signal)
    bottom_windows = [
        _window_brief(window_results[i], channel)
        for i in sorted_indices[-top_k:][::-1]
    ]

    # Determine if image has CFA signal at all (heuristic; avoid window-size breakage).
    has_cfa_signal = bool(populations["textured_regions"] > 0)
    if (not has_cfa_signal) and len(window_results) == 1:
        # Single-window analyses can't express "consistency"; treat any non-flat, non-zero response as signal.
        if (populations["flat_regions"] == 0) and (m_values[0] > 0.0):
            has_cfa_signal = True
    textured_pct = float(populations["textured_pct"])

    # CFA consistency score (0-1): robust dispersion + low-outlier rate among textured windows.
    textured_idxs = [i for i, tv in enumerate(texture_vars) if tv >= texture_var_threshold]
    if len(textured_idxs) >= 2:
        log_textured = log_m_norm[textured_idxs]
        med_t, mad_t = _robust_median_mad(log_textured)
        z_t = (log_textured - med_t) / (mad_t if mad_t > 0 else 1e-12)
        low_frac = float(np.mean(z_t < -outlier_z)) if z_t.size else 0.0
        dispersion_score = float(1.0 / (1.0 + mad_t))
        cfa_score = float(np.clip(dispersion_score * (1.0 - low_frac), 0.0, 1.0))
        cfa_score_details = {
            "textured_window_count": int(len(textured_idxs)),
            "texture_var_threshold": float(texture_var_threshold),
            "log10_area_normalized_M_median": float(med_t),
            "log10_area_normalized_M_mad_scaled": float(mad_t),
            "low_outlier_fraction_textured": float(low_frac),
            "z_threshold": float(outlier_z),
            "dispersion_score_component": float(dispersion_score),
        }
    else:
        cfa_score = None
        cfa_score_details = {
            "textured_window_count": int(len(textured_idxs)),
            "texture_var_threshold": float(texture_var_threshold),
            "note": "Insufficient textured windows to compute a stable consistency score.",
        }

    # Generate interpretation based on content analysis
    if len(window_results) == 1 and has_cfa_signal:
        interpretation = (
            "Single-window CFA analysis: non-zero CFA response detected, but window-to-window consistency "
            "cannot be assessed with only one window."
        )
    elif not has_cfa_signal:
        interpretation = (
            "No strong CFA signal detected in any region. "
            "This can happen for screenshots/screen captures, heavily processed/resized images, synthetic imagery, "
            "or images dominated by flat/uniform content. Absence of CFA signal is not definitive on its own."
        )
    elif textured_pct > 50:
        interpretation = (
            f"Strong CFA response detected in {textured_pct:.0f}% of windows. "
            "This can be consistent with demosaicing-like artifacts from camera pipelines, but CFA-like periodicity "
            "can also be introduced by resizing/processing and some generative/upscaling workflows. "
            "Use the outlier windows for localization; do not treat this as an authenticity proof."
        )
    elif textured_pct > 20:
        interpretation = (
            f"CFA response detected in {textured_pct:.0f}% of windows (mostly textured regions). "
            "Remaining windows may be low-texture areas where CFA evidence is weak/absent. "
            "This pattern can be normal for photos with uniform backgrounds; CFA evidence is not definitive alone."
        )
    else:
        interpretation = (
            f"Weak/limited CFA response - only {textured_pct:.0f}% of windows show strong CFA. "
            "Image may be heavily processed/resized, low-texture, or partially synthetic. CFA evidence alone is not definitive."
        )

    # Build result
    result: Dict[str, Any] = {
        "tool": "perform_cfa_detection",
        "status": "completed",
        "image_path": str(image_path),
        "analysis_channel": channel,
        "window_size": window,
        "window_count": len(window_results),
        "pattern": pattern,

        # Main output: population analysis
        "has_cfa_signal": has_cfa_signal,
        "cfa_consistency_score": cfa_score,
        "cfa_consistency_details": cfa_score_details,
        "interpretation": interpretation,

        # Window populations
        "window_populations": {
            "textured_with_cfa": populations["textured_regions"],
            "flat_no_texture": populations["flat_regions"],
            "intermediate": populations["intermediate"],
            "textured_pct": populations["textured_pct"],
            "flat_pct": populations["flat_pct"],
            "diagnostics": {
                "texture_var_threshold": populations.get("texture_var_threshold"),
                "texture_var_stats": populations.get("texture_var_stats"),
                "strong_cfa_threshold_log10_area_normalized": populations.get("strong_cfa_threshold_log10_area_normalized"),
            },
        },

        # Distribution analysis (for advanced users)
        "distribution": {
            "type": bimodality["distribution_type"],
            "is_bimodal": bimodality["is_bimodal"],
            "bimodality_coefficient": bimodality["bimodality_coefficient"],
            "basis": "log10(area-normalized M) across windows",
            "note": "Bimodality can be NORMAL for photos with mixed content (flat + textured regions).",
        },

        # Statistics
        "m_value_stats": stats,
        "m_value_log10_area_normalized_stats": log_stats,

        # Reference windows
        "strongest_cfa_windows": top_windows,
        "weakest_cfa_windows": bottom_windows,

        # Outliers (potential splice indicators if TEXTURED yet weak CFA)
        "outliers": {
            "z_threshold": outlier_z,
            "z_score_basis": "robust z-score on log10(area-normalized M) from median/MAD",
            "low_M_count_total": int(len(low_outliers)),
            "high_M_count_total": int(len(high_outliers)),
            "low_M": low_outliers[: top_k],
            "high_M": high_outliers[: top_k],
        },

        "note": (
            "CFA analysis detects demosaicing artifacts from camera sensors. "
            "Flat regions (sky, walls) naturally have weak/no CFA signal. "
            "Look for TEXTURED regions with unusually weak CFA response (especially if spatially clustered). "
            "This tool is for localization/consistency, not whole-image authenticity; use TruFor and other signals too."
        ),
    }

    # Non-breaking diagnostic for rare numeric issues
    if invalid_m_count:
        result["diagnostics"] = {
            "invalid_m_count": int(invalid_m_count),
            "invalid_m_note": "Some windows produced non-finite M values (NaN/Inf) and were clamped to 0.0.",
        }

    return result


def _calibrate(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calibrate reference statistics from a set of camera images."""
    neg_dir = params.get("neg_dir") or params.get("ref_dir")
    if not neg_dir:
        return {"error": "neg_dir (or ref_dir) is required for calibrate mode."}

    window = int(params.get("window", DEFAULT_WINDOW))
    pattern = params.get("pattern", DEFAULT_PATTERN)
    em_kwargs = params.get("em") or params.get("em_kwargs") or {}
    save_to = params.get("save_to") or params.get("output")

    neg_files = cfa.list_image_files(str(neg_dir))
    if not neg_files:
        return {"error": f"No images found in directory: {neg_dir}"}

    # Collect M values from all reference images
    all_m_values: Dict[str, List[float]] = {"R": [], "G": [], "B": []}

    for path in neg_files:
        try:
            img = cfa.load_rgb_image(str(path))
            window_results = cfa.analyze_image_windows(
                img,
                window=window,
                pattern=pattern,
                em_kwargs=em_kwargs,
                return_maps=False,
            )
            for r in window_results:
                all_m_values["R"].append(float(r["R"]["M"]))
                all_m_values["G"].append(float(r["G"]["M"]))
                all_m_values["B"].append(float(r["B"]["M"]))
        except Exception:
            continue  # Skip problematic images

    if not all_m_values["G"]:
        return {"error": "No valid windows collected from reference images."}

    # Compute reference statistics for each channel
    reference_stats = {
        c: _compute_stats(vals) for c, vals in all_m_values.items()
    }

    payload = {
        "reference_stats": reference_stats,
        "pattern": pattern,
        "window": window,
        "em_params": em_kwargs,
        "num_images": len(neg_files),
        "num_windows": len(all_m_values["G"]),
    }

    if save_to:
        Path(save_to).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["saved_to"] = str(save_to)

    return payload


def perform_cfa_detection(input_str: str) -> str:
    """
    LangChain tool entrypoint for CFA consistency analysis.

    This tool analyzes CFA (Color Filter Array) demosaicing patterns to detect
    INCONSISTENCIES within an image. It is designed for splice detection and
    source consistency analysis.

    Input (JSON):
      - mode: "analyze" (default) or "calibrate"
      - image_path: required for analyze
      - window: int (default 256)
      - pattern: Bayer pattern (default RGGB)
      - channel: which channel to analyze (default "G" - green is most reliable)
      - em / em_kwargs: dict for EM params (N, sigma0, p0, max_iter, tol, seed)
      - top_k: int (default 5) - number of top/outlier windows to return
      - outlier_zscore: float (default 2.0) - z-score threshold for outlier detection
      - neg_dir/ref_dir: required for calibrate mode
      - save_to: optional path to write reference stats JSON (calibrate)

    Output:
      - cfa_consistency_score: 0-1 score (higher = more consistent)
      - distribution: analysis of M value distribution (unimodal/bimodal)
      - outliers: windows with unusually low/high CFA patterns
      - interpretation: human-readable summary
    """
    params = _parse_request(input_str)
    mode = params.get("mode", "analyze").lower()

    # Support legacy "detect" mode name
    if mode in ("detect", "visualize"):
        mode = "analyze"

    if mode == "calibrate":
        result = _calibrate(params)
    elif mode == "analyze":
        result = _analyze(params)
    else:
        result = {"error": "mode must be 'analyze' or 'calibrate'."}

    try:
        return json.dumps(result, indent=2)
    except Exception:
        # Fallback in case something is not JSON-serializable
        return json.dumps({"error": "Failed to serialize result."}, indent=2)


__all__ = ["perform_cfa_detection"]
