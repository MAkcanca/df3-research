"""
Deterministic tool-output normalization for research + court-facing reports.

Goal:
- Parse raw tool JSON outputs (strings).
- Extract a small, stable set of evidence fields per tool.
- Produce a concise, consistent summary with explicit caveats.

This module MUST NOT call any LLMs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON, tolerating extra text around the object."""
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(t[start : end + 1])
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
    return None


def _fmt_float(val: Any, ndigits: int = 3) -> str:
    try:
        f = float(val)
        return f"{f:.{ndigits}f}"
    except Exception:
        return "n/a"


def normalize_tool_outputs(tool_outputs: Dict[str, str]) -> Dict[str, Any]:
    """
    Normalize raw tool outputs.

    Args:
        tool_outputs: mapping of tool_name -> raw output string (expected JSON).

    Returns:
        Dict with:
          - evidence: per-tool extracted fields
          - summary_lines: deterministic bullets
          - caveats: deterministic caveats
          - parse_errors: tools whose output couldn't be parsed
    """
    evidence: Dict[str, Any] = {}
    summary_lines: List[str] = []
    caveats: List[str] = []
    parse_errors: List[str] = []

    for tool_name, raw in (tool_outputs or {}).items():
        parsed = _safe_json_loads(raw)
        if parsed is None:
            parse_errors.append(tool_name)
            evidence[tool_name] = {"status": "unparsed", "raw_preview": str(raw)[:200]}
            summary_lines.append(f"- {tool_name}: status=unparsed")
            continue

        status = parsed.get("status", "unknown")
        err = parsed.get("error")

        # Extract small, stable subsets per tool
        if tool_name == "perform_trufor" or parsed.get("tool") == "perform_trufor":
            mp = parsed.get("manipulation_probability")
            ds = parsed.get("detection_score")
            evidence[tool_name] = {
                "status": status,
                "manipulation_probability": mp,
                "detection_score": ds,
                "error": err,
            }
            summary_lines.append(
                f"- perform_trufor: status={status}, manipulation_probability={_fmt_float(mp, 3)}, detection_score={_fmt_float(ds, 3)}"
            )
            caveats.append(
                "TruFor is primarily a manipulation/splice detector. Low manipulation_probability does NOT rule out fully synthetic (AI-generated) images."
            )

        elif tool_name == "perform_ela" or parsed.get("tool") == "perform_ela":
            score = parsed.get("ela_anomaly_score")
            q = parsed.get("quality")
            evidence[tool_name] = {
                "status": status,
                "ela_anomaly_score": score,
                "quality": q,
                "error": err,
            }
            summary_lines.append(
                f"- perform_ela: status={status}, anomaly_score={_fmt_float(score, 3)}, quality={q if q is not None else 'n/a'}"
            )
            caveats.append(
                "ELA is a compression-consistency heuristic; interpret in context. Uniform ELA does not prove authenticity, and strong ELA anomalies can also arise from benign processing."
            )

        elif tool_name == "detect_jpeg_quantization" or parsed.get("tool") == "detect_jpeg_quantization":
            is_jpeg = parsed.get("is_jpeg")
            sac = parsed.get("sac_score")
            coef_src = parsed.get("coefficient_source")
            q_est = None
            try:
                q_est = (parsed.get("quality_estimates") or {}).get("0")
            except Exception:
                q_est = None
            evidence[tool_name] = {
                "status": status,
                "is_jpeg": is_jpeg,
                "quality_estimate_primary": q_est,
                "sac_score": sac,
                "coefficient_source": coef_src,
                "error": err,
            }
            summary_lines.append(
                f"- detect_jpeg_quantization: status={status}, is_jpeg={is_jpeg}, quality_estimate_primary={q_est}, sac_score={_fmt_float(sac, 3)}"
            )
            if is_jpeg is False:
                caveats.append("JPEG quantization analysis is not applicable to non-JPEG inputs (PNG/WebP/etc.).")

        elif tool_name == "analyze_jpeg_compression" or parsed.get("tool") == "analyze_jpeg_compression":
            is_jpeg = parsed.get("is_jpeg")
            sac = parsed.get("sac_score")
            fmt = parsed.get("format")
            evidence[tool_name] = {
                "status": status,
                "format": fmt,
                "is_jpeg": is_jpeg,
                "sac_score": sac,
                "error": err,
            }
            summary_lines.append(
                f"- analyze_jpeg_compression: status={status}, format={fmt}, is_jpeg={is_jpeg}, sac_score={_fmt_float(sac, 3)}"
            )

        elif tool_name == "analyze_frequency_domain" or parsed.get("tool") == "analyze_frequency_domain":
            decay = parsed.get("fft_radial_decay")
            peak = parsed.get("fft_peakiness")
            dct_mean = parsed.get("dct_mean")
            evidence[tool_name] = {
                "status": status,
                "fft_radial_decay": decay,
                "fft_peakiness": peak,
                "dct_mean": dct_mean,
                "error": err,
            }
            summary_lines.append(
                f"- analyze_frequency_domain: status={status}, fft_radial_decay={_fmt_float(decay, 3)}, fft_peakiness={_fmt_float(peak, 3)}, dct_mean={_fmt_float(dct_mean, 3)}"
            )
            caveats.append(
                "Frequency-domain features can be influenced by resizing, compression, camera pipelines, and content. Treat as weak/auxiliary evidence unless corroborated."
            )

        elif tool_name == "extract_residuals" or parsed.get("tool") == "extract_residuals":
            e95 = parsed.get("residual_energy_p95")
            estd = parsed.get("residual_std")
            ekurt = parsed.get("residual_kurtosis")
            evidence[tool_name] = {
                "status": status,
                "residual_energy_p95": e95,
                "residual_std": estd,
                "residual_kurtosis": ekurt,
                "error": err,
            }
            summary_lines.append(
                f"- extract_residuals: status={status}, residual_energy_p95={_fmt_float(e95, 3)}, residual_std={_fmt_float(estd, 3)}, residual_kurtosis={_fmt_float(ekurt, 3)}"
            )
            caveats.append(
                "Residual statistics depend on denoiser behavior, image content, and compression; treat as auxiliary evidence and avoid overclaiming."
            )

        elif tool_name == "perform_cfa_detection":
            score = parsed.get("cfa_consistency_score")
            dist = parsed.get("distribution")
            evidence[tool_name] = {
                "status": status,
                "cfa_consistency_score": score,
                "distribution": dist,
                "error": err,
            }
            summary_lines.append(
                f"- perform_cfa_detection: status={status}, cfa_consistency_score={_fmt_float(score, 3)}"
            )
            caveats.append(
                "CFA consistency is mainly useful for splice detection on demosaiced photos; many synthetic images or heavily processed images may not exhibit meaningful CFA artifacts."
            )

        else:
            # Generic fallback: keep status/error and a small preview.
            evidence[tool_name] = {
                "status": status,
                "error": err,
                "raw_preview": str(raw)[:200],
            }
            summary_lines.append(f"- {tool_name}: status={status}")

    # De-duplicate caveats deterministically
    deduped_caveats = []
    seen = set()
    for c in caveats:
        if c not in seen:
            seen.add(c)
            deduped_caveats.append(c)

    return {
        "evidence": evidence,
        "summary_lines": summary_lines,
        "caveats": deduped_caveats,
        "parse_errors": parse_errors,
    }

