"""
SWGDE-inspired Image Authentication Report generator.

This produces a markdown "form" at the end of an analysis run, loosely based on
Appendix C of `docs/sw.md`, but adapted to DF3's actual toolchain and outputs.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # Optional dependency at runtime (DF3 normally has Pillow installed)
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkbox(checked: bool) -> str:
    return "[x]" if checked else "[ ]"


def _format_bytes(n: Optional[int]) -> str:
    if n is None:
        return "N/A"
    try:
        n_int = int(n)
    except Exception:
        return "N/A"
    # Human readable (base-1024)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(max(n_int, 0))
    for u in units:
        if size < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(size)} {u}"
            return f"{size:.2f} {u}"
        size /= 1024.0
    return f"{n_int} B"


def _format_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    try:
        s = float(seconds)
    except Exception:
        return "N/A"
    if s < 0:
        return "N/A"
    if s < 1:
        return f"{s:.3f}s"
    if s < 10:
        return f"{s:.2f}s"
    return f"{s:.1f}s"


def _safe_get(d: Any, *keys: str) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _as_tuple2(v: Any) -> Optional[Tuple[int, int]]:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            return int(v[0]), int(v[1])
        except Exception:
            return None
    return None


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _collect_tool_runs(tool_results: Any) -> Dict[str, List[Dict[str, Any]]]:
    """
    Normalize `analysis["tool_results"]` into {tool_name: [runs...]}.
    """
    runs: Dict[str, List[Dict[str, Any]]] = {}
    if not isinstance(tool_results, list):
        return runs
    for tr in tool_results:
        if not isinstance(tr, dict):
            continue
        name = tr.get("tool")
        if not isinstance(name, str) or not name:
            continue
        runs.setdefault(name, []).append(tr)
    return runs


def _last_run(runs: Dict[str, List[Dict[str, Any]]], tool_name: str) -> Optional[Dict[str, Any]]:
    lst = runs.get(tool_name)
    if not lst:
        return None
    return lst[-1]


def _last_parsed(runs: Dict[str, List[Dict[str, Any]]], tool_name: str) -> Optional[Dict[str, Any]]:
    lst = runs.get(tool_name)
    if not lst:
        return None
    for tr in reversed(lst):
        parsed = tr.get("parsed")
        if isinstance(parsed, dict) and parsed:
            return parsed
    # If parsed is empty/invalid, return empty dict to signal "ran but unparsed"
    parsed_any = lst[-1].get("parsed")
    return parsed_any if isinstance(parsed_any, dict) else None


def _tool_state(runs: Dict[str, List[Dict[str, Any]]], tool_name: str) -> Tuple[bool, str, Optional[float]]:
    """
    Returns (was_called, status, seconds).
    status is one of: completed|skipped|error|unknown|not_called
    """
    tr = _last_run(runs, tool_name)
    if not tr:
        return False, "not_called", None
    status = tr.get("status")
    if not isinstance(status, str) or not status:
        status = "unknown"
    seconds = tr.get("seconds")
    try:
        seconds_val = float(seconds) if seconds is not None else None
    except Exception:
        seconds_val = None
    return True, status, seconds_val


def _detect_depiction(mode: Optional[str]) -> str:
    if not mode:
        return "unknown"
    m = str(mode).upper().strip()
    if m in {"1", "L", "LA"}:
        return "monotone"
    return "color"


def _compression_artifacts_bucket(fmt: Optional[str]) -> str:
    """
    Present / Not Present / Unable to Determine
    """
    if not fmt:
        return "unable"
    f = str(fmt).upper().strip()
    if f in {"JPEG", "JPG", "WEBP"}:
        return "present"
    if f in {"PNG", "BMP", "TIFF", "TIF"}:
        return "not_present"
    return "unable"


def generate_image_authentication_report(
    analysis: Dict[str, Any],
    *,
    examiner: Optional[str] = None,
    lab_number: Optional[str] = None,
    case_number: Optional[str] = None,
    evidence_number: Optional[str] = None,
) -> str:
    """
    Generate a SWGDE-inspired Image Authentication Report in markdown.

    This is deterministic post-processing. It reflects what DF3 actually ran; if a
    tool/field wasn't available, the form marks it as "Not performed" / "N/A".
    """
    image_path = analysis.get("image_path")
    image_path_str = str(image_path) if image_path else ""
    p = Path(image_path_str) if image_path_str else None

    # Tool runs (agentic mode)
    tool_results = analysis.get("tool_results", [])
    runs = _collect_tool_runs(tool_results)

    # Pull metadata (if tool was called)
    md = _last_parsed(runs, "metadata") or {}
    md_file_size = _safe_get(md, "file", "size_bytes")
    md_format = _safe_get(md, "image", "format")
    md_mode = _safe_get(md, "image", "mode")
    md_size = _safe_get(md, "image", "size")

    # Fallbacks from filesystem / PIL
    file_exists = bool(image_path_str and os.path.exists(image_path_str))
    file_name = p.name if p else "N/A"
    file_size_bytes = None
    if isinstance(md_file_size, (int, float)):
        file_size_bytes = int(md_file_size)
    elif file_exists and p:
        try:
            file_size_bytes = int(p.stat().st_size)
        except Exception:
            file_size_bytes = None

    image_format = str(md_format) if isinstance(md_format, str) and md_format else None
    image_mode = str(md_mode) if isinstance(md_mode, str) and md_mode else None
    image_size = _as_tuple2(md_size)

    if (image_format is None or image_mode is None or image_size is None) and file_exists and Image is not None:
        try:
            with Image.open(image_path_str) as im:
                image_format = image_format or getattr(im, "format", None)
                image_mode = image_mode or getattr(im, "mode", None)
                image_size = image_size or (int(im.size[0]), int(im.size[1]))
        except Exception:
            pass

    depiction = _detect_depiction(image_mode)
    compression_bucket = _compression_artifacts_bucket(image_format)

    # SWGDE-style header fields (best-effort defaults)
    examiner_name = examiner or os.getenv("DF3_EXAMINER") or "DF3 Forensic Agent"
    lab_no = lab_number or os.getenv("DF3_LAB_NUMBER") or ""
    case_no = case_number or os.getenv("DF3_CASE_NUMBER") or ""
    evidence_no = evidence_number or os.getenv("DF3_EVIDENCE_NUMBER") or ""

    # Analysis conclusion fields
    verdict = str(analysis.get("verdict") or "uncertain").lower()
    confidence = analysis.get("confidence")
    try:
        conf_val = float(confidence) if confidence is not None else None
    except Exception:
        conf_val = None
    rationale = str(analysis.get("rationale") or "").strip()
    visual_description = str(analysis.get("visual_description") or "").strip()
    forensic_summary = str(analysis.get("forensic_summary") or "").strip()

    # Tool-derived highlights
    trufor = _last_parsed(runs, "perform_trufor") or {}
    trufor_prob = trufor.get("manipulation_probability")
    ela = _last_parsed(runs, "perform_ela") or {}
    ela_score = ela.get("ela_anomaly_score")
    ela_quality = ela.get("quality")
    qtz = _last_parsed(runs, "detect_jpeg_quantization") or {}
    qtz_q = _safe_get(qtz, "quality_estimates", "0", "estimated_quality")
    qtz_double = _safe_get(qtz, "sac_score", "score")
    freq = _last_parsed(runs, "analyze_frequency_domain") or {}
    freq_peak = freq.get("fft_peakiness")
    resid = _last_parsed(runs, "extract_residuals") or {}
    resid_p95 = resid.get("residual_energy_p95")

    # Metadata signals
    exif_present = bool(_safe_get(md, "exif", "present")) if md else False
    exif_summary = _safe_get(md, "exif", "summary") if isinstance(_safe_get(md, "exif", "summary"), dict) else {}
    exif_software = exif_summary.get("software") if isinstance(exif_summary, dict) else None
    xmp_present = bool(_safe_get(md, "xmp", "present")) if md else False
    icc_present = bool(_safe_get(md, "icc", "present")) if md else False
    c2pa_present = bool(_safe_get(md, "c2pa", "present")) if md else False
    c2pa_summary = _safe_get(md, "c2pa", "summary") if isinstance(_safe_get(md, "c2pa", "summary"), dict) else {}

    software_tags_present = bool(exif_software) or bool(c2pa_present) or bool(xmp_present)

    # Tool checklist (DF3 methods)
    tool_names = [
        "metadata",
        "perform_trufor",
        "perform_ela",
        "analyze_jpeg_compression",
        "detect_jpeg_quantization",
        "analyze_frequency_domain",
        "extract_residuals",
        "execute_python_code",
    ]
    called_tool_usage = analysis.get("tool_usage") or []
    called_tool_usage_list = [str(x) for x in called_tool_usage if isinstance(x, str)]
    called_tool_usage_list = _unique_preserve_order(called_tool_usage_list)

    # Build tool log rows
    tool_log_rows: List[str] = []
    tool_log_rows.append("| Tool | Status | Time | Notes |")
    tool_log_rows.append("| --- | --- | ---: | --- |")
    for tname in tool_names:
        was_called, status, seconds = _tool_state(runs, tname)
        note = ""
        if tname == "perform_trufor" and isinstance(trufor_prob, (int, float)):
            note = f"manipulation_probability={float(trufor_prob):.3f}"
        elif tname == "perform_ela" and isinstance(ela_score, (int, float)):
            q = f", quality={int(ela_quality)}" if isinstance(ela_quality, (int, float)) else ""
            note = f"ela_anomaly_score={float(ela_score):.3f}{q}"
        elif tname == "detect_jpeg_quantization" and isinstance(qtz_q, (int, float)):
            note = f"estimated_quality={int(qtz_q)}"
            if isinstance(qtz_double, (int, float)):
                note += f", sac_score={float(qtz_double):.3f}"
        elif tname == "analyze_frequency_domain" and isinstance(freq_peak, (int, float)):
            note = f"fft_peakiness={float(freq_peak):.3f}"
        elif tname == "extract_residuals" and isinstance(resid_p95, (int, float)):
            note = f"residual_energy_p95={float(resid_p95):.3f}"
        elif tname == "metadata" and md:
            note = "EXIF/XMP/ICC/C2PA scan"
        elif tname == "execute_python_code":
            # This tool returns free-form text, not JSON; "parsed" is often empty.
            raw = _safe_get(_last_run(runs, tname) or {}, "raw_truncated")
            if isinstance(raw, str) and raw.strip():
                note = "custom analysis output captured"
        if status == "skipped":
            reason = _safe_get(_last_parsed(runs, tname) or {}, "reason")
            if isinstance(reason, str) and reason:
                note = reason
        if not was_called and tname in called_tool_usage_list:
            # Defensive: should not happen, but keeps report sane.
            status = "unknown"
        tool_log_rows.append(f"| `{tname}` | {status} | {_format_seconds(seconds)} | {note} |")

    # Compression artifacts checkboxes
    ca_present = compression_bucket == "present"
    ca_not_present = compression_bucket == "not_present"
    ca_unable = compression_bucket == "unable"

    # Markdown report
    lines: List[str] = []
    lines.append("## DF3 Image Authentication Report (SWGDE-style)")
    lines.append("")
    lines.append("> Generated automatically from DF3 tool outputs. Fields not evaluated are marked accordingly.")
    lines.append("")

    # Examiner information
    lines.append("### Examiner Information")
    lines.append(f"- **Examiner:** {examiner_name}")
    lines.append(f"- **Date (UTC):** {_now_iso_utc()}")
    lines.append(f"- **Lab #:** {lab_no or '—'}")
    lines.append(f"- **Case #:** {case_no or '—'}")
    lines.append("- **Page:** 1 of 1")
    lines.append("")

    # Evidence information
    lines.append("### Evidence Information")
    lines.append(f"- **Evidence #:** {evidence_no or '—'}")
    lines.append(f"- **File Name:** {file_name}")
    lines.append("- **Media Type:**")
    lines.append(f"  - {_checkbox(True)} Digital Image")
    lines.append(f"  - {_checkbox(False)} Printed Still Photo")
    lines.append(f"- **Source:** {'—'}")
    lines.append(f"- **Mode:** {'—'}")
    lines.append("- **Depiction:**")
    lines.append(f"  - {_checkbox(depiction == 'color')} Color")
    lines.append(f"  - {_checkbox(depiction == 'monotone')} Monotone")
    lines.append("")

    # File properties
    lines.append("### File Properties")
    lines.append(f"- **File Path:** {image_path_str or 'N/A'}")
    lines.append(f"- **File Size:** {_format_bytes(file_size_bytes)}")
    if image_size:
        lines.append(f"- **Resolution:** {image_size[0]} × {image_size[1]} px")
    else:
        lines.append("- **Resolution:** N/A")
    lines.append(f"- **Format:** {image_format or 'N/A'}")
    lines.append(f"- **Color Mode:** {image_mode or 'N/A'}")
    lines.append("- **Compression Artifacts (container-level):**")
    lines.append(f"  - {_checkbox(ca_present)} Present")
    lines.append(f"  - {_checkbox(ca_not_present)} Not Present")
    lines.append(f"  - {_checkbox(ca_unable)} Unable to Determine")
    lines.append("")

    # Software & metadata analysis (DF3-adapted)
    md_called, md_status, _ = _tool_state(runs, "metadata")
    lines.append("### Software & Metadata Analysis (DF3-adapted)")
    lines.append("- **Software Tools Used:**")
    lines.append(f"  - {_checkbox(md_called and md_status == 'completed')} Metadata / EXIF / XMP / ICC / C2PA (`metadata`)")
    lines.append(f"  - {_checkbox(False)} Container/hex-level packaging analysis (not implemented)")
    lines.append(f"  - {_checkbox(False)} PRNU evaluation (not implemented)")
    lines.append(f"- **Software Tags Present:** {'Yes' if (md_called and software_tags_present) else ('No' if md_called else 'Not evaluated')}")
    if md_called:
        if exif_software:
            lines.append(f"  - **EXIF Software:** {exif_software}")
        lines.append(f"  - **EXIF present:** {exif_present}")
        lines.append(f"  - **XMP present:** {xmp_present}")
        lines.append(f"  - **ICC present:** {icc_present}")
        lines.append(f"  - **Content Credentials (C2PA) present:** {c2pa_present}")
        if c2pa_present and isinstance(c2pa_summary, dict) and c2pa_summary:
            cgs = c2pa_summary.get("claim_generators")
            if isinstance(cgs, list) and cgs:
                lines.append(f"  - **C2PA claim_generators:** {', '.join(str(x) for x in cgs[:6])}")
    lines.append("")

    # DF3 forensic methods checklist
    lines.append("### DF3 Forensic Methods Used")
    for tname, label in (
        ("perform_trufor", "TruFor neural forgery detection"),
        ("perform_ela", "Error Level Analysis (ELA)"),
        ("analyze_jpeg_compression", "JPEG compression analysis (Sac/JPEGness)"),
        ("detect_jpeg_quantization", "JPEG quantization + double-compression cues"),
        ("analyze_frequency_domain", "Frequency-domain analysis (FFT/DCT)"),
        ("extract_residuals", "Noise/residual analysis (DRUNet residual statistics)"),
        ("execute_python_code", "Custom Python analysis (sandboxed)"),
    ):
        was_called, status, _ = _tool_state(runs, tname)
        checked = was_called and status == "completed"
        suffix = ""
        if was_called and status != "completed":
            suffix = f" — {status}"
        elif not was_called:
            suffix = " — not performed"
        lines.append(f"- {_checkbox(checked)} **{label}** (`{tname}`){suffix}")
    lines.append("")

    # Observations
    lines.append("### Observations")
    if visual_description:
        lines.append("- **Visual (LLM):**")
        lines.append(f"  - {visual_description}")
    else:
        lines.append("- **Visual (LLM):** N/A")
    if forensic_summary:
        lines.append("- **Forensic summary (LLM):**")
        lines.append(f"  - {forensic_summary}")
    else:
        lines.append("- **Forensic summary (LLM):** N/A")
    lines.append("- **Key tool measurements (if available):**")
    if isinstance(trufor_prob, (int, float)):
        lines.append(f"  - TruFor manipulation_probability: {float(trufor_prob):.3f}")
    else:
        lines.append("  - TruFor manipulation_probability: N/A (tool not run)")
    if isinstance(ela_score, (int, float)):
        q = f" (quality={int(ela_quality)})" if isinstance(ela_quality, (int, float)) else ""
        lines.append(f"  - ELA anomaly_score: {float(ela_score):.3f}{q}")
    else:
        lines.append("  - ELA anomaly_score: N/A (tool not run / skipped)")
    if isinstance(qtz_q, (int, float)):
        lines.append(f"  - JPEG estimated_quality (from quant tables): {int(qtz_q)}")
    else:
        lines.append("  - JPEG estimated_quality: N/A")
    if isinstance(freq_peak, (int, float)):
        lines.append(f"  - FFT peakiness: {float(freq_peak):.3f}")
    else:
        lines.append("  - FFT peakiness: N/A")
    if isinstance(resid_p95, (int, float)):
        lines.append(f"  - Residual energy p95: {float(resid_p95):.3f}")
    else:
        lines.append("  - Residual energy p95: N/A")
    lines.append("")

    # Opinion / Conclusion
    lines.append("### Opinion")
    lines.append(f"- **Verdict:** {verdict.upper() if verdict else 'UNCERTAIN'}")
    if conf_val is not None:
        lines.append(f"- **Confidence (0–1):** {conf_val:.2f}")
    else:
        lines.append("- **Confidence (0–1):** N/A")
    if rationale:
        lines.append(f"- **Rationale:** {rationale}")
    else:
        lines.append("- **Rationale:** N/A")
    lines.append("")

    # Tool run log
    lines.append("### Tool Run Log")
    lines.extend(tool_log_rows)
    lines.append("")

    # Notes / limitations (short, SWGDE-aligned)
    lines.append("### Notes / Limitations")
    lines.append("- This report reflects DF3 automated analysis and should be independently reviewed by a trained examiner.")
    lines.append("- Absence of detected artifacts does not prove authenticity; some manipulations may be undetectable in a single image.")
    lines.append("- Tool outputs are best interpreted in context (image quality, compression history, and scene content).")

    return "\n".join(lines).strip() + "\n"


__all__ = ["generate_image_authentication_report"]


