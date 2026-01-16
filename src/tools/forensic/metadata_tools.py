"""
Metadata extraction tool.

Extracts common image metadata for forensic analysis:
- EXIF (camera/device/software/timestamps/GPS when present)
- XMP (including potential C2PA-related XMP references)
- ICC profile presence (and best-effort description)
- C2PA / Content Credentials (best-effort via c2patool when available)

The tool returns a JSON string compatible with the agent's tool pipeline.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ExifTags


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _truncate_text(s: Optional[str], max_chars: int) -> Optional[str]:
    if s is None:
        return None
    t = str(s)
    if len(t) <= max_chars:
        return t
    return t[: max(0, max_chars - 3)] + "..."


def _truncate_mapping(d: Dict[str, Any], max_items: int) -> Dict[str, Any]:
    if len(d) <= max_items:
        return d
    out: Dict[str, Any] = {}
    for i, (k, v) in enumerate(d.items()):
        if i >= max_items:
            break
        out[k] = v
    out["__truncated__"] = True
    out["__total_items__"] = len(d)
    return out


def _read_bytes_prefix(path: str, max_bytes: int) -> bytes:
    with open(path, "rb") as f:
        return f.read(max_bytes)


def _looks_like_json(s: str) -> bool:
    t = (s or "").lstrip()
    return t.startswith("{") or t.startswith("[")


def _extract_first_json_object(text: str) -> Optional[dict]:
    """
    Extract the first JSON object embedded in a larger text blob.
    This is more robust than naive slicing if the tool prints logs around JSON.
    """
    if not text:
        return None
    s = str(text)
    in_str = False
    esc = False
    depth = 0
    start: Optional[int] = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
            continue
        if ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = s[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    return None
    return None


def _jsonify_value(v: Any) -> Any:
    """Convert EXIF/XMP-derived values into JSON-friendly types."""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, bytes):
        return f"<bytes:{len(v)}>"
    # Pillow rationals
    try:
        # IFDRational supports float()
        if hasattr(v, "numerator") and hasattr(v, "denominator"):
            return float(v)
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        # Limit long sequences
        seq = list(v[:32]) if len(v) > 32 else list(v)
        return [_jsonify_value(x) for x in seq]
    if isinstance(v, dict):
        return {str(k): _jsonify_value(val) for k, val in list(v.items())[:128]}
    return str(v)


def _gps_to_decimal(gps_info: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Convert EXIF GPSInfo dict to decimal lat/lon/alt when possible.
    """
    try:
        lat_ref = gps_info.get("GPSLatitudeRef")
        lat = gps_info.get("GPSLatitude")
        lon_ref = gps_info.get("GPSLongitudeRef")
        lon = gps_info.get("GPSLongitude")
        alt = gps_info.get("GPSAltitude")

        def _dms_to_deg(dms: Any) -> Optional[float]:
            if not isinstance(dms, (list, tuple)) or len(dms) != 3:
                return None
            d, m, s = dms
            try:
                return float(d) + float(m) / 60.0 + float(s) / 3600.0
            except Exception:
                return None

        lat_deg = _dms_to_deg(lat)
        lon_deg = _dms_to_deg(lon)
        if isinstance(lat_ref, str) and lat_deg is not None:
            if lat_ref.upper().strip() == "S":
                lat_deg = -lat_deg
        if isinstance(lon_ref, str) and lon_deg is not None:
            if lon_ref.upper().strip() == "W":
                lon_deg = -lon_deg
        alt_m = None
        try:
            if alt is not None:
                alt_m = float(alt)
        except Exception:
            alt_m = None
        return lat_deg, lon_deg, alt_m
    except Exception:
        return None, None, None


def _extract_exif(img: Image.Image) -> Dict[str, Any]:
    exif_obj = None
    try:
        exif_obj = img.getexif()
    except Exception:
        exif_obj = None

    if not exif_obj:
        return {"present": False, "summary": {}, "tags": {}, "tag_count": 0}

    # Build tag map with human-readable names
    tags: Dict[str, Any] = {}
    gps_raw: Optional[Dict[str, Any]] = None
    for tag_id, value in exif_obj.items():
        name = ExifTags.TAGS.get(tag_id, f"Tag_{tag_id}")
        if name == "GPSInfo" and isinstance(value, dict):
            gps_raw = {
                ExifTags.GPSTAGS.get(k, str(k)): v for k, v in value.items()
            }
            tags[name] = _jsonify_value(gps_raw)
        else:
            tags[name] = _jsonify_value(value)

    # Curated summary fields
    def _get(*names: str) -> Optional[Any]:
        for n in names:
            if n in tags and tags[n] not in (None, "", {}):
                return tags[n]
        return None

    gps_summary: Dict[str, Any] = {}
    if gps_raw:
        lat, lon, alt = _gps_to_decimal(gps_raw)
        if lat is not None and lon is not None:
            gps_summary["latitude"] = round(float(lat), 7)
            gps_summary["longitude"] = round(float(lon), 7)
        if alt is not None:
            gps_summary["altitude_m"] = float(alt)

    summary: Dict[str, Any] = {
        "make": _get("Make"),
        "model": _get("Model"),
        "software": _get("Software"),
        "datetime_original": _get("DateTimeOriginal"),
        "datetime_digitized": _get("DateTimeDigitized"),
        "datetime": _get("DateTime"),
        "artist": _get("Artist"),
        "copyright": _get("Copyright"),
        "orientation": _get("Orientation"),
        "lens_model": _get("LensModel"),
        "serial_number": _get("BodySerialNumber", "SerialNumber"),
        "gps": gps_summary if gps_summary else None,
    }
    # Remove None entries for compactness
    summary = {k: v for k, v in summary.items() if v is not None}

    # Size-limit tags
    tags_limited = _truncate_mapping(tags, max_items=64)

    return {
        "present": True,
        "summary": summary,
        "tags": tags_limited,
        "tag_count": int(len(tags)),
    }


def _extract_jpeg_xmp(data: bytes) -> List[str]:
    """
    Extract XMP packets from JPEG APP1 segments (best-effort).
    Returns a list of decoded XML strings.
    """
    if len(data) < 4 or not data.startswith(b"\xFF\xD8"):
        return []
    out: List[str] = []
    i = 2
    header = b"http://ns.adobe.com/xap/1.0/\x00"
    # Iterate markers until SOS (0xFFDA)
    while i + 4 <= len(data):
        # Find 0xFF marker prefix (skip fill bytes)
        if data[i] != 0xFF:
            i += 1
            continue
        while i < len(data) and data[i] == 0xFF:
            i += 1
        if i >= len(data):
            break
        marker = data[i]
        i += 1

        # Standalone markers
        if marker in (0xD9, 0xDA):  # EOI or SOS
            break
        if i + 2 > len(data):
            break
        seg_len = int.from_bytes(data[i : i + 2], "big", signed=False)
        i += 2
        if seg_len < 2:
            break
        seg_data_len = seg_len - 2
        if i + seg_data_len > len(data):
            break
        seg = data[i : i + seg_data_len]
        i += seg_data_len

        if marker == 0xE1 and seg.startswith(header):
            xml_bytes = seg[len(header) :]
            try:
                xml = xml_bytes.decode("utf-8", errors="replace")
            except Exception:
                xml = str(xml_bytes[:2048])
            out.append(xml)
    return out


def _extract_xmp_snippet(data: bytes) -> Optional[str]:
    """
    Heuristic XMP extraction: look for <x:xmpmeta ... </x:xmpmeta>.
    Returns a snippet (may be truncated).
    """
    try:
        # Search in a UTF-8 lossy decode for robustness
        text = data.decode("utf-8", errors="ignore")
        start = text.find("<x:xmpmeta")
        if start == -1:
            start = text.find("<xmpmeta")
        if start == -1:
            return None
        end = text.find("</x:xmpmeta>", start)
        if end != -1:
            end = end + len("</x:xmpmeta>")
            return text[start:end]
        # Fallback: take a window
        return text[start : start + 8192]
    except Exception:
        return None


def _extract_icc(img: Image.Image) -> Dict[str, Any]:
    icc_bytes = None
    try:
        icc_bytes = img.info.get("icc_profile")
    except Exception:
        icc_bytes = None

    if not icc_bytes:
        return {"present": False}

    desc: Optional[str] = None
    try:
        from PIL import ImageCms  # type: ignore

        profile = ImageCms.ImageCmsProfile(BytesIO(icc_bytes))
        # Try a couple of helpers; availability varies by Pillow build.
        try:
            desc = ImageCms.getProfileDescription(profile)  # type: ignore[attr-defined]
        except Exception:
            desc = None
        if not desc:
            try:
                desc = ImageCms.getProfileName(profile)  # type: ignore[attr-defined]
            except Exception:
                desc = None
    except Exception:
        desc = None

    return {
        "present": True,
        "byte_length": int(len(icc_bytes)),
        "description": _truncate_text(desc, 256),
    }


def _find_c2patool_executable() -> Optional[str]:
    # Prefer PATH
    exe = shutil.which("c2patool")
    if exe:
        return exe
    exe = shutil.which("c2patool.exe")
    if exe:
        return exe
    # Best-effort: repo root contains c2patool.exe
    try:
        workspace_root = Path(__file__).resolve().parent.parent.parent.parent
        candidate = workspace_root / "c2patool.exe"
        if candidate.exists():
            return str(candidate)
        candidate = workspace_root / "c2patool"
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass
    return None


def _summarize_c2pa_report(report: dict) -> Dict[str, Any]:
    """
    Summarize a c2patool ManifestStoreReport-like JSON object into stable, small fields.
    """
    active = report.get("active_manifest")
    manifests = report.get("manifests") or {}
    manifest_count = len(manifests) if isinstance(manifests, dict) else 0
    claim_generators: List[str] = []
    try:
        if isinstance(manifests, dict):
            for _, m in manifests.items():
                if isinstance(m, dict):
                    cg = m.get("claim_generator")
                    if isinstance(cg, str) and cg:
                        claim_generators.append(cg)
    except Exception:
        pass
    # Unique, preserve order
    seen = set()
    claim_generators_unique: List[str] = []
    for cg in claim_generators:
        if cg in seen:
            continue
        seen.add(cg)
        claim_generators_unique.append(cg)

    # Validation info can be large; keep small counts only
    validation_status = report.get("validation_status")
    validation_results = report.get("validation_results")
    status_count = len(validation_status) if isinstance(validation_status, list) else 0
    success_count = 0
    info_count = 0
    failure_count = 0
    if isinstance(validation_results, dict):
        for key, tgt in (
            ("success", "success_count"),
            ("informational", "info_count"),
            ("failure", "failure_count"),
        ):
            val = validation_results.get(key)
            if isinstance(val, list):
                if tgt == "success_count":
                    success_count = len(val)
                elif tgt == "info_count":
                    info_count = len(val)
                else:
                    failure_count = len(val)

    summary: Dict[str, Any] = {
        "active_manifest": active if isinstance(active, str) else None,
        "manifest_count": int(manifest_count),
        "claim_generators": claim_generators_unique[:8],
        "validation_status_count": int(status_count),
        "validation_results_counts": {
            "success": int(success_count),
            "informational": int(info_count),
            "failure": int(failure_count),
        },
    }
    summary = {k: v for k, v in summary.items() if v is not None}
    return summary


def _run_c2patool(exe_path: str, image_path: str, timeout_seconds: float) -> Dict[str, Any]:
    """
    Run c2patool on an asset and parse summary JSON from stdout.
    """
    cmd = [exe_path, image_path]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "checked_with": "c2patool",
            "present": False,
            "error": f"c2patool timed out after {timeout_seconds}s",
        }
    except Exception as e:
        return {
            "checked_with": "c2patool",
            "present": False,
            "error": str(e),
        }

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    # Common "no manifest" signals
    no_manifest_markers = (
        "No claim found",
        "No manifests",
        "No C2PA Manifests",
    )
    if any(m in stderr for m in no_manifest_markers) or any(m in stdout for m in no_manifest_markers):
        return {
            "checked_with": "c2patool",
            "present": False,
            "note": "c2patool reported no C2PA claim/manifests",
        }

    report: Optional[dict] = None
    if _looks_like_json(stdout):
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                report = parsed
        except Exception:
            report = None
    if report is None:
        report = _extract_first_json_object(stdout)

    if report is None:
        # Could be a non-JSON human-readable output; capture small snippet for debugging.
        return {
            "checked_with": "c2patool",
            "present": False,
            "error": "c2patool output was not valid JSON",
            "stdout_snippet": _truncate_text(stdout, 512),
            "stderr_snippet": _truncate_text(stderr, 512),
            "exit_code": int(proc.returncode),
        }

    manifests = report.get("manifests")
    present = bool(manifests) if isinstance(manifests, dict) else False
    if report.get("active_manifest"):
        present = True

    return {
        "checked_with": "c2patool",
        "present": bool(present),
        "summary": _summarize_c2pa_report(report),
        "exit_code": int(proc.returncode),
    }


@dataclass(frozen=True)
class _IncludeFlags:
    exif: bool = True
    xmp: bool = True
    icc: bool = True
    c2pa: bool = True


@dataclass(frozen=True)
class _Options:
    # Max bytes to scan for XMP/C2PA heuristics
    max_scan_bytes: int = 16 * 1024 * 1024
    # Timeout for c2patool invocation
    c2pa_timeout_seconds: float = 8.0


def _parse_request(input_value: Union[str, dict]) -> Tuple[str, _IncludeFlags, _Options]:
    defaults_inc = _IncludeFlags()
    defaults_opt = _Options()

    if isinstance(input_value, dict):
        data = input_value
    else:
        s = str(input_value or "").strip()
        try:
            parsed = json.loads(s)
            data = parsed if isinstance(parsed, dict) else None
        except Exception:
            data = None
        if not data:
            return s, defaults_inc, defaults_opt

    path = str(data.get("path") or data.get("image_path") or "").strip()
    if not path:
        # Fall back to raw input string if it wasn't dict-shaped correctly
        path = str(input_value or "").strip()

    include = data.get("include") if isinstance(data.get("include"), dict) else {}
    inc = _IncludeFlags(
        exif=bool(include.get("exif", defaults_inc.exif)),
        xmp=bool(include.get("xmp", defaults_inc.xmp)),
        icc=bool(include.get("icc", defaults_inc.icc)),
        c2pa=bool(include.get("c2pa", defaults_inc.c2pa)),
    )

    c2pa_opts = data.get("c2pa") if isinstance(data.get("c2pa"), dict) else {}
    try:
        max_scan_bytes = int(data.get("max_scan_bytes", defaults_opt.max_scan_bytes))
    except Exception:
        max_scan_bytes = defaults_opt.max_scan_bytes
    try:
        c2pa_timeout = float(c2pa_opts.get("timeout_seconds", defaults_opt.c2pa_timeout_seconds))
    except Exception:
        c2pa_timeout = defaults_opt.c2pa_timeout_seconds

    opt = _Options(
        max_scan_bytes=max(1024 * 1024, min(max_scan_bytes, 128 * 1024 * 1024)),
        c2pa_timeout_seconds=max(1.0, min(c2pa_timeout, 60.0)),
    )
    return path, inc, opt


def metadata(input_str: str) -> str:
    """
    Extract EXIF/XMP/ICC and C2PA/Content Credentials signals from an image file.

    Input format:
    - Plain string path: "path/to/image.jpg"
    - Or JSON: {"path":"...","include":{...},"c2pa":{"timeout_seconds":8}}
    """
    image_path, include, options = _parse_request(input_str)
    try:
        if not image_path:
            raise ValueError("No image path provided.")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        p = Path(image_path)
        stat = p.stat()
        file_info = {
            "name": p.name,
            "suffix": p.suffix,
            "size_bytes": int(stat.st_size),
            "mtime_epoch": float(stat.st_mtime),
            "mtime_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        }

        # Byte prefix for XMP/C2PA heuristics (also helps if PIL can't open)
        scan_bytes = _read_bytes_prefix(image_path, options.max_scan_bytes)

        img: Optional[Image.Image] = None
        image_info: Dict[str, Any] = {}
        exif_info: Dict[str, Any] = {"present": False}
        icc_info: Dict[str, Any] = {"present": False}
        try:
            img = Image.open(image_path)
            image_info = {
                "format": img.format,
                "mode": img.mode,
                "size": list(img.size) if hasattr(img, "size") else None,
            }
            if include.exif:
                exif_info = _extract_exif(img)
            if include.icc:
                icc_info = _extract_icc(img)
        except Exception as e:
            image_info = {"error": f"PIL failed to open image: {e}"}

        # XMP
        xmp_packets: List[str] = []
        xmp_snippet: Optional[str] = None
        if include.xmp:
            xmp_packets = _extract_jpeg_xmp(scan_bytes)
            if xmp_packets:
                xmp_snippet = xmp_packets[0]
            else:
                xmp_snippet = _extract_xmp_snippet(scan_bytes)
        xmp_present = bool(xmp_packets or xmp_snippet)
        xmp_summary: Dict[str, Any] = {
            "present": bool(xmp_present),
            "packet_count": int(len(xmp_packets)),
            "snippet": _truncate_text(xmp_snippet, 4096),
        }
        xmp_summary = {k: v for k, v in xmp_summary.items() if v is not None}

        # C2PA / Content Credentials
        c2pa_info: Dict[str, Any] = {"present": False}
        if include.c2pa:
            exe = _find_c2patool_executable()
            if exe:
                c2pa_info = _run_c2patool(exe, image_path, options.c2pa_timeout_seconds)
                c2pa_info["c2patool_available"] = True
                c2pa_info["c2patool_path"] = exe
            else:
                # Heuristic fallback: search for telltale strings in prefix
                hay = scan_bytes.lower()
                hits: List[str] = []
                for term in (
                    b"c2pa",
                    b"urn:c2pa",
                    b"jumbf",
                    b"contentcredentials",
                    b"content credentials",
                ):
                    if term in hay:
                        hits.append(term.decode("utf-8", errors="ignore"))
                # XMP can contain the strongest hints
                if xmp_present and xmp_snippet and re.search(r"(?i)\bc2pa\b", xmp_snippet):
                    hits.append("xmp:c2pa")
                hits = list(dict.fromkeys(hits))  # unique
                c2pa_info = {
                    "checked_with": "heuristic",
                    "present": bool(hits),
                    "hits": hits[:16],
                    "note": "Heuristic scan only (c2patool not available).",
                }

        result: Dict[str, Any] = {
            "tool": "metadata",
            "status": "completed",
            "image_path": image_path,
            "generated_at": _now_iso(),
            "file": file_info,
            "image": image_info,
        }
        if include.exif:
            result["exif"] = exif_info
        if include.xmp:
            result["xmp"] = xmp_summary
        if include.icc:
            result["icc"] = icc_info
        if include.c2pa:
            result["c2pa"] = c2pa_info

        # Keep overall payload bounded in size
        return _safe_json_dumps(result)
    except Exception as e:  # pragma: no cover - defensive
        return _safe_json_dumps({"tool": "metadata", "status": "error", "error": str(e)})


__all__ = ["metadata"]


