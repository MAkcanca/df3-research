"""
Error Level Analysis (ELA) helper.

ELA recompresses an image at a controlled JPEG quality level and measures
pixel-wise error between the original and the recompressed copy. Regions that
were edited or generated separately often compress differently, producing
higher error levels.

Algorithm based on the Sherloq ELA implementation.

References:
- Nature Scientific Reports 2023: ELA + CNN achieves ~89.5% deepfake accuracy
- PMC 2024: ELA integration improves detection precision for AI imagery
"""

import base64
import json
from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image


def _parse_request(input_str: str) -> Tuple[str, int, int, bool, bool, int, int, bool]:
    """
    Accept plain path or JSON payload:
    {
        "path": "/path/to/image.jpg",
        "quality": 75,            # JPEG recompression quality (1-100), default 75 to match Sherloq
        "max_size": 0,            # Max side length for returned map (px), 0 = no resize
        "return_map": false,       # Include base64 PNG ELA map
        "linear": false,          # Use linear mode (default: false = non-linear with sqrt)
        "scale": 50,              # Scale factor (default: 50, applied as scale/20 in non-linear)
        "contrast": 20,           # Contrast adjustment percentage (default: 20)
        "grayscale": false        # Output grayscale (default: false = color RGB)
    }
    """
    default_quality = 75  # Match Sherloq default
    default_max_size = 0  # No resize by default (match Sherloq)
    default_return_map = False
    default_linear = False
    default_scale = 50
    default_contrast = 20
    default_grayscale = False  # Color output by default (match Sherloq)
    try:
        data = json.loads(input_str)
        if isinstance(data, dict):
            path = data.get("path", input_str).strip()
            quality = int(data.get("quality", default_quality))
            max_size = int(data.get("max_size", default_max_size))
            return_map = bool(data.get("return_map", default_return_map))
            linear = bool(data.get("linear", default_linear))
            scale = int(data.get("scale", default_scale))
            contrast = int(data.get("contrast", default_contrast))
            grayscale = bool(data.get("grayscale", default_grayscale))
            return path, quality, max_size, return_map, linear, scale, contrast, grayscale
    except Exception:
        pass
    return (
        input_str.strip(),
        default_quality,
        default_max_size,
        default_return_map,
        default_linear,
        default_scale,
        default_contrast,
        default_grayscale,
    )


def _clamp_quality(quality: int) -> int:
    return max(1, min(100, int(quality)))


def _resize_max(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    w, h = img.size
    scale = max(w, h)
    if scale <= max_side:
        return img
    ratio = max_side / float(scale)
    new_size = (max(1, int(round(w * ratio))), max(1, int(round(h * ratio))))
    return img.resize(new_size, Image.LANCZOS)


def _recompress_jpeg(img: Image.Image, quality: int) -> Image.Image:
    """
    Recompress image as JPEG with specified quality.
    
    Uses default chroma subsampling (4:2:0) to match OpenCV/Sherloq behavior.
    This produces more color compression artifacts, resulting in brighter
    color differences in the ELA output.
    """
    buf = BytesIO()
    # Don't override subsampling - use default (4:2:0) to match OpenCV behavior
    # OpenCV's imencode uses 4:2:0 chroma subsampling by default
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _encode_png(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _create_lut(low: int, high: int) -> np.ndarray:
    """
    Create a lookup table for contrast adjustment, matching Sherloq's create_lut.
    
    Args:
        low: Lower bound adjustment (typically contrast value)
        high: Upper bound adjustment (typically contrast value)
    
    Returns:
        LUT array of shape (256,) with uint8 values
    """
    if low >= 0:
        p1 = (+low, 0)
    else:
        p1 = (0, -low)
    if high >= 0:
        p2 = (255 - high, 255)
    else:
        p2 = (255, 255 + high)
    if p1[0] == p2[0]:
        return np.full(256, 255, dtype=np.uint8)
    # Vectorized LUT creation (more efficient than list comprehension)
    x = np.arange(256, dtype=np.float64)
    lut = (x * (p1[1] - p2[1]) + p1[0] * p2[1] - p1[1] * p2[0]) / (p1[0] - p2[0])
    return np.clip(lut, 0, 255).astype(np.uint8)


def _convert_scale_abs(arr: np.ndarray, scale: float) -> np.ndarray:
    """
    Equivalent to cv.convertScaleAbs: scale array and convert to uint8.
    
    Args:
        arr: Input array (float)
        scale: Scale factor
    
    Returns:
        Scaled and clipped uint8 array
    """
    scaled = arr * scale
    return np.clip(scaled, 0, 255).astype(np.uint8)


def perform_ela(input_str: str) -> str:
    """
    Run Error Level Analysis on an image path.
    
    Matches Sherloq's ELA implementation behavior:
    - Non-linear mode (default): sqrt of normalized difference with scale factor
    - Linear mode: direct subtraction with scale factor
    - Applies contrast adjustment via LUT
    - Outputs color (3-channel RGB) by default, grayscale optional

    Returns JSON with:
    - ela_map: base64 PNG (color or grayscale, optional)
    - ela_mean: mean absolute error (0-255)
    - ela_std: stddev of absolute error
    - ela_anomaly_score: z-score of 95th percentile vs mean (higher = more localized anomalies)
    """
    (
        image_path,
        quality,
        max_side,
        want_map,
        linear,
        scale,
        contrast,
        grayscale,
    ) = _parse_request(input_str)
    quality = _clamp_quality(quality)
    try:
        img = Image.open(image_path).convert("RGB")
        recompressed = _recompress_jpeg(img, quality)

        original_arr = np.asarray(img).astype(np.float32)
        recompressed_arr = np.asarray(recompressed).astype(np.float32)

        # Align shapes in case of encoder padding differences
        h = min(original_arr.shape[0], recompressed_arr.shape[0])
        w = min(original_arr.shape[1], recompressed_arr.shape[1])
        original_arr = original_arr[:h, :w]
        recompressed_arr = recompressed_arr[:h, :w]

        # Match Sherloq's ELA computation (keep 3-channel color)
        if not linear:
            # Non-linear mode (default): normalize to 0-1, compute absdiff, sqrt, scale
            original_norm = original_arr / 255.0
            recompressed_norm = recompressed_arr / 255.0
            difference = np.abs(original_norm - recompressed_norm)
            # Keep per-channel (3-channel color output like Sherloq)
            ela_sqrt = np.sqrt(difference) * 255.0
            # Apply scale factor (scale/20 for non-linear mode)
            ela_rgb = _convert_scale_abs(ela_sqrt, scale / 20.0)
        else:
            # Linear mode: direct subtraction
            ela = np.subtract(recompressed_arr, original_arr)
            # Apply scale factor (scale for linear mode)
            ela_rgb = _convert_scale_abs(ela, scale)

        # Apply contrast adjustment via LUT to each channel (matching Sherloq)
        contrast_val = int(contrast / 100 * 128)  # Convert percentage to 0-128 range
        lut = _create_lut(contrast_val, contrast_val)
        ela_rgb = lut[ela_rgb]  # Apply LUT to all channels

        # Optionally convert to grayscale (Sherloq default is color)
        if grayscale:
            # Desaturate: convert to grayscale, then back to 3-channel for consistency
            ela_gray = np.mean(ela_rgb, axis=2).astype(np.uint8)
            ela_output = ela_gray
            output_mode = "L"
        else:
            ela_output = ela_rgb
            output_mode = "RGB"

        # Compute statistics on the grayscale version for consistency
        ela_gray_stats = np.mean(ela_rgb.astype(np.float32), axis=2)
        ela_mean = float(np.mean(ela_gray_stats))
        ela_std = float(np.std(ela_gray_stats))
        p95 = float(np.percentile(ela_gray_stats, 95)) if ela_gray_stats.size else 0.0
        ela_anomaly_score = float(max(0.0, (p95 - ela_mean) / (ela_std + 1e-6)))

        ela_map_encoded = None
        map_size = None
        if want_map:
            ela_img = Image.fromarray(ela_output, mode=output_mode)
            if max_side > 0:
                ela_img = _resize_max(ela_img, max_side)
            ela_map_encoded = _encode_png(ela_img)
            map_size = ela_img.size

        result = {
            "tool": "perform_ela",
            "status": "completed",
            "image_path": image_path,
            "quality": quality,
            "ela_mean": ela_mean,
            "ela_std": ela_std,
            "ela_anomaly_score": ela_anomaly_score,
            "ela_map": ela_map_encoded,
            "ela_map_size": map_size,
            "note": (
                "ELA recompresses at fixed JPEG quality. "
                "Anomaly score is the z-score of the 95th percentile vs mean; "
                "values >2 suggest localized high error regions."
            ),
        }
        return json.dumps(result)
    except Exception as e:  # pragma: no cover - defensive
        return json.dumps(
            {
                "tool": "perform_ela",
                "status": "error",
                "error": str(e),
            }
        )


__all__ = ["perform_ela"]
