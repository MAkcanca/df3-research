"""
JPEG-focused forensic helpers.
Implements quantization-step estimation and double-compression cues aligned
with the MATLAB reference toolbox (factor histograms, Sac score, block maps).

Algorithms based on:
- Factor histogram quantization step estimation (fh_jpgstep.m from MATLAB forensics toolbox)
- Sac/JPEGness score (fh_jpgdetect.m) - measures JPEG compression artifacts.
  Higher values indicate stronger JPEG compression. Used to detect if an image was
  previously JPEG-compressed, NOT directly for double-compression detection.
- Block-level tamper probability map (Extract_Features_JPEG.m from DCT coefficient analysis)
- JPEG zig-zag ordering (Sherloq jpeg.py)

Dependencies:
- jpeglib (optional): For raw JPEG DCT coefficient access, matching MATLAB's jpeg_read().
  Install with: pip install jpeglib
  Supports libjpeg 6b-9e, libjpeg-turbo, and mozjpeg.
  When jpeglib is not available, falls back to pixel-domain DCT (less accurate for
  factor histogram estimation, especially DC coefficient).
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.fftpack import dct

# Try to import jpeglib for raw coefficient access (like MATLAB's jpeg_read)
# https://github.com/martinbenes1996/jpeglib
try:
    import jpeglib
    _HAS_JPEGLIB = True
except ImportError:
    _HAS_JPEGLIB = False

# Zig-zag order from sherloq/gui/jpeg.py (ZIG_ZAG constant, line 6-71)
_ZIG_ZAG: List[Tuple[int, int]] = [
    (0, 0),
    (0, 1),
    (1, 0),
    (2, 0),
    (1, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (2, 1),
    (3, 0),
    (4, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 4),
    (0, 5),
    (1, 4),
    (2, 3),
    (3, 2),
    (4, 1),
    (5, 0),
    (6, 0),
    (5, 1),
    (4, 2),
    (3, 3),
    (2, 4),
    (1, 5),
    (0, 6),
    (0, 7),
    (1, 6),
    (2, 5),
    (3, 4),
    (4, 3),
    (5, 2),
    (6, 1),
    (7, 0),
    (7, 1),
    (6, 2),
    (5, 3),
    (4, 4),
    (3, 5),
    (2, 6),
    (1, 7),
    (2, 7),
    (3, 6),
    (4, 5),
    (5, 4),
    (6, 3),
    (7, 2),
    (7, 3),
    (6, 4),
    (5, 5),
    (4, 6),
    (3, 7),
    (4, 7),
    (5, 6),
    (6, 5),
    (7, 4),
    (7, 5),
    (6, 6),
    (5, 7),
    (6, 7),
    (7, 6),
    (7, 7),
]


def _std_luma_table() -> Tuple[int, ...]:
    """ITU-T81 standard luminance quantization table."""
    return (
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    )


def _read_jpeg_raw(image_path: str) -> Optional[Dict]:
    """
    Read raw JPEG DCT coefficients using jpeglib.
    This is the Python equivalent of MATLAB's jpeg_read() from the JPEG Toolbox.

    Reference: https://github.com/martinbenes1996/jpeglib
    Uses libjpeg internally, supporting versions 6b-9e, libjpeg-turbo, mozjpeg.

    Returns dict with:
    - Y: Luminance DCT coefficients - already quantized integers
      jpeglib returns Y as 4D array (h_blocks, w_blocks, 8, 8)
    - Cb, Cr: Chrominance DCT coefficients (may be None for grayscale)
    - qt: Quantization tables list
    - height, width: Image dimensions in pixels

    Returns None if jpeglib is not available or read fails.
    """
    if not _HAS_JPEGLIB:
        return None
    try:
        im = jpeglib.read_dct(image_path)
        # Verify we have the expected attributes
        if not hasattr(im, 'Y') or im.Y is None:
            return None
        
        # jpeglib returns Y as 4D array: (h_blocks, w_blocks, 8, 8)
        # Convert to our expected format: (n_blocks, 8, 8)
        y_shape = im.Y.shape
        if len(y_shape) == 4:
            # Shape is (h_blocks, w_blocks, 8, 8)
            h_blocks, w_blocks = y_shape[0], y_shape[1]
            # Reshape to (n_blocks, 8, 8)
            y_blocks = im.Y.reshape(-1, 8, 8)
            # Calculate actual image dimensions
            height = h_blocks * 8
            width = w_blocks * 8
        elif len(y_shape) == 2:
            # Fallback: if it's 2D, treat as (h, w) and we'll blockify later
            height, width = y_shape
            y_blocks = None  # Will need to blockify
        else:
            return None
        # Verify quantization tables exist
        if not hasattr(im, 'qt') or im.qt is None:
            return None
        
        # Convert qt to list format for consistency with our API
        # jpeglib returns qt as numpy array with shape (n_tables, 8, 8)
        qt_list = []
        try:
            qt_array = np.asarray(im.qt, dtype=np.int32)
            if qt_array.size == 0:
                return None
            
            # Handle qt array - jpeglib returns (n_tables, 8, 8)
            if qt_array.ndim == 3 and qt_array.shape[1] == 8 and qt_array.shape[2] == 8:
                # Shape (n_tables, 8, 8) - multiple tables
                qt_list = [qt_array[i] for i in range(qt_array.shape[0])]
            elif qt_array.ndim == 2 and qt_array.shape == (8, 8):
                # Single 8x8 table
                qt_list = [qt_array]
            else:
                # Try to extract 8x8 blocks
                flat = qt_array.flatten()
                n_tables = flat.size // 64
                if n_tables > 0:
                    qt_list = [flat[i*64:(i+1)*64].reshape(8, 8) for i in range(n_tables)]
            
            if not qt_list:
                return None
        except Exception:
            # If qt conversion fails, return None (fall back to pixel domain)
            return None
        
        # Return data with Y as blocks (n_blocks, 8, 8)
        return {
            "Y": y_blocks,  # Luminance DCT blocks (n_blocks, 8, 8) - quantized coefficients
            "Cb": im.Cb,  # Chrominance Cb (may be None or 4D like Y)
            "Cr": im.Cr,  # Chrominance Cr (may be None or 4D like Y)
            "qt": qt_list,  # List of quantization tables (8x8 arrays)
            "height": height,
            "width": width,
            "grid_shape": (h_blocks, w_blocks) if len(y_shape) == 4 else None,  # Store grid shape for convenience
        }
    except Exception:
        # Return None on any error (file not found, invalid JPEG, etc.)
        # In production, you might want to log this: import logging; logging.debug(f"jpeglib read failed: {e}")
        return None


def _coef_array_to_blocks(coef_array: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Convert jpeglib coefficient array format to per-block format.

    jpeglib stores DCT coefficients in a 2D array where blocks are arranged
    spatially (each 8x8 region is one DCT block). This matches MATLAB's
    jpeg_read().coef_arrays{1} format.

    Returns:
    - blocks: (n_blocks, 8, 8) array of DCT coefficients
    - grid_shape: (h_blocks, w_blocks) tuple
    """
    h, w = coef_array.shape
    h_blocks = h // 8
    w_blocks = w // 8
    if h_blocks == 0 or w_blocks == 0:
        return np.empty((0, 8, 8), dtype=coef_array.dtype), (0, 0)
    # Crop to exact multiple of 8
    cropped = coef_array[:h_blocks * 8, :w_blocks * 8]
    # Reshape to blocks: (h_blocks, 8, w_blocks, 8) -> (h_blocks, w_blocks, 8, 8) -> (n_blocks, 8, 8)
    blocks = cropped.reshape(h_blocks, 8, w_blocks, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    return blocks, (h_blocks, w_blocks)


def _blockify(gray: np.ndarray):
    """Crop to 8x8 grid and return blocks, grid shape, and cropped shape."""
    blk = 8
    h, w = gray.shape
    h_crop = h - (h % blk)
    w_crop = w - (w % blk)
    if h_crop < blk or w_crop < blk:
        return None, (0, 0), (0, 0)
    cropped = gray[:h_crop, :w_crop]
    blocks = cropped.reshape(h_crop // blk, blk, w_crop // blk, blk).transpose(0, 2, 1, 3).reshape(-1, blk, blk)
    return blocks, (h_crop // blk, w_crop // blk), (h_crop, w_crop)


def _parse_request(input_str: str):
    """
    Allow either plain path or JSON string:
    {
        "path": "/path/to/image.jpg",
        "include": {
            "primary_table": true,
            "truncation_mask": true,
            "block_map": true,
            "per_frequency": false
        }
    }
    """
    try:
        data = json.loads(input_str)
        path = data.get("path", input_str.strip())
        include = data.get("include", {}) if isinstance(data, dict) else {}
        include = include if isinstance(include, dict) else {}
        return path, include
    except Exception:
        return input_str.strip(), {}


def _truncation_mask_from_pixels(image: np.ndarray, grid_shape: Tuple[int, int]) -> np.ndarray:
    """
    True for blocks that are NOT truncated (no 0/255 clipping).
    Reference: fh_jpgstep.m lines 17-29, fh_jpgdetect.m lines 10-23
    """
    blk = 8
    h_blocks, w_blocks = grid_shape
    if h_blocks == 0 or w_blocks == 0:
        return np.zeros((h_blocks, w_blocks), dtype=bool)
    cropped = image[: h_blocks * blk, : w_blocks * blk]
    blocks = cropped.reshape(h_blocks, blk, w_blocks, blk).transpose(0, 2, 1, 3)
    block_max = blocks.max(axis=(2, 3))
    block_min = blocks.min(axis=(2, 3))
    # MATLAB: if pmax == 255 || pmin == 0 then exclude
    return np.logical_and(block_max < 255, block_min > 0)


def _block_dcts_from_pixels(gray: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Compute block DCTs from pixel values.
    Reference: bdct() in MATLAB.
    Note: This is a fallback when jpegio is not available.
    """
    blocks, grid_shape, _ = _blockify(gray)
    if blocks is None:
        return np.empty((0, 8, 8), dtype=np.float32), grid_shape
    # MATLAB: bdct(pmtx - 128)
    dcts = dct(dct(blocks.astype(np.float32) - 128.0, axis=1, norm="ortho"), axis=2, norm="ortho")
    return dcts, grid_shape


def _estimate_quality(qtable) -> Dict[str, object]:
    """
    Estimate JPEG quality from quantization table by matching against
    standard tables scaled per IJG formula.
    Reference: sherloq/gui/jpeg.py get_tables() and quality.py
    """
    base = np.array(_std_luma_table(), dtype=np.int32).reshape(8, 8)
    best_q = None
    best_err = float("inf")
    best_table = None
    for quality in range(1, 101):
        # IJG quality scaling formula
        scale = 5000 / quality if quality < 50 else 200 - quality * 2
        cand = np.floor((base * scale + 50) / 100).astype(np.int32)
        cand = np.clip(cand, 1, 255)
        err = float(np.mean(np.abs(cand - qtable)))
        if err < best_err:
            best_err = err
            best_q = quality
            best_table = cand
    return {
        "estimated_quality": int(best_q) if best_q is not None else None,
        "mean_abs_error": best_err if best_q is not None else None,
        "exact_match": bool(best_err == 0.0) if best_q is not None else False,
        "best_fit_table": best_table.tolist() if best_table is not None else None,
    }


def _factor_histogram_step(coeffs: np.ndarray, threshold: float = 0.7) -> int:
    """
    Estimate quantization step via factor histogram.
    Reference: fh_jpgstep.m lines 32-52

    MATLAB code:
        fhcell = coefhist(dctmtx, mask, 'factor_histogram');
        cfh = cfh / cfh(1);
        step(invpos(i)) = find(cfh>=t, 1, 'last');
    """
    samples = np.abs(np.round(coeffs)).astype(np.int64)
    # MATLAB coefhist.m line 31: samples = samples(samples>1)
    samples = samples[samples > 1]
    if samples.size == 0:
        return 0
    maxel = int(samples.max())
    if maxel <= 1:
        return 1
    qsmax = min(100, maxel)
    # MATLAB: mode_hist = hist(samples(:), 1:maxel)
    # Creates array where mode_hist(1) = count of value 1, mode_hist(2) = count of value 2, etc.
    # Since samples only contains values >= 2, mode_hist(1) = 0.
    # Python: hist_full = np.bincount(samples) creates hist_full[0] = count(0), hist_full[1] = count(1), etc.
    # So mode_hist(q) corresponds to hist_full[q] for q >= 1.
    hist_full = np.bincount(samples, minlength=maxel + 1)
    # Check if we have any samples (values >= 2)
    if np.sum(hist_full[2:]) == 0:
        return 1
    # MATLAB: fh(q) = sum(mode_hist(q:q:end)) for q = 1:fhlen
    # mode_hist(q:q:end) accesses indices q, 2q, 3q, ... up to maxel (MATLAB 1-based)
    # In Python, hist_full[q::q] accesses indices q, 2q, 3q, ... which correspond to the same coefficient values
    fhlen = min(qsmax, maxel)
    fh = np.array([hist_full[q::q].sum() for q in range(1, fhlen + 1)], dtype=np.float64)
    if fh[0] == 0:
        return 1
    fh /= fh[0]
    # MATLAB: find(cfh>=t, 1, 'last')
    above = np.where(fh >= threshold)[0]
    if above.size == 0:
        return 1
    return int(above[-1] + 1)


def _estimate_primary_qtable_raw(coef_blocks: np.ndarray, grid_shape: Tuple[int, int],
                                  gray: Optional[np.ndarray] = None,
                                  threshold: float = 0.7) -> Dict[str, object]:
    """
    Per-frequency quantization step estimation using factor histograms.
    Reference: fh_jpgstep.m

    Uses raw DCT coefficients from jpegio for accurate estimation.
    """
    if coef_blocks.size == 0:
        return {"table": None, "mask": None}
    h_blocks, w_blocks = grid_shape
    # Truncation mask from pixels if available
    if gray is not None:
        mask_valid = _truncation_mask_from_pixels(gray, grid_shape).reshape(-1)
    else:
        mask_valid = np.ones(coef_blocks.shape[0], dtype=bool)
    steps = np.zeros((8, 8), dtype=np.int32)
    for idx, (u, v) in enumerate(_ZIG_ZAG):
        coeffs = coef_blocks[:, u, v].astype(np.float64)
        if mask_valid.size == coeffs.size:
            coeffs = coeffs[mask_valid]
        step = _factor_histogram_step(coeffs, threshold=threshold)
        steps[u, v] = step
    return {"table": steps.tolist(), "mask": mask_valid.reshape(h_blocks, w_blocks).tolist()}


def _sac_score_raw(coef_blocks: np.ndarray) -> Dict[str, object]:
    """
    Sac/JPEGness score using raw DCT coefficients.
    Reference: fh_jpgdetect.m

    Higher score = stronger JPEG artifacts = more likely to be JPEG-compressed.

    MATLAB code (fh_jpgdetect.m):
        dctmtx(1:8:end,1:8:end) = 0;           % only AC coefficients
        samples = abs(round(dctmtx(:)));
        samples = samples(samples>1);          % exclude 0, -1, and 1
        coef_histo = hist(samples, 1:maxel);
        fh(q) = sum(coef_histo(q:q:end));      % factor histogram
        fh = fh / fh(1);                       % normalize
        deriv1 = fh(2:end) - fh(1:end-1);      % first derivative
        S = max(deriv1);                       % Sac score
    """
    if coef_blocks.size == 0:
        return {"score": None, "note": "No DCT blocks available."}
    dcts = coef_blocks.copy().astype(np.float64)
    # MATLAB line 27: dctmtx(1:8:end,1:8:end) = 0 (ignore DC)
    dcts[:, 0, 0] = 0
    samples = np.abs(np.round(dcts.reshape(-1))).astype(np.int64)
    # MATLAB line 30-31: exclude 0, -1, and 1 -> samples = samples(samples>1)
    samples = samples[samples > 1]
    if samples.size == 0:
        return {"score": 0.0, "note": "Insufficient AC energy (no |coef| > 1)."}
    maxel = int(samples.max())
    if maxel <= 1:
        return {"score": 0.0, "note": "Max coefficient <= 1."}
    # MATLAB line 34: coef_histo = hist(samples, 1:maxel)
    coef_histo = np.bincount(samples, minlength=maxel + 1)[1:maxel + 1]  # bins 1 to maxel
    if coef_histo.size == 0:
        return {"score": 0.0, "note": "Empty histogram."}
    # MATLAB lines 42-46: factor histogram
    qsmax = 100
    fhlen = min(qsmax, maxel)
    fh = np.zeros(fhlen, dtype=np.float64)
    for q in range(1, fhlen + 1):
        # MATLAB: fh(q) = sum(coef_histo(q:q:end))
        fh[q - 1] = coef_histo[q - 1::q].sum()
    if fh[0] == 0:
        return {"score": 0.0, "note": "Factor histogram empty at q=1."}
    # MATLAB line 49: normalize
    fh /= fh[0]
    # MATLAB lines 51-52: S = max(deriv1)
    deriv1 = fh[1:] - fh[:-1]
    score = float(np.max(deriv1)) if deriv1.size > 0 else 0.0
    return {"score": score, "histogram_length": int(maxel), "source": "raw_coefficients"}


def _sac_score_pixels(gray: np.ndarray) -> Dict[str, object]:
    """
    Sac/JPEGness score from pixel-domain DCT (fallback when jpegio unavailable).
    Less accurate than raw coefficient version.
    """
    dcts, _ = _block_dcts_from_pixels(gray)
    if dcts.size == 0:
        return {"score": None, "note": "Image too small for DCT grid."}
    result = _sac_score_raw(dcts)
    result["source"] = "pixel_domain"
    return result


def _period_from_histogram(hist: np.ndarray) -> int:
    """
    Find dominant period via FFT peak.
    Reference: Extract_Features_JPEG.m lines 67-94

    MATLAB code:
        FFT=abs(fft(coeffHist));
        DC=FFT(1);
        FreqValley=1;
        while (FreqValley<length(FFT)-1) && (FFT(FreqValley)>= FFT(FreqValley+1))
            FreqValley=FreqValley+1;
        end
        FFT=FFT(FreqValley:floor(length(FFT)/2));
        [maxPeak,FFTPeak]=max(FFT);
        FFTPeak=FFTPeak+FreqValley-1-1;
        if length(FFTPeak)==0 | maxPeak<DC/5 | min(FFT)/maxPeak>0.9
            p_h_fft(coeffIndex)=1;
        else
            p_h_fft(coeffIndex)=round(length(coeffHist)/FFTPeak);
        end
    """
    if hist.size == 0 or hist.sum() == 0:
        return 1
    fft_vals = np.abs(np.fft.fft(hist))
    if fft_vals.size < 3:
        return 1
    dc = fft_vals[0]
    # Find first local minimum to remove DC peak
    freq_valley = 0
    while freq_valley < fft_vals.size - 1 and fft_vals[freq_valley] >= fft_vals[freq_valley + 1]:
        freq_valley += 1
    # MATLAB: FFT=FFT(FreqValley:floor(length(FFT)/2))
    fft_slice = fft_vals[freq_valley: max(freq_valley + 1, fft_vals.size // 2)]
    if fft_slice.size == 0:
        return 1
    max_peak = fft_slice.max()
    fft_peak_local = int(np.argmax(fft_slice))
    # MATLAB: FFTPeak=FFTPeak+FreqValley-1-1
    fft_peak = fft_peak_local + freq_valley
    # MATLAB thresholds: maxPeak<DC/5 | min(FFT)/maxPeak>0.9
    if max_peak < dc / 5:
        return 1
    if fft_slice.size > 0 and fft_slice.min() / max(max_peak, 1e-9) > 0.9:
        return 1
    if fft_peak == 0:
        return 1
    # MATLAB: round(length(coeffHist)/FFTPeak)
    period = int(round(hist.size / fft_peak))
    return max(period, 1)


def _block_level_map(
    coef_blocks: np.ndarray,
    grid_shape: Tuple[int, int],
    max_coeffs: int = 15,
    include_per_frequency: bool = True,
) -> Dict[str, object]:
    """
    Block-level tamper probability map.
    Reference: Extract_Features_JPEG.m lines 101-143

    For each DCT frequency, computes per-block probability of tampering based on
    how well the coefficient matches the global histogram periodicity.

    MATLAB code:
        P_u=num./denom;
        P_t=1./p_final(coeffIndex);
        P_tampered(:,:,coeffIndex)=P_t./(P_u+P_t);
        P_untampered(:,:,coeffIndex)=P_u./(P_u+P_t);
        ...
        P_tampered_Overall=prod(P_tampered,3)./(prod(P_tampered,3)+prod(P_untampered,3));
    """
    h_blocks, w_blocks = grid_shape
    if h_blocks == 0 or w_blocks == 0:
        return {"map": None, "per_frequency": []}
    # Use log-odds for numerical stability when combining across frequencies
    log_odds = np.zeros((h_blocks, w_blocks), dtype=np.float64)
    per_freq_meta = []
    for coeff_idx, (u, v) in enumerate(_ZIG_ZAG[:max_coeffs]):
        coeff_matrix = np.round(coef_blocks[:, u, v]).reshape(h_blocks, w_blocks).astype(np.int32)
        coeff_list = coeff_matrix.flatten()
        if coeff_list.size == 0:
            continue
        min_hist = int(coeff_list.min()) - 1
        max_hist = int(coeff_list.max()) + 1
        if max_hist <= min_hist:
            continue
        # MATLAB: coeffHist=hist(coeffList,minHistValue:maxHistValue)
        hist = np.bincount(coeff_list - min_hist, minlength=max_hist - min_hist + 1)
        if hist.sum() == 0:
            continue
        period = _period_from_histogram(hist)
        if period <= 1:
            # No periodicity detected - neutral probability
            tampered = np.full_like(coeff_matrix, 0.5, dtype=np.float64)
        else:
            # MATLAB lines 103-127: compute per-block probabilities
            s0 = int(np.argmax(hist))
            adjusted = coeff_matrix - min_hist
            period_start = adjusted - ((adjusted - s0) % period)
            # Gather histogram counts across one period for denominator
            idxs = (period_start[..., None] + np.arange(period)) % hist.size
            denom = hist[idxs].sum(axis=-1).astype(np.float64)
            num = hist[np.clip(adjusted, 0, hist.size - 1)].astype(np.float64)
            # MATLAB: P_u = num./denom; P_t = 1./period
            pu = num / np.maximum(denom, 1e-9)
            pt = 1.0 / period
            # MATLAB: P_tampered = P_t./(P_u+P_t)
            tampered = pt / (pu + pt)
            tampered = np.clip(tampered, 1e-9, 1 - 1e-9)
        # Accumulate log-odds for final sigmoid combination
        log_odds += np.log(tampered) - np.log(1.0 - tampered)
        if include_per_frequency:
            per_freq_meta.append({"idx": coeff_idx + 1, "coord": [u, v], "period": period})
    # Convert log-odds back to probability
    prob_map = 1.0 / (1.0 + np.exp(-log_odds))
    return {"map": prob_map.tolist(), "per_frequency": per_freq_meta}


def analyze_jpeg_compression(input_str: str) -> str:
    """
    Analyze JPEG compression artifacts and quantization tables.
    Returns format/mode/size plus Sac score (JPEGness indicator).

    Uses raw JPEG DCT coefficients when jpegio is available for accuracy.
    """
    image_path = input_str.strip()
    try:
        from PIL import Image

        img = Image.open(image_path)
        result = {
            "tool": "analyze_jpeg_compression",
            "status": "completed",
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "jpeglib_available": _HAS_JPEGLIB,
        }
        if img.format == "JPEG":
            # Try raw coefficient access first (like MATLAB's jpeg_read)
            jpeg_data = _read_jpeg_raw(image_path)
            if jpeg_data is not None:
                # jpeglib already returns blocks in format (n_blocks, 8, 8)
                coef_blocks = jpeg_data["Y"]
                sac = _sac_score_raw(coef_blocks)
            else:
                gray = np.array(img.convert("L"), dtype=np.float32)
                sac = _sac_score_pixels(gray)
            result.update({"is_jpeg": True, "sac_score": sac})
        else:
            result.update({"is_jpeg": False, "note": f"Image format is {img.format}, not JPEG"})
        return json.dumps(result)
    except Exception as e:  # pragma: no cover - defensive
        return json.dumps(
            {
                "tool": "analyze_jpeg_compression",
                "status": "error",
                "error": str(e),
            }
        )


def detect_jpeg_quantization(input_str: str) -> str:
    """
    Extract JPEG quantization tables, estimate quality, and optionally compute
    block-level tamper probability map.

    Uses raw JPEG DCT coefficients when jpegio is available for accuracy.
    This matches MATLAB's jpeg_read() behavior for forensic analysis.

    Accepts either a plain path string or a JSON payload:
    {
        "path": "...",
        "include": {
            "primary_table": true,
            "truncation_mask": true,
            "block_map": true,
            "per_frequency": false
        }
    }
    If omitted, heavy fields (primary table/mask, block map, per-frequency metadata) are skipped.

    Output fields:
    - quantization_tables: Extracted JPEG quantization tables
    - quality_estimates: Estimated JPEG quality from quant tables
    - sac_score: JPEGness indicator (higher = stronger JPEG artifacts)
    - estimated_primary_quantization: (optional) Factor-histogram based quant estimation
    - block_map: (optional) Per-block tamper probability map
    - coefficient_source: "raw_coefficients" or "pixel_domain"
    """
    image_path, include = _parse_request(input_str)
    want_primary_table = bool(include.get("primary_table"))
    want_trunc_mask = bool(include.get("truncation_mask"))
    want_block_map = bool(include.get("block_map"))
    want_per_freq = bool(include.get("per_frequency", True))
    try:
        from PIL import Image

        img = Image.open(image_path)
        result = {
            "tool": "detect_jpeg_quantization",
            "status": "completed",
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "jpeglib_available": _HAS_JPEGLIB,
        }
        if img.format != "JPEG":
            result["is_jpeg"] = False
            result["note"] = f"Image format is {img.format}, not JPEG."
            return json.dumps(result)

        # Try raw coefficient access (like MATLAB's jpeg_read)
        jpeg_data = _read_jpeg_raw(image_path)
        use_raw = jpeg_data is not None

        # Get quantization tables
        if use_raw:
            parsed_tables = {}
            quality_estimates = {}
            for idx, qtable in enumerate(jpeg_data["qt"]):
                arr = np.array(qtable, dtype=np.int32)
                if arr.size == 64:
                    arr = arr.reshape(8, 8)
                parsed_tables[str(idx)] = arr.tolist()
                if idx == 0:
                    quality_estimates[str(idx)] = _estimate_quality(arr)
        else:
            qtables = img.quantization or {}
            parsed_tables = {}
            quality_estimates = {}
            for idx, table in qtables.items():
                arr = np.array(table, dtype=np.int32).reshape(8, 8)
                parsed_tables[str(idx)] = arr.tolist()
                if idx == 0:
                    quality_estimates[str(idx)] = _estimate_quality(arr)

        # Get DCT coefficients (Y channel for luminance analysis)
        gray = np.array(img.convert("L"), dtype=np.float32)
        # Pixel-domain DCTs (bdct) are needed for factor-histogram quant estimation,
        # which in the MATLAB reference is performed on decompressed pixels, not on
        # already-quantized raw coefficients.
        pixel_blocks, pixel_grid = _block_dcts_from_pixels(gray)

        if use_raw:
            # jpeglib already returns blocks in format (n_blocks, 8, 8)
            raw_blocks = jpeg_data["Y"]  # Already (n_blocks, 8, 8)
            raw_grid_shape = jpeg_data.get("grid_shape")
            if raw_grid_shape is None:
                # Fallback: calculate from blocks
                n_blocks = raw_blocks.shape[0]
                # Estimate grid shape (assume roughly square)
                h_blocks = int(np.sqrt(n_blocks))
                w_blocks = (n_blocks + h_blocks - 1) // h_blocks
                raw_grid_shape = (h_blocks, w_blocks)
            sac = _sac_score_raw(raw_blocks)
            coef_source = "raw_coefficients"
            blocks_for_map = raw_blocks
            grid_for_map = raw_grid_shape
        else:
            sac = _sac_score_pixels(gray)
            coef_source = "pixel_domain"
            blocks_for_map = pixel_blocks
            grid_for_map = pixel_grid

        # Primary estimation always uses pixel-domain bdct per MATLAB fh_jpgstep.m
        blocks_for_primary = pixel_blocks
        grid_for_primary = pixel_grid

        # Primary table estimation (use pixel-domain DCTs per MATLAB fh_jpgstep.m)
        if want_primary_table or want_trunc_mask:
            primary_q = _estimate_primary_qtable_raw(blocks_for_primary, grid_for_primary, gray=gray)
            primary_out = {}
            if want_primary_table:
                primary_out["table"] = primary_q.get("table")
            if want_trunc_mask:
                primary_out["mask"] = primary_q.get("mask")
            if primary_out:
                result["estimated_primary_quantization"] = primary_out

        # Block map
        if want_block_map:
            block_map = _block_level_map(blocks_for_map, grid_for_map, include_per_frequency=want_per_freq)
        else:
            block_map = None

        result.update(
            {
                "is_jpeg": True,
                "quantization_tables": parsed_tables,
                "quality_estimates": quality_estimates,
                "sac_score": sac,
                "coefficient_source": coef_source,
                **({"block_map": block_map} if want_block_map else {}),
            }
        )
        return json.dumps(result)
    except Exception as e:  # pragma: no cover - defensive
        return json.dumps(
            {
                "tool": "detect_jpeg_quantization",
                "status": "error",
                "error": str(e),
            }
        )


__all__ = ["analyze_jpeg_compression", "detect_jpeg_quantization"]
