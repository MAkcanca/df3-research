"""
Popescu-Farid CFA Interpolation Forensics

Implements the algorithm from:

  A. C. Popescu and H. Farid,
  "Exposing Digital Forgeries in Color Filter Array Interpolated Images,"
  IEEE Trans. Signal Processing, 2005.

Features:
  - EM estimation of the linear correlation model for each color channel
    (single-channel Gaussian vs uniform mixture).
  - Posterior probability map per channel.
  - Synthetic CFA maps s_r, s_g, s_b for a Bayer pattern.
  - Fourier-domain similarity M(p_c, s_c) per channel.
  - Sliding-window analysis with 50% overlap.
  - Threshold calibration on a set of negative images to get ~0% FPs.
  - Multi-channel fusion (default): window authentic if ANY channel is CFA;
    optional green-only mode: window authentic if GREEN channel is CFA.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.fft import fft2
from PIL import Image


# -------------------------------------------------------------------------
# Parameters and utilities
# -------------------------------------------------------------------------

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_image_files(directory: str) -> List[str]:
    """Return all image files in a directory (non-recursive) with known extensions."""
    paths = []
    for name in os.listdir(directory):
        if name.lower().endswith(IMG_EXTS):
            paths.append(os.path.join(directory, name))
    paths.sort()
    return paths


def load_rgb_image(path: str) -> np.ndarray:
    """
    Load an image from disk and return as float64 RGB array in [0,1].

    Parameters
    ----------
    path : str
        Path to image file.

    Returns
    -------
    img : np.ndarray, shape (H, W, 3), dtype float64
        RGB image, intensities in [0, 1].
    """
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an RGB image at {path}")
    return arr / 255.0


# -------------------------------------------------------------------------
# EM algorithm for the linear CFA correlation model (single channel)
# -------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _neighbor_offsets(N: int) -> Tuple[Tuple[int, int], ...]:
    """Cached (dy,dx) offsets for a given neighborhood radius N."""
    if N < 1:
        raise ValueError("N must be >= 1.")
    offsets: List[Tuple[int, int]] = []
    for dy in range(-N, N + 1):
        for dx in range(-N, N + 1):
            if dy == 0 and dx == 0:
                continue
            offsets.append((dy, dx))
    return tuple(offsets)


def em_probability_map(
    f: np.ndarray,
    N: int = 1,
    sigma0: float = 0.0075,
    p0: float = 1.0 / 256.0,
    max_iter: int = 50,
    tol: float = 1e-5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the posterior probability map w(x,y) that each sample belongs to
    the linearly correlated model M1 via EM (Gaussian vs uniform mixture).

    Model for a single channel f(x,y):
        f(x,y) = sum_{u,v} alpha_{u,v} f(x+u, y+v) + n(x,y)
        where n(x,y) ~ N(0, sigma^2), alpha_{0,0} = 0.
    """

    if f.ndim != 2:
        raise ValueError("em_probability_map expects a single 2D channel.")

    f = f.astype(np.float64, copy=False)
    H, W = f.shape
    if H <= 2 * N or W <= 2 * N:
        raise ValueError("Image too small for N = %d neighborhood." % N)

    # For N=1, there are (2N+1)^2 - 1 = 8 neighbors.
    OFFSETS = _neighbor_offsets(N)
    K = len(OFFSETS)

    # Build design matrix X and observation vector y (vectorized).
    # This matches the original nested-loop ordering (row-major).
    center = f[N : H - N, N : W - N]
    y = center.reshape(-1, 1)
    num_pixels = int(y.shape[0])

    X = np.empty((num_pixels, K), dtype=np.float64)
    for k, (dy, dx) in enumerate(OFFSETS):
        neigh = f[N + dy : H - N + dy, N + dx : W - N + dx]
        X[:, k] = neigh.reshape(-1)

    rng = np.random.default_rng(seed)
    alpha = rng.normal(scale=0.01, size=(K, 1))

    # ---- NEW: enforce lower bound on sigma to avoid degeneracy ----
    EPS_SIGMA = 1e-6
    sigma = float(max(sigma0, EPS_SIGMA))

    WX = np.empty_like(X)
    wy = np.empty_like(y)

    for _ in range(max_iter):
        # E-step: residuals and posterior weights
        r = y - X @ alpha  # (num_pixels,1)

        # Guard sigma here too
        if sigma < EPS_SIGMA:
            sigma = EPS_SIGMA

        coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
        P = coef * np.exp(-0.5 * (r / sigma) ** 2)  # Gaussian likelihood for M1

        # Posterior w = P / (P + p0)
        w = P / (P + p0)

        # M-step: weighted LS for alpha
        np.multiply(X, w, out=WX)
        A = X.T @ WX
        np.multiply(y, w, out=wy)
        b = X.T @ wy

        try:
            alpha_new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            alpha_new, *_ = np.linalg.lstsq(A, b, rcond=None)

        # Update sigma^2 = sum w_i r_i^2 / sum w_i
        r = y - X @ alpha_new
        num = float(np.sum(w * (r ** 2)))
        den = float(np.sum(w))

        if den > 0.0:
            sigma_new = math.sqrt(num / den)
        else:
            sigma_new = sigma

        # Clamp again to avoid zero
        if sigma_new < EPS_SIGMA:
            sigma_new = EPS_SIGMA

        # Convergence check
        diff = np.linalg.norm(alpha_new - alpha)
        norm = np.linalg.norm(alpha)
        if norm > 0 and diff < tol * norm:
            alpha = alpha_new
            sigma = sigma_new
            break

        alpha = alpha_new
        sigma = sigma_new

    # Final posterior with converged alpha
    if sigma < EPS_SIGMA:
        sigma = EPS_SIGMA

    r = y - X @ alpha
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    P = coef * np.exp(-0.5 * (r / sigma) ** 2)
    w = P / (P + p0)

    prob_map = np.zeros_like(f)
    prob_map[N : H - N, N : W - N] = w.reshape(H - 2 * N, W - 2 * N)

    return prob_map, alpha.ravel()



# -------------------------------------------------------------------------
# CFA synthetic maps and Fourier-domain similarity
# -------------------------------------------------------------------------

def synthetic_cfa_map(
    shape: Tuple[int, int],
    channel: str,
    pattern: str = "RGGB",
) -> np.ndarray:
    """
    Build the synthetic binary map s_c(x,y) for a Bayer pattern:

        s_c(x,y) = 0 if CFA at (x,y) is color c
                 = 1 otherwise

    Bayer patterns are specified as a 4-character string:
        'RGGB', 'BGGR', 'GRBG', 'GBRG', etc.

    pattern[0] -> (row%2==0, col%2==0)
    pattern[1] -> (row%2==0, col%2==1)
    pattern[2] -> (row%2==1, col%2==0)
    pattern[3] -> (row%2==1, col%2==1)
    """
    H, W = shape
    if len(pattern) != 4:
        raise ValueError("Bayer pattern string must have length 4, e.g. 'RGGB'.")

    channel = channel.upper()
    if channel not in ("R", "G", "B"):
        raise ValueError("channel must be one of 'R', 'G', 'B'.")

    # Vectorized parity-slice construction (same semantics as the loop version).
    p = pattern.upper()
    s = np.ones((H, W), dtype=np.float64)
    if p[0] == channel:
        s[0::2, 0::2] = 0.0
    if p[1] == channel:
        s[0::2, 1::2] = 0.0
    if p[2] == channel:
        s[1::2, 0::2] = 0.0
    if p[3] == channel:
        s[1::2, 1::2] = 0.0
    return s


@lru_cache(maxsize=128)
def _synthetic_abs_fft(
    shape: Tuple[int, int],
    channel: str,
    pattern: str,
) -> np.ndarray:
    """Cached |FFT2(synthetic_cfa_map)| for a given (shape, channel, pattern)."""
    syn = synthetic_cfa_map(shape, channel, pattern=pattern)
    Fs_abs = np.abs(fft2(syn))
    # Treat as immutable cache payload.
    Fs_abs.setflags(write=False)
    return Fs_abs


def similarity_measure(prob_map: np.ndarray, synthetic_map: np.ndarray) -> float:
    """
    Phase-insensitive similarity between a probability map and its CFA
    synthetic map:

        M(p,s) = sum |F(p)| * |F(s)|  (excluding DC component)

    The DC component (frequency 0,0) is excluded because:
    1. Probability maps are ~1.0 for all textured pixels (high linear correlation)
    2. This creates a massive DC peak that dominates the sum
    3. The CFA-specific signal is in the higher frequencies (2x2 periodicity)
    4. Excluding DC allows the CFA-specific frequencies to determine discrimination
    """
    if prob_map.shape != synthetic_map.shape:
        raise ValueError("prob_map and synthetic_map must have the same shape.")

    Fp = np.abs(fft2(prob_map))
    Fs = np.abs(fft2(synthetic_map))
    
    # Zero out DC component to focus on CFA-specific frequencies
    Fp_no_dc = Fp.copy()
    Fp_no_dc[0, 0] = 0.0
    
    return float(np.sum(Fp_no_dc * Fs))


# -------------------------------------------------------------------------
# Sliding-window analysis
# -------------------------------------------------------------------------

def sliding_window_indices(H: int, W: int, window: int) -> List[Tuple[int, int]]:
    """
    Generate (y,x) indices for sliding windows with 50% overlap.
    stride = window // 2 along each axis.
    """
    if window > H or window > W:
        return [(0, 0)]

    stride = max(1, window // 2)
    indices = []
    y = 0
    while y + window <= H:
        x = 0
        while x + window <= W:
            indices.append((y, x))
            x += stride
        y += stride
    return indices


def analyze_window(
    window: np.ndarray,
    pattern: str = "RGGB",
    em_kwargs: Dict = None,
    channels: Sequence[str] = ("R", "G", "B"),
    return_maps: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run EM + CFA similarity on a single RGB window.

    Returns per-channel:
        'prob_map', 'synthetic', 'M', 'alpha'
    """
    if window.ndim != 3 or window.shape[2] != 3:
        raise ValueError("Expected RGB window of shape (H,W,3).")

    if em_kwargs is None:
        em_kwargs = {}

    H, W, _ = window.shape
    channel_data = {
        "R": window[:, :, 0],
        "G": window[:, :, 1],
        "B": window[:, :, 2],
    }

    chan_list = tuple(str(c).upper() for c in channels)
    for c in chan_list:
        if c not in ("R", "G", "B"):
            raise ValueError("channels must be a subset of ('R','G','B').")

    result: Dict[str, Dict[str, np.ndarray]] = {}
    for cname in chan_list:
        ch = channel_data[cname]
        prob_map, alpha = em_probability_map(ch, **em_kwargs)
        Fs_abs = _synthetic_abs_fft((H, W), cname, pattern)
        # Compute FFT of probability map and exclude DC component
        Fp_abs = np.abs(fft2(prob_map))
        Fp_abs[0, 0] = 0.0  # Exclude DC to focus on CFA-specific frequencies
        M = float(np.sum(Fp_abs * Fs_abs))

        entry: Dict[str, np.ndarray] = {
            "M": np.array(M, dtype=np.float64),
            "alpha": alpha,
        }
        if return_maps:
            entry["prob_map"] = prob_map
            entry["synthetic"] = synthetic_cfa_map((H, W), cname, pattern=pattern)
        result[cname] = entry

    return result


def analyze_image_windows(
    img: np.ndarray,
    window: int = 256,
    pattern: str = "RGGB",
    em_kwargs: Dict = None,
    channels: Sequence[str] = ("R", "G", "B"),
    return_maps: bool = True,
) -> List[Dict]:
    """
    Apply CFA EM analysis to all sliding windows of an image.

    If image is smaller than `window`, the entire image is treated as one window.
    """
    H, W, C = img.shape
    if C != 3:
        raise ValueError("Expected RGB image with 3 channels.")

    if em_kwargs is None:
        em_kwargs = {}

    if H < window or W < window:
        windows = [(0, 0)]
        w_h, w_w = H, W
    else:
        windows = sliding_window_indices(H, W, window)
        w_h = w_w = window

    results = []
    for (yy, xx) in windows:
        sub = img[yy : yy + w_h, xx : xx + w_w, :]
        res = analyze_window(
            sub,
            pattern=pattern,
            em_kwargs=em_kwargs,
            channels=channels,
            return_maps=return_maps,
        )
        entry = {"y": yy, "x": xx, "h": sub.shape[0], "w": sub.shape[1]}
        entry.update(res)
        results.append(entry)

    return results


# -------------------------------------------------------------------------
# Threshold calibration and classification
# -------------------------------------------------------------------------

def calibrate_thresholds(
    negative_image_paths: Sequence[str],
    window: int = 256,
    pattern: str = "RGGB",
    em_kwargs: Dict = None,
) -> Dict[str, float]:
    """
    Estimate per-channel thresholds T_R, T_G, T_B to obtain ~0% false positives
    on a negative set (non-CFA / tampered images).

    Threshold per channel is defined as the maximum M value observed for that
    channel over all windows of all negative images.
    """
    if em_kwargs is None:
        em_kwargs = {}

    Ms = {"R": [], "G": [], "B": []}

    for path in negative_image_paths:
        img = load_rgb_image(path)
        window_results = analyze_image_windows(
            img,
            window=window,
            pattern=pattern,
            em_kwargs=em_kwargs,
        )
        for r in window_results:
            Ms["R"].append(float(r["R"]["M"]))
            Ms["G"].append(float(r["G"]["M"]))
            Ms["B"].append(float(r["B"]["M"]))

    thresholds: Dict[str, float] = {}
    for c in ("R", "G", "B"):
        values = Ms[c]
        if not values:
            raise RuntimeError(f"No M values collected for channel {c}.")
        thresholds[c] = max(values)

    return thresholds


def classify_windows(
    window_results: List[Dict],
    thresholds: Dict[str, float],
    green_only: bool = False,
) -> List[Dict]:
    """
    Add CFA/tampered labels to each window based on per-channel thresholds.

    If green_only=False (default, PF-style):
        channel is CFA-interpolated  <=>  M_c > T_c
        window authentic             <=>  any channel is CFA-interpolated

    If green_only=True:
        channel flags are still computed, but
        window authentic             <=>  GREEN channel is CFA-interpolated.
    """
    classified = []
    for r in window_results:
        M_R = float(r["R"]["M"])
        M_G = float(r["G"]["M"])
        M_B = float(r["B"]["M"])

        chan_cfa = {
            "R": M_R > thresholds["R"],
            "G": M_G > thresholds["G"],
            "B": M_B > thresholds["B"],
        }

        if green_only:
            authentic = chan_cfa["G"]
        else:
            authentic = chan_cfa["R"] or chan_cfa["G"] or chan_cfa["B"]

        out = dict(r)
        out["channel_cfa"] = chan_cfa
        out["authentic"] = authentic
        classified.append(out)

    return classified


def classify_image(
    img_path: str,
    thresholds: Dict[str, float],
    window: int = 256,
    pattern: str = "RGGB",
    em_kwargs: Dict = None,
    green_only: bool = False,
) -> Dict:
    """
    Run full sliding-window Popescu-Farid-style detector on a single image.

    Returns:
        {
          "image_path": str,
          "windows": [...],
          "image_authentic": bool,
        }
    """
    img = load_rgb_image(img_path)
    window_results = analyze_image_windows(
        img,
        window=window,
        pattern=pattern,
        em_kwargs=em_kwargs,
    )
    classified = classify_windows(window_results, thresholds, green_only=green_only)

    image_authentic = any(w["authentic"] for w in classified)

    return {
        "image_path": img_path,
        "windows": classified,
        "image_authentic": image_authentic,
    }


__all__ = [
    "IMG_EXTS",
    "list_image_files",
    "load_rgb_image",
    "em_probability_map",
    "synthetic_cfa_map",
    "similarity_measure",
    "sliding_window_indices",
    "analyze_window",
    "analyze_image_windows",
    "calibrate_thresholds",
    "classify_windows",
    "classify_image",
]
