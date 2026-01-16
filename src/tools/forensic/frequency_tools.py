"""
Frequency-domain forensic helpers.
"""

import json

import cv2
import numpy as np
from PIL import Image
from scipy import stats
from scipy.fftpack import dct, fft2, fftshift


def analyze_frequency_domain(input_str: str) -> str:
    """
    Analyze DCT/FFT frequency domain features.
    
    Extracts comprehensive frequency domain features including:
    - DCT coefficient statistics (mean, std)
    - FFT radial profile statistics (mean, std, decay rate)
    - Frequency band energies (low, mid, high)
    - Peakiness metric for detecting upsampling artifacts
    """
    image_path = input_str.strip()
    try:
        img = Image.open(image_path).convert("RGB")
        image = np.array(img)

        # Convert to grayscale using cv2 (matches frequency.py)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        features = {}

        # --- DCT Features ---
        # Compute 2D DCT (orthonormal normalization)
        dct_map = dct(dct(gray.T, norm='ortho').T, norm='ortho')

        # Histogram of DCT coefficients (excluding DC component)
        dct_coeffs = dct_map.flatten()
        dct_coeffs = dct_coeffs[1:]  # Remove DC component

        # Statistics on DCT coefficients (using absolute values)
        dct_abs = np.abs(dct_coeffs)
        features['dct_mean'] = float(np.mean(dct_abs))
        features['dct_std'] = float(np.std(dct_abs))

        # --- FFT Features ---
        # Compute 2D FFT and shift to center DC component
        f = fft2(gray.astype(np.float64))  # Use float64 for better precision
        fshift = fftshift(f)

        # Use log-magnitude spectrum (20*log10) for consistent normalization
        # This matches the noiseprint extractor and provides better dynamic range
        magnitude_spectrum = 20 * np.log10(np.abs(fshift) + 1e-10)

        # Azimuthal average (Radial Profile)
        # This computes the average magnitude at each radial distance from center
        h, w = magnitude_spectrum.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        r = np.sqrt(x**2 + y**2)
        r = r.astype(int)

        # Compute radial profile: average magnitude at each radius
        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / np.maximum(nr, 1)

        # Remove zero-radius (DC component) for better statistics
        if len(radial_profile) > 1:
            radial_profile_nonzero = radial_profile[1:]
        else:
            radial_profile_nonzero = radial_profile

        # Summary stats of radial profile
        features['fft_radial_mean'] = float(np.mean(radial_profile_nonzero))
        features['fft_radial_std'] = float(np.std(radial_profile_nonzero))

        # Improved radial decay metric: fit linear slope instead of assuming monotonic decay
        # This is more robust to non-monotonic profiles (e.g., peaks at intermediate frequencies)
        n = len(radial_profile_nonzero)
        if n >= 3:
            # Fit linear regression to log(radius) vs magnitude to estimate decay rate
            # This captures the overall trend without assuming monotonicity
            radii = np.arange(1, n + 1, dtype=np.float64)
            # Use log(radius) to better capture power-law decay
            log_radii = np.log(radii + 1e-10)

            # Fit linear model: magnitude = a * log(radius) + b
            # Negative slope indicates decay (typical for natural images)
            # Positive slope indicates high-frequency emphasis (typical for upsampled images)
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    log_radii, radial_profile_nonzero
                )
                features['fft_radial_decay'] = float(slope)  # Decay rate (negative = decay)
                features['fft_radial_decay_r2'] = float(r_value**2)  # Goodness of fit
            except Exception:
                # Fallback: simple difference if regression fails
                features['fft_radial_decay'] = float(
                    radial_profile_nonzero[0] - radial_profile_nonzero[-1]
                )
                features['fft_radial_decay_r2'] = 0.0
        else:
            features['fft_radial_decay'] = 0.0
            features['fft_radial_decay_r2'] = 0.0

        # Frequency band energies (low, mid, high)
        if n >= 9:  # Ensure enough samples for band division
            # Divide radial profile into 3 bands: low, mid, high frequency
            edges = np.linspace(0, n, 4, dtype=int)  # 3 bands
            low_band = radial_profile_nonzero[edges[0]:edges[1]]
            mid_band = radial_profile_nonzero[edges[1]:edges[2]]
            high_band = radial_profile_nonzero[edges[2]:edges[3]]

            features['fft_low_energy'] = float(np.mean(low_band))
            features['fft_mid_energy'] = float(np.mean(mid_band))
            features['fft_high_energy'] = float(np.mean(high_band))

            # Peakiness: ratio of max to mean (detects sharp peaks from upsampling)
            # High peakiness indicates periodic patterns (upsampling artifacts)
            profile_mean = np.mean(radial_profile_nonzero)
            profile_max = np.max(radial_profile_nonzero)
            features['fft_peakiness'] = float(profile_max / (profile_mean + 1e-10))
        else:
            # Not enough samples for band analysis
            features['fft_low_energy'] = 0.0
            features['fft_mid_energy'] = 0.0
            features['fft_high_energy'] = 0.0
            features['fft_peakiness'] = 0.0

        result = {
            "tool": "analyze_frequency_domain",
            "status": "completed",
            **features,
        }

        return json.dumps(result)
    except Exception as e:  # pragma: no cover - defensive
        return json.dumps(
            {
                "tool": "analyze_frequency_domain",
                "status": "error",
                "error": str(e),
            }
        )


__all__ = ["analyze_frequency_domain"]
