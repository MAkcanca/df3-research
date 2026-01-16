"""
Noise-based forensic helpers.
"""

import os
import json
import threading
from contextlib import nullcontext
import numpy as np
import cv2
from scipy.stats import skew, kurtosis

try:
    import torch
except ImportError:
    torch = None

# Module-level cache for ResidualExtractor to avoid reloading DRUNet on every call
_RESIDUAL_EXTRACTOR_CACHE = None

# Serialize CUDA inference on single-GPU machines to avoid OOM/thrashing under multi-threaded eval.
_DRUNET_CUDA_SEMAPHORE = threading.BoundedSemaphore(int(os.getenv("DF3_CUDA_TOOL_CONCURRENCY", "1")))

# Protect DRUNet/ResidualExtractor initialization from concurrent loads.
_RESIDUAL_EXTRACTOR_LOCK = threading.Lock()


def prewarm_residual_extractor() -> bool:
    """
    Pre-warm the ResidualExtractor (DRUNet) cache by loading it before workers start.
    
    This is useful for multi-worker evaluation to avoid concurrent model loading.
    Returns True if successful, False otherwise.
    """
    try:
        extractor = _get_cached_residual_extractor()
        if extractor is None:
            print("[ResidualExtractor] Pre-warming failed: extractor is None")
            return False
        print("[ResidualExtractor] Model pre-warmed")
        return True
    except Exception as e:
        print(f"[ResidualExtractor] Pre-warming failed with exception: {e}")
        return False


def _get_cached_residual_extractor():
    """
    Get cached ResidualExtractor, creating it only on first call.
    
    This dramatically improves performance by avoiding DRUNet model reload on every inference.
    """
    global _RESIDUAL_EXTRACTOR_CACHE
    
    if _RESIDUAL_EXTRACTOR_CACHE is None:
        with _RESIDUAL_EXTRACTOR_LOCK:
            if _RESIDUAL_EXTRACTOR_CACHE is None:
                _RESIDUAL_EXTRACTOR_CACHE = ResidualExtractor(denoiser='auto')
                print("[ResidualExtractor] Model loaded and cached")
    
    return _RESIDUAL_EXTRACTOR_CACHE


def _is_stateless_gpu_environment():
    """
    Check if we're in a Stateless GPU environment (e.g., Hugging Face Spaces).
    
    In Stateless GPU environments, CUDA must not be initialized in the main process.
    Returns True if we should avoid CUDA initialization in the main process.
    """
    return os.getenv("SPACES_ID") is not None


def _safe_get_device(device=None):
    """
    Safely get device without initializing CUDA in Stateless GPU environments.
    
    In Stateless GPU environments, CUDA initialization is deferred until actual inference.
    This function returns 'cpu' in Stateless GPU environments to avoid CUDA init in main process.
    
    Args:
        device: Optional torch device. If None, will determine device safely.
    
    Returns:
        torch.device: Device to use. 'cpu' in Stateless GPU main process, otherwise CUDA if available.
    """
    if device is not None:
        return device
    
    # In Stateless GPU environments, avoid CUDA initialization in main process
    if _is_stateless_gpu_environment():
        # Return CPU device - CUDA will be initialized later in worker processes
        return torch.device("cpu")
    
    # For non-Stateless environments, check CUDA normally
    if torch is None:
        raise RuntimeError("PyTorch is required but not available.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_default_drunet():
    """Load DRUNet grayscale model from default weights path."""
    if torch is None:
        return None

    try:
        from src.tools.forensic.drunet import load_drunet_gray
        from pathlib import Path

        weights_path = os.path.join(
            os.path.dirname(__file__),
            'drunet', 'weights', 'drunet_gray.pth'
        )

        # Try to auto-download if weights are missing
        if not os.path.exists(weights_path):
            try:
                from src.utils.weight_downloader import ensure_drunet_weights
                weights_path_obj = Path(weights_path)
                success, message = ensure_drunet_weights(weights_path=weights_path_obj, auto_download=True)
                if success:
                    print(f"[DRUNet] {message}")
                else:
                    print(f"[DRUNet] {message}")
                    print(f"[DRUNet] Weights not found at {weights_path} and download failed.")
            except ImportError:
                print("[DRUNet] Weight downloader not available. Weights must be manually downloaded.")
            except Exception as e:
                print(f"[DRUNet] Error downloading weights: {e}")

        if os.path.exists(weights_path):
            return load_drunet_gray(weights_path, noise_level=15)
        else:
            print(f"Warning: DRUNet weights not found at {weights_path}")
            return None
    except Exception as e:
        print(f"Warning: Failed to load DRUNet: {e}")
        return None


class ResidualExtractor:
    def __init__(self, denoiser='auto', 
                 max_tile_size=1024, tile_overlap=64, max_image_size=4096,
                 auto_downscale=True):
        """
        Args:
            denoiser: Denoiser model. Options:
                - 'auto': Load DRUNet automatically (required)
                - torch.nn.Module: Custom denoiser that takes (B,1,H,W) in [0,1]
            max_tile_size (int): Maximum tile size for tiled processing (default: 1024)
            tile_overlap (int): Overlap between tiles to avoid boundary artifacts (default: 64)
            max_image_size (int): Maximum image dimension before auto-downscaling (default: 4096)
            auto_downscale (bool): Automatically downscale very large images (default: True)
        """
        self.max_tile_size = max_tile_size
        self.tile_overlap = tile_overlap
        self.max_image_size = max_image_size
        self.auto_downscale = auto_downscale

        if denoiser == 'auto':
            self.denoiser = _get_default_drunet()
            if self.denoiser is None:
                raise RuntimeError(
                    "DRUNet denoiser is required but not available. "
                    "Please ensure PyTorch is installed and DRUNet weights are available at "
                    "src/tools/forensic/drunet/weights/drunet_gray.pth"
                )
            print("ResidualExtractor: Using DRUNet denoiser")
        else:
            self.denoiser = denoiser
            if self.denoiser is None:
                raise RuntimeError("Denoiser is required but None was provided.")

        if torch is None:
            raise RuntimeError("PyTorch is required but not available. Please install PyTorch.")

        # Use safe device selection to avoid CUDA init in Stateless GPU main process
        self.device = _safe_get_device()
        if getattr(self.device, "type", None) == "cuda":
            with _DRUNET_CUDA_SEMAPHORE:
                self.denoiser.to(self.device).eval()
        else:
            self.denoiser.to(self.device).eval()
        
        # In Stateless GPU environments, device will be CPU initially
        # CUDA will be initialized later when actual inference happens in worker processes

    def _get_inference_device(self):
        """
        Get device for inference, handling Stateless GPU environments.
        
        In Stateless GPU environments, CUDA may become available in worker processes.
        This function checks for CUDA availability at inference time.
        """
        # If we're in a Stateless GPU environment and currently on CPU,
        # check if CUDA is now available (in worker process)
        if _is_stateless_gpu_environment() and self.device.type == 'cpu':
            # In worker processes, CUDA should be available
            if torch.cuda.is_available():
                # Move model to CUDA for this inference
                with _DRUNET_CUDA_SEMAPHORE:
                    self.denoiser = self.denoiser.cuda()
                    self.device = torch.device("cuda")
                return self.device
        return self.device

    def _extract_features_torch_fast(self, gray, gray_uint8):
        """
        Fast path: keep denoising + statistics on-device (no large CPU copies).
        Only used when we can run the full frame without tiling.
        """
        if self.denoiser is None or torch is None:
            raise RuntimeError("Torch fast path requires a denoiser and torch.")

        device = self._get_inference_device()
        if device.type == 'cuda':
            def amp_ctx():
                return torch.amp.autocast('cuda')
        else:
            amp_ctx = nullcontext

        gray_clamped = np.clip(gray, 0, 255)
        gray_t = torch.from_numpy(gray_clamped).to(device=device, dtype=torch.float32)
        gray_4d = gray_t.unsqueeze(0).unsqueeze(0)
        denoise_input = (gray_4d / 255.0).clamp(0.0, 1.0)

        with torch.inference_mode():
            with amp_ctx():
                if device.type == "cuda":
                    with _DRUNET_CUDA_SEMAPHORE:
                        denoised = self.denoiser(denoise_input)
                else:
                    denoised = self.denoiser(denoise_input)

        residual = (gray_4d - denoised * 255.0).float()
        residual_flat = residual.view(-1)
        abs_res = residual_flat.abs()

        mean = residual_flat.mean()
        var = residual_flat.var(unbiased=False)
        std = torch.sqrt(var + 1e-12)

        # Match scipy defaults: bias=True, fisher=True
        centered = residual_flat - mean
        skew_val = (centered.pow(3).mean()) / (std.pow(3) + 1e-12)
        kurt_val = (centered.pow(4).mean()) / (std.pow(4) + 1e-12) - 3.0

        features = {
            'residual_mean': float(mean.item()),
            'residual_std': float(std.item()),
            'residual_skew': float(skew_val.item()),
            'residual_kurtosis': float(kurt_val.item()),
            'residual_energy': float(residual_flat.pow(2).mean().item()),
            'residual_energy_mean': float(abs_res.mean().item()),
            'residual_energy_std': float(abs_res.std(unbiased=False).item()),
            'residual_energy_p95': float(torch.quantile(abs_res, 0.95).item()),
        }

        del residual, residual_flat, abs_res, denoised, gray_t, gray_4d
        return features

    def _run_denoiser_tiled(self, gray_uint8):
        """
        Process large images in tiles to avoid VRAM exhaustion.
        
        Args:
            gray_uint8: HxW uint8 grayscale image.
        Returns:
            denoised image as float32 in [0,255].
        """
        h, w = gray_uint8.shape
        
        # Check if image is small enough to process directly
        if h <= self.max_tile_size and w <= self.max_tile_size:
            return self._run_denoiser_single(gray_uint8)
        
        # Process in tiles with overlap
        denoised = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)  # For averaging overlapping regions
        
        slide = self.max_tile_size - 2 * self.tile_overlap  # Effective slide size
        
        for y0 in range(0, h, slide):
            y_start = max(0, y0 - self.tile_overlap)
            y_end = min(h, y0 + self.max_tile_size - self.tile_overlap)
            
            for x0 in range(0, w, slide):
                x_start = max(0, x0 - self.tile_overlap)
                x_end = min(w, x0 + self.max_tile_size - self.tile_overlap)
                
                # Extract tile
                tile = gray_uint8[y_start:y_end, x_start:x_end]
                
                # Process tile
                tile_denoised = self._run_denoiser_single(tile)
                
                # Determine output region (excluding overlap on first tile)
                out_y_start = y_start if y0 == 0 else y_start + self.tile_overlap
                out_y_end = y_end if y0 + slide >= h else y_end - self.tile_overlap
                out_x_start = x_start if x0 == 0 else x_start + self.tile_overlap
                out_x_end = x_end if x0 + slide >= w else x_end - self.tile_overlap
                
                # Extract corresponding region from denoised tile
                tile_y_start = out_y_start - y_start
                tile_y_end = out_y_end - y_start
                tile_x_start = out_x_start - x_start
                tile_x_end = out_x_end - x_start
                
                tile_output = tile_denoised[tile_y_start:tile_y_end, tile_x_start:tile_x_end]
                
                # Accumulate (for averaging overlapping regions)
                denoised[out_y_start:out_y_end, out_x_start:out_x_end] += tile_output
                counts[out_y_start:out_y_end, out_x_start:out_x_end] += 1.0
        
        # Average overlapping regions
        mask = counts > 0
        denoised[mask] /= counts[mask]
        
        return denoised

    def _run_denoiser_single(self, gray_uint8):
        """
        Process a single image/tile through the denoiser.
        
        Args:
            gray_uint8: HxW uint8 grayscale.
        Returns:
            denoised image as float32 in [0,255].
        """
        if self.denoiser is None or torch is None:
            raise RuntimeError("Denoiser not available.")

        device = self._get_inference_device()
        img = gray_uint8.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
        
        if device.type == 'cuda':
            def amp_ctx():
                return torch.amp.autocast('cuda')
        else:
            amp_ctx = nullcontext
        with torch.inference_mode():
            with amp_ctx():
                if device.type == "cuda":
                    with _DRUNET_CUDA_SEMAPHORE:
                        denoised = self.denoiser(tensor).cpu().numpy()[0, 0]
                else:
                    denoised = self.denoiser(tensor).cpu().numpy()[0, 0]
        
        # Clear tensor from GPU immediately
        del tensor

        denoised = np.clip(denoised * 255.0, 0, 255).astype(np.float32)
        return denoised

    def _run_denoiser(self, gray_uint8):
        """
        Main denoiser entry point with automatic memory management.
        
        Args:
            gray_uint8: HxW uint8 grayscale.
        Returns:
            denoised image as float32 in [0,255].
        """
        if self.denoiser is None or torch is None:
            raise RuntimeError("Denoiser not available.")
        
        h, w = gray_uint8.shape
        
        # Auto-downscale extremely large images
        if self.auto_downscale and (h > self.max_image_size or w > self.max_image_size):
            scale = min(self.max_image_size / h, self.max_image_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            print(f"ResidualExtractor: Downscaling {h}x{w} -> {new_h}x{new_w} to fit in VRAM")
            gray_downscaled = cv2.resize(gray_uint8, (new_w, new_h), interpolation=cv2.INTER_AREA)
            denoised_downscaled = self._run_denoiser_tiled(gray_downscaled)
            # Upscale back to original size
            denoised = cv2.resize(denoised_downscaled, (w, h), interpolation=cv2.INTER_LINEAR)
            return denoised
        
        # Use tiled processing for large images
        return self._run_denoiser_tiled(gray_uint8)

    def extract_features(self, image):
        """
        Extracts residual-based features using DRUNet denoiser.

        Args:
            image (PIL.Image or np.ndarray): Input image.

        Returns:
            dict: Dictionary of features.
        """
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        gray = gray.astype(np.float32)
        gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)

        if self.denoiser is None:
            raise RuntimeError("Denoiser is required but not available.")

        h, w = gray.shape
        use_fast_path = (
            h <= self.max_tile_size and w <= self.max_tile_size and
            not (self.auto_downscale and (h > self.max_image_size or w > self.max_image_size))
        )
        if use_fast_path:
            return self._extract_features_torch_fast(gray, gray_uint8)

        denoised = self._run_denoiser(gray_uint8)

        residual = gray - denoised
        abs_res = np.abs(residual)

        features = {
            # legacy stats
            'residual_mean': float(np.mean(residual)),
            'residual_std': float(np.std(residual)),
            'residual_skew': float(skew(residual.flatten())),
            'residual_kurtosis': float(kurtosis(residual.flatten())),
            'residual_energy': float(np.sum(residual ** 2)) / residual.size,
            'residual_energy_mean': float(abs_res.mean()),
            'residual_energy_std': float(abs_res.std()),
            'residual_energy_p95': float(np.percentile(abs_res, 95)),
        }

        return features


def extract_noiseprint(input_str: str) -> str:
    """
    Extract camera model fingerprint features (noiseprint).
    """
    image_path = input_str.strip()
    try:
        result = {
            "tool": "extract_noiseprint",
            "status": "completed",
            "note": "Noiseprint extraction requires trained CNN model. This is a placeholder.",
            "image_path": image_path,
        }

        return json.dumps(result)
    except Exception as e:  # pragma: no cover - defensive
        return json.dumps(
            {
                "tool": "extract_noiseprint",
                "status": "error",
                "error": str(e),
            }
        )


def extract_residuals(input_str: str) -> str:
    """
    Extract denoiser residual statistics using DRUNet (deep learning denoiser).
    
    This function applies a state-of-the-art neural network denoiser (DRUNet) to the image
    and analyzes the residual patterns. Returns comprehensive statistics including:
    - residual_mean, residual_std: Basic statistics of the residual distribution
    - residual_skew, residual_kurtosis: Higher-order moments indicating distribution shape
    - residual_energy: Overall energy in the residual signal
    - residual_energy_mean, residual_energy_std, residual_energy_p95: Statistics of absolute residuals
    
    These statistics can reveal manipulation, AI generation artifacts, or compression inconsistencies.
    
    Uses cached ResidualExtractor to avoid reloading DRUNet model on every call.
    
    Args:
        input_str: Path to the image file (string) or dict with 'path'/'image_path' key
        
    Returns:
        JSON string with status and all residual statistics
    """
    # Handle both string and dict inputs (LangGraph may pass dict)
    if isinstance(input_str, dict):
        image_path = input_str.get("path") or input_str.get("image_path") or ""
    else:
        image_path = input_str.strip() if isinstance(input_str, str) else str(input_str)
    
    if not image_path:
        return json.dumps(
            {
                "tool": "extract_residuals",
                "status": "error",
                "error": "No image path provided",
            }
        )
    
    try:
        from PIL import Image
        
        # Load image
        img = Image.open(image_path).convert("RGB")
        
        # Extract features using cached ResidualExtractor (avoids reloading DRUNet)
        extractor = _get_cached_residual_extractor()
        features = extractor.extract_features(img)
        
        result = {
            "tool": "extract_residuals",
            "status": "completed",
            "image_path": image_path,
            **features
        }

        return json.dumps(result)
    except Exception as e:  # pragma: no cover - defensive
        return json.dumps(
            {
                "tool": "extract_residuals",
                "status": "error",
                "error": str(e),
            }
        )


__all__ = ["extract_noiseprint", "extract_residuals"]
