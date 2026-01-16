import os
from contextlib import nullcontext
import numpy as np
import cv2
from scipy.stats import skew, kurtosis

try:
    import torch
except ImportError:
    torch = None


def _get_default_drunet():
    """Load DRUNet grayscale model from default weights path."""
    if torch is None:
        return None

    try:
        from example_tools.residuals.drunet import load_drunet_gray

        weights_path = os.path.join(
            os.path.dirname(__file__),
            'drunet', 'weights', 'drunet_gray.pth'
        )

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
                    "example_tools/residuals/drunet/weights/drunet_gray.pth"
                )
            print("ResidualExtractor: Using DRUNet denoiser")
        else:
            self.denoiser = denoiser
            if self.denoiser is None:
                raise RuntimeError("Denoiser is required but None was provided.")

        if torch is None:
            raise RuntimeError("PyTorch is required but not available. Please install PyTorch.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.denoiser.to(self.device).eval()

    def _extract_features_torch_fast(self, gray, gray_uint8):
        """
        Fast path: keep denoising + statistics on-device (no large CPU copies).
        Only used when we can run the full frame without tiling.
        """
        if self.denoiser is None or torch is None:
            raise RuntimeError("Torch fast path requires a denoiser and torch.")

        device = self.device
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

        img = gray_uint8.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)
        
        if self.device.type == 'cuda':
            def amp_ctx():
                return torch.amp.autocast('cuda')
        else:
            amp_ctx = nullcontext
        with torch.inference_mode():
            with amp_ctx():
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
