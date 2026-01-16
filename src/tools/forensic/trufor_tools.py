"""
TruFor: AI-driven image forgery detection and localization.

TruFor is an AI-driven solution for digital image forensics that combines
high-level RGB features with low-level noise-sensitive fingerprints (Noiseprint++)
through a transformer-based fusion architecture.

Original TruFor Work:
Research Group of University Federico II of Naples ('GRIP-UNINA')
https://github.com/grip-unina/TruFor

Reference Bibtex:
@InProceedings{Guillaro_2023_CVPR,
    author    = {Guillaro, Fabrizio and Cozzolino, Davide and Sud, Avneesh and Dufour, Nicholas and Verdoliva, Luisa},
    title     = {TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {20606-20615}
}
"""

import base64
import json
import os
import threading
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

# Try to import spaces module for ZeroGPU support (Hugging Face Spaces)
# This must be at module level for HF Spaces to detect @spaces.GPU decorators
try:
    import spaces
    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False
    spaces = None

# Module-level model cache to avoid reloading on every call
_TRUFOR_MODEL_CACHE = None
_TRUFOR_MODEL_DEVICE = None
_TRUFOR_CONFIG = None

# Serialize CUDA usage on single-GPU machines to avoid OOM/thrashing when tools are called concurrently.
# (CPU runs remain fully concurrent.)
_TRUFOR_CUDA_SEMAPHORE = threading.BoundedSemaphore(int(os.getenv("DF3_CUDA_TOOL_CONCURRENCY", "1")))

# Protect model cache initialization from concurrent loads.
_TRUFOR_MODEL_LOCK = threading.Lock()


def _get_default_device() -> str:
    """
    Determine the default device for TruFor inference.
    
    Device selection is an infrastructure concern, not controlled by the LLM.
    Uses GPU if available, otherwise CPU.
    
    Can be overridden via DF3_TRUFOR_DEVICE environment variable.
    """
    # Allow override via environment variable
    env_device = os.getenv("DF3_TRUFOR_DEVICE")
    if env_device:
        return env_device
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    
    return "cpu"


def prewarm_trufor_model(device: str = None) -> bool:
    """
    Pre-warm the TruFor model cache by loading it before workers start.
    
    This is useful for multi-worker evaluation to avoid concurrent model loading.
    
    Args:
        device: Optional device override. If None, uses _get_default_device().
    
    Returns True if successful, False otherwise.
    """
    try:
        target_device = device if device is not None else _get_default_device()
        model, config, actual_device_or_error = _get_cached_trufor_model(target_device)
        if model is None:
            print(f"[TruFor] Pre-warming failed: {actual_device_or_error}")
            return False
        print(f"[TruFor] Model pre-warmed on device: {actual_device_or_error}")
        return True
    except Exception as e:
        print(f"[TruFor] Pre-warming failed with exception: {e}")
        return False


def _get_cached_trufor_model(device: str):
    """
    Get cached TruFor model, loading it only on first call.
    
    If a model is already cached on a different device, it will be reused
    (inference will run on the cached device). This avoids expensive reloads
    when the LLM requests different GPU settings.
    
    This dramatically improves performance by avoiding model reload on every inference.
    """
    global _TRUFOR_MODEL_CACHE, _TRUFOR_MODEL_DEVICE, _TRUFOR_CONFIG
    
    # Fast path: return cached model if available (reuse regardless of device)
    if _TRUFOR_MODEL_CACHE is not None:
        if _TRUFOR_MODEL_DEVICE != device:
            # Model cached on different device - reuse it anyway to avoid reload
            # Inference will run on the cached device
            pass  # Fall through to return cached model
        return _TRUFOR_MODEL_CACHE, _TRUFOR_CONFIG, _TRUFOR_MODEL_DEVICE

    # Slow path: need to load model (protected by lock to avoid concurrent loads)
    with _TRUFOR_MODEL_LOCK:
        # Double-check after acquiring lock
        if _TRUFOR_MODEL_CACHE is not None:
            return _TRUFOR_MODEL_CACHE, _TRUFOR_CONFIG, _TRUFOR_MODEL_DEVICE
        
        # Need to load model - import dependencies
        try:
            import torch
            import sys
            from pathlib import Path
            
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from src.tools.forensic.trufor_support.config import update_config, _C as config
            from src.tools.forensic.trufor_support.models.cmx.builder_np_conf import myEncoderDecoder as confcmx
            
            # Find weights file
            workspace_root = Path(__file__).parent.parent.parent.parent
            weights_path = workspace_root / "weights" / "trufor" / "trufor.pth.tar"
            
            # Try to download weights if missing
            if not weights_path.exists():
                try:
                    from src.utils.weight_downloader import ensure_trufor_weights
                    success, message = ensure_trufor_weights(workspace_root=workspace_root, auto_download=True)
                    if not success:
                        return None, None, f"TruFor weights not found and download failed.\n{message}"
                    if not weights_path.exists():
                        return None, None, f"TruFor weights still not found at {weights_path} after download attempt."
                except ImportError:
                    return None, None, (
                        f"TruFor weights not found at {weights_path}.\n"
                        f"Please download from: https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip\n"
                        f"Extract and place at: {weights_path}"
                    )
            
            # Set up config
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument('-gpu', '--gpu', type=int, default=0)
            parser.add_argument('-in', '--input', type=str, default='')
            parser.add_argument('-out', '--output', type=str, default='')
            parser.add_argument('opts', nargs=argparse.REMAINDER, default=[])
            args = parser.parse_args([])
            update_config(config, args)
            
            # Set device-specific settings
            if device != 'cpu':
                import torch.backends.cudnn as cudnn
                cudnn.benchmark = config.CUDNN.BENCHMARK
                cudnn.deterministic = config.CUDNN.DETERMINISTIC
                cudnn.enabled = config.CUDNN.ENABLED
            
            # Load model
            if config.MODEL.NAME == 'detconfcmx':
                model = confcmx(cfg=config)
            else:
                return None, None, f"Unsupported model: {config.MODEL.NAME}"
            
            # Load checkpoint
            checkpoint = torch.load(
                str(weights_path),
                map_location=torch.device(device),
                weights_only=False,
            )
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            model.eval()
            
            # Cache the model
            _TRUFOR_MODEL_CACHE = model
            _TRUFOR_MODEL_DEVICE = device
            _TRUFOR_CONFIG = config
            
            print(f"[TruFor] Model loaded and cached on device: {device}")
            return model, config, device
            
        except ImportError as e:
            return None, None, f"TruFor model architecture not found: {str(e)}"
        except Exception as e:
            return None, None, str(e)


def _parse_request(input_str: str) -> Tuple[str, bool]:
    """
    Accept plain path or JSON payload:
    {
        "path": "/path/to/image.jpg",
        "return_map": true    # Include base64 PNG localization map
    }
    
    Note: GPU selection is handled internally via _get_default_device(),
    not exposed to callers (LLM should not control infrastructure).
    """
    default_return_map = False
    try:
        data = json.loads(input_str)
        if isinstance(data, dict):
            path = data.get("path", input_str).strip()
            return_map = bool(data.get("return_map", default_return_map))
            return path, return_map
    except Exception:
        pass
    return input_str.strip(), default_return_map


def _encode_png(arr: np.ndarray) -> str:
    """Encode numpy array as base64 PNG."""
    # Normalize to 0-255
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(arr, mode='L')
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"




def _perform_trufor_internal(image_path: str, want_map: bool) -> str:
    """
    Internal function that performs TruFor analysis.
    This is separated so it can be wrapped with @spaces.GPU for ZeroGPU support.
    
    Uses cached model to avoid reloading on every call.
    Device selection is handled internally via _get_default_device().
    """
    # Device is determined internally, not by caller/LLM
    requested_device = _get_default_device()
    
    try:
        import torch
        from torch.nn import functional as F
        from torch.utils.data import DataLoader

        # Get cached model (loads on first call, reuses cached model regardless of device)
        # Returns (model, config, actual_device) on success, or (None, None, error_string) on failure
        model, config, actual_device_or_error = _get_cached_trufor_model(requested_device)
        
        # Check if it's an error (model is None means error, third value is error string)
        if model is None:
            return json.dumps({
                "tool": "perform_trufor",
                "status": "error",
                "error": actual_device_or_error,
            })
        
        # Use the actual device the model is cached on
        device = actual_device_or_error

        # Serialize CUDA usage if running on GPU (helps with multi-threaded evaluation on a single GPU).
        cuda_ctx = _TRUFOR_CUDA_SEMAPHORE if device != 'cpu' else nullcontext()
        with cuda_ctx:
            
            # Import data loader
            import sys
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from src.tools.forensic.trufor_support.data_core import myDataset
            
            # Create dataset for this image
            test_dataset = myDataset(list_img=[image_path])
            testloader = DataLoader(test_dataset, batch_size=1)
            
            # Process image
            with torch.no_grad():
                for rgb, path in testloader:
                    rgb = rgb.to(device)
                    
                    pred, conf, det, npp = model(rgb)
                    
                    det_score = torch.sigmoid(det).item()
                    
                    pred = torch.squeeze(pred, 0)
                    pred = F.softmax(pred, dim=0)[1]
                    pred_np = pred.cpu().numpy()
                    
                    # Clean up GPU memory (but keep model cached)
                    del rgb, pred, conf, det, npp
                    if device != 'cpu':
                        torch.cuda.empty_cache()
                    
                    # Prepare results
                    manipulation_prob = float(det_score)
                    detection_score = float(det_score)
                    
                    localization_map_encoded = None
                    map_size = None
                    
                    if want_map:
                        # Normalize prediction map to 0-1 range for visualization
                        pred_normalized = pred_np
                        if pred_normalized.max() > 1.0:
                            pred_normalized = pred_normalized / pred_normalized.max()
                        
                        localization_map_encoded = _encode_png((pred_normalized * 255).astype(np.uint8))
                        map_size = (pred_np.shape[1], pred_np.shape[0])  # (width, height)
                    
                    result = {
                        "tool": "perform_trufor",
                        "status": "completed",
                        "image_path": image_path,
                        "manipulation_probability": manipulation_prob,
                        "detection_score": detection_score,
                        "localization_map": localization_map_encoded,
                        "localization_map_size": map_size,
                        "note": (
                            "TruFor combines RGB features with Noiseprint++ for forgery detection. "
                            "manipulation_probability indicates the likelihood of image manipulation (0-1). "
                            "Higher values suggest greater probability of forgery."
                        ),
                    }
                    
                    return json.dumps(result)
        
        # If we get here, no image was processed
        return json.dumps({
            "tool": "perform_trufor",
            "status": "error",
            "error": "No image was processed",
        })
        
    except ImportError as e:
        return json.dumps({
            "tool": "perform_trufor",
            "status": "error",
            "error": f"Missing dependencies: {e}. Please install PyTorch: pip install torch torchvision",
        })
    except Exception as e:
        return json.dumps({
            "tool": "perform_trufor",
            "status": "error",
            "error": str(e),
        })


# Module-level GPU function for ZeroGPU detection at startup
# This must be at module level for Hugging Face Spaces to detect @spaces.GPU
if SPACES_AVAILABLE:
    @spaces.GPU(duration=120)  # 120 seconds max for TruFor inference
    def _trufor_gpu_wrapper(image_path: str, want_map: bool) -> str:
        """
        GPU-wrapped version of TruFor for ZeroGPU Spaces.
        This function is detected by HF Spaces at startup.
        """
        return _perform_trufor_internal(image_path, want_map)
else:
    # Fallback for non-Spaces environments
    def _trufor_gpu_wrapper(image_path: str, want_map: bool) -> str:
        """Fallback wrapper when spaces module is not available."""
        return _perform_trufor_internal(image_path, want_map)


def perform_trufor(input_str: str) -> str:
    """
    Run TruFor forgery detection and localization on an image.
    
    TruFor combines RGB features with Noiseprint++ to detect and localize
    image forgeries using a transformer-based fusion architecture.
    
    Returns JSON with:
    - manipulation_probability: Overall probability of manipulation (0-1)
    - detection_score: Detection confidence score
    - localization_map: Base64 PNG of forgery localization map (optional)
    - localization_map_size: Size of the map (width, height)
    
    Device selection is handled internally based on CUDA availability.
    Can be overridden via DF3_TRUFOR_DEVICE environment variable.
    
    This function supports both traditional GPU Spaces and ZeroGPU Spaces on Hugging Face.
    For ZeroGPU, GPU resources are allocated dynamically only when needed.
    """
    image_path, want_map = _parse_request(input_str)
    
    # Use GPU wrapper if spaces module is available and we're using GPU
    # Device selection is internal - check if default device is GPU
    if SPACES_AVAILABLE and _get_default_device().startswith("cuda"):
        # Use the module-level decorated function for ZeroGPU
        # This ensures HF Spaces detects @spaces.GPU at startup
        return _trufor_gpu_wrapper(image_path, want_map)
    else:
        # CPU mode or spaces module not available - use internal function directly
        return _perform_trufor_internal(image_path, want_map)


__all__ = ["perform_trufor", "prewarm_trufor_model"]
