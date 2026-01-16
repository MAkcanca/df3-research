"""
Utility to download and verify TruFor model weights automatically.
"""

import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple
import requests
from tqdm import tqdm


TRUFOR_WEIGHTS_URL = "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"
TRUFOR_WEIGHTS_MD5 = "7bee48f3476c75616c3c5721ab256ff8"
TRUFOR_WEIGHTS_FILENAME = "trufor.pth.tar"

DRUNET_WEIGHTS_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth"
DRUNET_WEIGHTS_FILENAME = "drunet_gray.pth"
# Note: MD5 for drunet_gray.pth is not publicly documented, so we skip MD5 verification


def _calculate_md5(file_path: Path) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _download_file(url: str, destination: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file with progress bar.
    
    Returns:
        True if download successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            if total_size == 0:
                # No content length header
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            else:
                # Show progress bar
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading weights") as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Download error: {e}")
        if destination.exists():
            destination.unlink()  # Remove partial download
        return False


def ensure_trufor_weights(workspace_root: Optional[Path] = None, auto_download: bool = True) -> Tuple[bool, str]:
    """
    Ensure TruFor weights are available, downloading if necessary.
    
    Downloads TruFor_weights.zip from the official source, verifies MD5,
    and extracts trufor.pth.tar to weights/trufor/trufor.pth.tar.
    
    Zip structure: weights/trufor.pth.tar
    Final path: projectroot/weights/trufor/trufor.pth.tar
    MD5 is verified on the zip file (not the tar).
    
    Args:
        workspace_root: Root directory of the workspace. If None, tries to detect it.
        auto_download: If True, automatically download weights if missing.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    if workspace_root is None:
        # Try to detect workspace root
        # Go up from this file: src/utils/weight_downloader.py -> src/utils -> src -> workspace_root
        current_file = Path(__file__)
        workspace_root = current_file.parent.parent.parent
    
    weights_dir = workspace_root / "weights" / "trufor"
    weights_path = weights_dir / TRUFOR_WEIGHTS_FILENAME
    
    # Check if weights already exist
    if weights_path.exists():
        # File exists - we can't verify MD5 since it's for the zip, not the tar
        file_size = weights_path.stat().st_size
        if file_size > 0:
            return True, f"✅ TruFor weights found at {weights_path} ({file_size / 1024 / 1024:.1f} MB)"
        else:
            # Empty file - delete and re-download
            weights_path.unlink()
            print("⚠️  Found empty weights file, re-downloading...")
    
    # Weights don't exist
    if not auto_download:
        return False, (
            f"❌ TruFor weights not found at {weights_path}\n"
            f"   Download from: {TRUFOR_WEIGHTS_URL}\n"
            f"   Extract and place at: {weights_path}"
        )
    
    # Auto-download weights
    print(f"📥 TruFor weights not found. Downloading from {TRUFOR_WEIGHTS_URL}...")
    
    # Create weights directory if it doesn't exist
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Download zip file to temp location
    zip_path = weights_dir / "TruFor_weights.zip"
    
    try:
        # Download the zip file
        if not _download_file(TRUFOR_WEIGHTS_URL, zip_path):
            return False, f"❌ Failed to download weights from {TRUFOR_WEIGHTS_URL}"
        
        # Verify MD5 of the zip file immediately after download
        print("🔐 Verifying download integrity (MD5)...")
        try:
            zip_md5 = _calculate_md5(zip_path)
            if zip_md5.lower() != TRUFOR_WEIGHTS_MD5.lower():
                zip_path.unlink()
                return False, (
                    f"❌ Downloaded zip MD5 mismatch!\n"
                    f"   Expected: {TRUFOR_WEIGHTS_MD5}\n"
                    f"   Got: {zip_md5}\n"
                    f"   The download may be corrupted. Please try again."
                )
            print(f"✅ MD5 verified: {zip_md5}")
        except Exception as e:
            print(f"⚠️  MD5 verification failed: {e}. Continuing with extraction...")
        
        # Extract zip file
        # Zip structure: weights/trufor.pth.tar
        print("📦 Extracting weights...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find the weights file in the zip
            members = zip_ref.namelist()
            weights_member = None
            
            # Look for trufor.pth.tar in the zip (expected: weights/trufor.pth.tar)
            for member in members:
                if member.endswith(TRUFOR_WEIGHTS_FILENAME):
                    weights_member = member
                    break
            
            if not weights_member:
                zip_path.unlink()
                return False, (
                    f"❌ Could not find {TRUFOR_WEIGHTS_FILENAME} in downloaded zip.\n"
                    f"   Zip contents: {members}"
                )
            
            # Extract just the weights file to a temp location
            # zip_ref.extract will create the nested directory structure
            zip_ref.extract(weights_member, weights_dir)
            
            # Move from extracted location to final location
            # e.g., weights/trufor/weights/trufor.pth.tar -> weights/trufor/trufor.pth.tar
            extracted_path = weights_dir / weights_member
            
            if extracted_path != weights_path:
                # Move to final location
                if weights_path.exists():
                    weights_path.unlink()
                shutil.move(str(extracted_path), str(weights_path))
                
                # Clean up any empty directories left from extraction
                try:
                    # Remove the 'weights' folder if it was created inside weights_dir
                    extracted_parent = extracted_path.parent
                    while extracted_parent != weights_dir and extracted_parent.exists():
                        if not any(extracted_parent.iterdir()):
                            extracted_parent.rmdir()
                        extracted_parent = extracted_parent.parent
                except Exception:
                    pass  # Ignore cleanup errors
        
        # Clean up zip file
        zip_path.unlink()
        
        # Verify final file exists and has content
        if weights_path.exists():
            file_size = weights_path.stat().st_size
            if file_size > 0:
                return True, f"✅ TruFor weights downloaded successfully to {weights_path} ({file_size / 1024 / 1024:.1f} MB)"
            else:
                weights_path.unlink()
                return False, "❌ Extracted weights file is empty"
        else:
            return False, f"❌ Failed to extract weights to {weights_path}"
        
    except Exception as e:
        # Clean up on error
        if zip_path.exists():
            zip_path.unlink()
        return False, f"❌ Error downloading/extracting weights: {str(e)}"


def ensure_drunet_weights(weights_path: Optional[Path] = None, auto_download: bool = True) -> Tuple[bool, str]:
    """
    Ensure DRUNet weights are available, downloading if necessary.
    
    Downloads drunet_gray.pth from the GitHub release and places it at the specified path.
    
    Args:
        weights_path: Full path where weights should be located. If None, tries to detect it
                      relative to src/tools/forensic/noise_tools.py
        auto_download: If True, automatically download weights if missing.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    if weights_path is None:
        # Try to detect the path relative to noise_tools.py
        # Go from src/utils/weight_downloader.py -> src/utils -> src -> src/tools/forensic/drunet/weights
        current_file = Path(__file__)
        workspace_root = current_file.parent.parent.parent
        weights_path = workspace_root / "tools" / "forensic" / "drunet" / "weights" / DRUNET_WEIGHTS_FILENAME
    
    # Ensure weights_path is a Path object
    if isinstance(weights_path, str):
        weights_path = Path(weights_path)
    
    # Check if weights already exist
    if weights_path.exists():
        file_size = weights_path.stat().st_size
        if file_size > 0:
            return True, f"✅ DRUNet weights found at {weights_path} ({file_size / 1024 / 1024:.1f} MB)"
        else:
            # Empty file - delete and re-download
            weights_path.unlink()
            print("⚠️  Found empty weights file, re-downloading...")
    
    # Weights don't exist
    if not auto_download:
        return False, (
            f"❌ DRUNet weights not found at {weights_path}\n"
            f"   Download from: {DRUNET_WEIGHTS_URL}\n"
            f"   Place at: {weights_path}"
        )
    
    # Auto-download weights
    print(f"📥 DRUNet weights not found. Downloading from {DRUNET_WEIGHTS_URL}...")
    
    # Create weights directory if it doesn't exist
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download the weights file directly
        if not _download_file(DRUNET_WEIGHTS_URL, weights_path):
            return False, f"❌ Failed to download weights from {DRUNET_WEIGHTS_URL}"
        
        # Verify final file exists and has content
        if weights_path.exists():
            file_size = weights_path.stat().st_size
            if file_size > 0:
                return True, f"✅ DRUNet weights downloaded successfully to {weights_path} ({file_size / 1024 / 1024:.1f} MB)"
            else:
                weights_path.unlink()
                return False, "❌ Downloaded weights file is empty"
        else:
            return False, f"❌ Failed to download weights to {weights_path}"
        
    except Exception as e:
        # Clean up on error
        if weights_path.exists():
            weights_path.unlink()
        return False, f"❌ Error downloading weights: {str(e)}"


if __name__ == "__main__":
    # Test the downloader
    success, message = ensure_trufor_weights(auto_download=True)
    print(message)
