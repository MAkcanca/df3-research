"""
Create a JSONL dataset file from FaceForensicsPP frames folder.

Samples images randomly from the frames directory and creates a dataset file
in the same format as other evaluation datasets.
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple


def find_all_images(root_dir: Path) -> List[Tuple[Path, str]]:
    """
    Recursively find all image files in directory tree.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of tuples: (image_path, label) where label is "real" or "fake"
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    images = []
    
    # Check if root_dir has fake/real subdirectories
    fake_dir = root_dir / "fake"
    real_dir = root_dir / "real"
    
    if fake_dir.exists() and real_dir.exists():
        # Structure: root/fake/... and root/real/...
        for img_path in fake_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                images.append((img_path, "fake"))
        
        for img_path in real_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                images.append((img_path, "real"))
    else:
        # Flat structure or different organization
        # Try to infer label from parent directory name
        for img_path in root_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                # Check parent directories for "fake" or "real"
                parts = img_path.parts
                label = "real"  # default
                for part in parts:
                    if part.lower() == "fake":
                        label = "fake"
                        break
                    elif part.lower() == "real":
                        label = "real"
                        break
                images.append((img_path, label))
    
    return images


def create_dataset(
    frames_dir: Path,
    output_path: Path,
    limit: int = None,
    seed: int = None,
    relative_to: Path = None
) -> None:
    """
    Create a JSONL dataset file from FaceForensicsPP frames.
    
    Args:
        frames_dir: Directory containing FaceForensicsPP frames (with fake/real subdirs)
        output_path: Path to output JSONL file
        limit: Maximum number of samples to include (None = all)
        seed: Random seed for reproducibility
        relative_to: Base path for relative image paths (default: output_path.parent)
    """
    if not frames_dir.exists():
        print(f"Error: Frames directory does not exist: {frames_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Find all images
    print(f"Scanning for images in {frames_dir}...")
    images = find_all_images(frames_dir)
    
    if not images:
        print(f"Error: No images found in {frames_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(images)} images")
    
    # Count by label
    fake_count = sum(1 for _, label in images if label == "fake")
    real_count = sum(1 for _, label in images if label == "real")
    print(f"  Fake: {fake_count}")
    print(f"  Real: {real_count}")
    
    # Randomly sample if limit is set
    if limit is not None and limit < len(images):
        print(f"Randomly sampling {limit} images...")
        images = random.sample(images, limit)
        
        # Re-count after sampling
        fake_count = sum(1 for _, label in images if label == "fake")
        real_count = sum(1 for _, label in images if label == "real")
        print(f"  Sampled Fake: {fake_count}")
        print(f"  Sampled Real: {real_count}")
    
    # Determine base path for relative paths
    if relative_to is None:
        relative_to = output_path.parent
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSONL file
    print(f"Writing dataset to {output_path}...")
    with output_path.open("w", encoding="utf-8") as f:
        for idx, (img_path, label) in enumerate(sorted(images), 1):
            # Create relative path from relative_to to image
            try:
                rel_path = img_path.relative_to(relative_to)
            except ValueError:
                # If image is not under relative_to, use absolute path
                rel_path = img_path
            
            # Create unique ID
            sample_id = f"ffpp-{idx:05d}"
            
            # Create record
            record = {
                "id": sample_id,
                "image": str(rel_path).replace("\\", "/"),  # Use forward slashes
                "label": label,
                "meta": {
                    "source": "FaceForensicsPP",
                    "original_path": str(img_path)
                }
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Created dataset with {len(images)} samples")
    print(f"  Fake: {fake_count}")
    print(f"  Real: {real_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a JSONL dataset from FaceForensicsPP frames folder"
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default="FaceForensicsPP_frames",
        help="Directory containing FaceForensicsPP frames (default: FaceForensicsPP_frames)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path (e.g., data/ffpp_dataset.jsonl)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to include (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--relative_to",
        type=str,
        default=None,
        help="Base path for relative image paths (default: output file's parent directory)"
    )
    
    args = parser.parse_args()
    
    frames_dir = Path(args.frames_dir)
    output_path = Path(args.output)
    relative_to = Path(args.relative_to) if args.relative_to else None
    
    create_dataset(
        frames_dir=frames_dir,
        output_path=output_path,
        limit=args.limit,
        seed=args.seed,
        relative_to=relative_to
    )


if __name__ == "__main__":
    main()
