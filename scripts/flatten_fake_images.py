#!/usr/bin/env python3
"""
Move all fake images from nested subdirectories to a single folder.
"""
import os
import shutil
from pathlib import Path
from collections import Counter

def flatten_fake_images(source_dir: str, target_dir: str = None):
    """
    Move all fake images from nested subdirectories to a single folder.
    
    Args:
        source_dir: Root directory containing fake subdirectories
        target_dir: Target directory to move all images to (defaults to source_dir/fake/all)
    """
    source_path = Path(source_dir)
    fake_dir = source_path / "fake"
    
    if not fake_dir.exists():
        print(f"Error: {fake_dir} does not exist")
        return
    
    # Default target is fake/all
    if target_dir is None:
        target_path = fake_dir / "all"
    else:
        target_path = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {target_path}")
    
    # Find all PNG files in fake subdirectories (excluding target_dir if it's inside fake)
    image_files = []
    for subdir in fake_dir.iterdir():
        if subdir.is_dir() and subdir.name != target_path.name:
            for img_file in subdir.rglob("*.png"):
                image_files.append(img_file)
    
    print(f"Found {len(image_files)} image files to move")
    
    # Track filename conflicts
    filename_counts = Counter()
    moved_count = 0
    conflict_count = 0
    
    for img_file in image_files:
        filename = img_file.name
        
        # Check if file already exists in target
        target_file = target_path / filename
        
        if target_file.exists():
            # Handle conflict by adding subdirectory name prefix
            subdir_name = img_file.parent.name
            new_filename = f"{subdir_name}_{filename}"
            target_file = target_path / new_filename
            conflict_count += 1
            print(f"Conflict: {filename} -> {new_filename}")
        
        # Move file
        try:
            shutil.move(str(img_file), str(target_file))
            moved_count += 1
            if moved_count % 100 == 0:
                print(f"Moved {moved_count} files...")
        except Exception as e:
            print(f"Error moving {img_file}: {e}")
    
    print(f"\nCompleted!")
    print(f"Total files moved: {moved_count}")
    print(f"Files with conflicts (renamed): {conflict_count}")
    print(f"All fake images are now in: {target_path}")

if __name__ == "__main__":
    import sys
    
    source = r"C:\Users\prota\Repos\df3\FaceForensicsPP_frames"
    
    if len(sys.argv) > 1:
        source = sys.argv[1]
    
    if len(sys.argv) > 2:
        target = sys.argv[2]
        flatten_fake_images(source, target)
    else:
        flatten_fake_images(source)
