#!/usr/bin/env python3
"""
Rename all fake images to a simple sequential format.
Options: sequential numbers, UUID, or numberletter_number format.
"""
import os
import uuid
import string
import random
from pathlib import Path
from typing import Literal

def rename_to_sequential(source_dir: str, prefix: str = "", start_num: int = 1):
    """
    Rename all images to sequential format: 00001.png, 00002.png, etc.
    
    Args:
        source_dir: Directory containing images
        prefix: Prefix for filenames (default: "" - no prefix)
        start_num: Starting number (default: 1)
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Error: {source_path} does not exist")
        return
    
    # Get all PNG files
    image_files = sorted(list(source_path.glob("*.png")))
    
    if not image_files:
        print(f"No PNG files found in {source_path}")
        return
    
    print(f"Found {len(image_files)} files to rename")
    
    # Determine padding width based on total count
    padding = len(str(len(image_files)))
    
    renamed_count = 0
    
    for idx, img_file in enumerate(image_files, start=start_num):
        if prefix:
            new_name = f"{prefix}_{idx:0{padding}d}.png"
        else:
            new_name = f"{idx:0{padding}d}.png"
        new_path = source_path / new_name
        
        # Skip if already renamed
        if img_file.name == new_name:
            continue
        
        # Handle case where target name already exists
        if new_path.exists() and new_path != img_file:
            # Find next available number
            counter = idx + 1
            while True:
                if prefix:
                    check_path = source_path / f"{prefix}_{counter:0{padding}d}.png"
                else:
                    check_path = source_path / f"{counter:0{padding}d}.png"
                if not check_path.exists():
                    break
                counter += 1
            if prefix:
                new_name = f"{prefix}_{counter:0{padding}d}.png"
            else:
                new_name = f"{counter:0{padding}d}.png"
            new_path = source_path / new_name
        
        try:
            img_file.rename(new_path)
            renamed_count += 1
            if renamed_count % 1000 == 0:
                print(f"Renamed {renamed_count} files...")
        except Exception as e:
            print(f"Error renaming {img_file}: {e}")
    
    print(f"\nCompleted! Renamed {renamed_count} files")
    if prefix:
        print(f"Format: {prefix}_XXXXX.png (where XXXXX is zero-padded number)")
    else:
        print(f"Format: XXXXX.png (where XXXXX is zero-padded number)")

def rename_to_uuid(source_dir: str, prefix: str = ""):
    """
    Rename all images to UUID format: fake_<uuid>.png
    
    Args:
        source_dir: Directory containing images
        prefix: Prefix for filenames (default: "fake")
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Error: {source_path} does not exist")
        return
    
    # Get all PNG files
    image_files = list(source_path.glob("*.png"))
    
    if not image_files:
        print(f"No PNG files found in {source_path}")
        return
    
    print(f"Found {len(image_files)} files to rename")
    
    renamed_count = 0
    
    for img_file in image_files:
        if prefix:
            new_name = f"{prefix}_{uuid.uuid4()}.png"
        else:
            new_name = f"{uuid.uuid4()}.png"
        new_path = source_path / new_name
        
        try:
            img_file.rename(new_path)
            renamed_count += 1
            if renamed_count % 1000 == 0:
                print(f"Renamed {renamed_count} files...")
        except Exception as e:
            print(f"Error renaming {img_file}: {e}")
    
    print(f"\nCompleted! Renamed {renamed_count} files")
    if prefix:
        print(f"Format: {prefix}_<uuid>.png")
    else:
        print(f"Format: <uuid>.png")

def rename_to_numberletter(source_dir: str, prefix: str = "", start_num: int = 1):
    """
    Rename all images to numberletter_number format: fake_1a_1.png, fake_1a_2.png, etc.
    
    Args:
        source_dir: Directory containing images
        prefix: Prefix for filenames (default: "fake")
        start_num: Starting number (default: 1)
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Error: {source_path} does not exist")
        return
    
    # Get all PNG files
    image_files = sorted(list(source_path.glob("*.png")))
    
    if not image_files:
        print(f"No PNG files found in {source_path}")
        return
    
    print(f"Found {len(image_files)} files to rename")
    
    letters = string.ascii_lowercase
    renamed_count = 0
    current_num = start_num
    current_letter_idx = 0
    file_idx = 0
    
    for img_file in image_files:
        # Format: prefix_numberletter_number.png
        # e.g., fake_1a_1.png, fake_1a_2.png, ..., fake_1z_1.png, fake_2a_1.png
        if file_idx % 100 == 0 and file_idx > 0:
            # Move to next letter after 100 files
            current_letter_idx += 1
            if current_letter_idx >= len(letters):
                current_letter_idx = 0
                current_num += 1
        
        letter = letters[current_letter_idx]
        sub_num = (file_idx % 100) + 1
        if prefix:
            new_name = f"{prefix}_{current_num}{letter}_{sub_num}.png"
        else:
            new_name = f"{current_num}{letter}_{sub_num}.png"
        new_path = source_path / new_name
        
        # Skip if already renamed
        if img_file.name == new_name:
            file_idx += 1
            continue
        
        # Handle conflicts
        if new_path.exists() and new_path != img_file:
            # Find next available name
            while new_path.exists():
                sub_num += 1
                if sub_num > 100:
                    sub_num = 1
                    current_letter_idx += 1
                    if current_letter_idx >= len(letters):
                        current_letter_idx = 0
                        current_num += 1
                if prefix:
                    new_name = f"{prefix}_{current_num}{letters[current_letter_idx]}_{sub_num}.png"
                else:
                    new_name = f"{current_num}{letters[current_letter_idx]}_{sub_num}.png"
                new_path = source_path / new_name
        
        try:
            img_file.rename(new_path)
            renamed_count += 1
            file_idx += 1
            if renamed_count % 1000 == 0:
                print(f"Renamed {renamed_count} files...")
        except Exception as e:
            print(f"Error renaming {img_file}: {e}")
    
    print(f"\nCompleted! Renamed {renamed_count} files")
    if prefix:
        print(f"Format: {prefix}_<number><letter>_<number>.png")
    else:
        print(f"Format: <number><letter>_<number>.png")

if __name__ == "__main__":
    import sys
    
    source = r"C:\Users\prota\Repos\df3\FaceForensicsPP_frames\fake\all"
    format_type = "sequential"  # Options: sequential, uuid, numberletter
    prefix = ""  # No prefix by default
    
    if len(sys.argv) > 1:
        source = sys.argv[1]
    
    if len(sys.argv) > 2:
        format_type = sys.argv[2].lower()
    
    if len(sys.argv) > 3:
        prefix = sys.argv[3]
    
    print(f"Renaming files in: {source}")
    print(f"Format: {format_type}")
    if prefix:
        print(f"Prefix: {prefix}")
    else:
        print(f"Prefix: (none)")
    print("-" * 50)
    
    if format_type == "uuid":
        rename_to_uuid(source, prefix)
    elif format_type == "numberletter":
        rename_to_numberletter(source, prefix)
    else:  # sequential (default)
        rename_to_sequential(source, prefix)
