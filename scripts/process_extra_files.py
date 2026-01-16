#!/usr/bin/env python3
"""
Script to process files from extra_real and extra_fake folders.
Renames files sequentially, copies them to data folder, and updates my_eval.jsonl.
"""

import os
import shutil
import json
from pathlib import Path

def get_next_sample_number(eval_file):
    """Get the next sample number from existing my_eval.jsonl"""
    if not os.path.exists(eval_file):
        return 1
    
    max_num = 0
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                sample_id = data.get('id', '')
                if sample_id.startswith('sample-'):
                    try:
                        num = int(sample_id.split('-')[1])
                        max_num = max(max_num, num)
                    except (ValueError, IndexError):
                        pass
    return max_num + 1

def get_file_extension(filename):
    """Get file extension, handling case-insensitive extensions"""
    return Path(filename).suffix.lower()

def process_extra_files(data_dir):
    """Process files from extra_real and extra_fake folders"""
    data_path = Path(data_dir)
    extra_real_dir = data_path / 'extra_real'
    extra_fake_dir = data_path / 'extra_fake'
    eval_file = data_path / 'my_eval.jsonl'
    
    # Check if folders exist
    if not extra_real_dir.exists():
        print(f"Warning: {extra_real_dir} does not exist")
        return
    
    if not extra_fake_dir.exists():
        print(f"Warning: {extra_fake_dir} does not exist")
        return
    
    # Get starting sample number
    next_sample = get_next_sample_number(eval_file)
    print(f"Starting from sample-{next_sample}")
    
    # Collect all files to process
    files_to_process = []
    
    # Process extra_real files
    for file_path in sorted(extra_real_dir.iterdir()):
        if file_path.is_file():
            files_to_process.append((file_path, 'real'))
    
    # Process extra_fake files
    for file_path in sorted(extra_fake_dir.iterdir()):
        if file_path.is_file():
            files_to_process.append((file_path, 'fake'))
    
    print(f"Found {len(files_to_process)} files to process")
    
    # Process files and update eval file
    new_entries = []
    sample_num = next_sample
    
    for source_file, ground_truth in files_to_process:
        # Get original extension
        ext = get_file_extension(source_file.name)
        if not ext:
            ext = '.jpg'  # Default extension if none found
        
        # Generate new filename
        new_filename = f"example{sample_num}{ext}"
        dest_file = data_path / new_filename
        
        # Copy file
        print(f"Copying {source_file.name} -> {new_filename} ({ground_truth})")
        shutil.copy2(source_file, dest_file)
        
        # Create eval entry
        entry = {
            "id": f"sample-{sample_num}",
            "image_path": new_filename,
            "ground_truth": ground_truth
        }
        new_entries.append(entry)
        sample_num += 1
    
    # Append to eval file
    if new_entries:
        with open(eval_file, 'a', encoding='utf-8') as f:
            for entry in new_entries:
                f.write(json.dumps(entry) + '\n')
        print(f"\nAdded {len(new_entries)} entries to {eval_file}")
        print(f"Processed samples: sample-{next_sample} to sample-{sample_num - 1}")

if __name__ == '__main__':
    # Get data directory (parent of scripts directory)
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        exit(1)
    
    process_extra_files(data_dir)
    print("\nDone!")
