"""
Extract 2 frames per video from FaceForensicsPP dataset.

Extracts frames without re-encoding or modifying video properties.
Frames are saved as PNG files to preserve quality.

Optimized with multiprocessing for fast parallel processing.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import cv2
from tqdm import tqdm


def extract_frames_optimized(
    video_path: str,
    output_dir: str,
    frame_positions: Tuple[float, float] = (0.25, 0.75)
) -> bool:
    """
    Extract frames from a video efficiently (single VideoCapture open).
    
    Optimized version that opens VideoCapture only once.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        frame_positions: Tuple of relative positions (0.0-1.0) to extract frames
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return False
        
        # Calculate frame numbers
        frame_numbers = [
            int(pos * (total_frames - 1)) for pos in frame_positions
        ]
        # Ensure we don't exceed bounds
        frame_numbers = [min(f, total_frames - 1) for f in frame_numbers]
        
        # Extract frames (reuse same VideoCapture)
        video_name = Path(video_path).stem
        frames_extracted = []
        
        for i, frame_num in enumerate(frame_numbers):
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
            
            # Save frame as PNG (lossless)
            output_filename = f"{video_name}_frame{i+1:02d}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save using cv2.imwrite (preserves quality)
            success = cv2.imwrite(output_path, frame)
            
            if success:
                frames_extracted.append(output_path)
        
        cap.release()
        return len(frames_extracted) > 0
        
    except Exception as e:
        return False


def process_single_video(args_tuple):
    """
    Process a single video (worker function for multiprocessing).
    
    Args:
        args_tuple: Tuple of (video_path, output_dir, frame_positions, preserve_structure, input_dir)
        
    Returns:
        Tuple of (success: bool, video_path: str)
    """
    video_path, output_dir, frame_positions, preserve_structure, input_dir = args_tuple
    
    try:
        if preserve_structure:
            # Save frames in same directory as video
            video_dir = os.path.dirname(video_path)
            target_output_dir = video_dir
        else:
            # Preserve relative path structure (fake/real folders, etc.)
            video_dir = os.path.dirname(video_path)
            # Get relative path from input_dir to video's directory
            rel_path = os.path.relpath(video_dir, input_dir)
            # Create output path maintaining structure
            target_output_dir = os.path.join(output_dir, rel_path)
        
        # Ensure output directory exists
        os.makedirs(target_output_dir, exist_ok=True)
        
        # Extract frames
        success = extract_frames_optimized(video_path, target_output_dir, frame_positions)
        return (success, video_path)
    except Exception:
        return (False, video_path)


def find_all_videos(root_dir: str) -> list:
    """
    Recursively find all .mp4 files in directory tree.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of video file paths
    """
    video_files = []
    root_path = Path(root_dir)
    
    for video_path in root_path.rglob("*.mp4"):
        video_files.append(str(video_path))
    
    return sorted(video_files)


def main():
    parser = argparse.ArgumentParser(
        description="Extract 2 frames per video from FaceForensicsPP dataset"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="FaceForensicsPP",
        help="Root directory containing videos (default: FaceForensicsPP)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="FaceForensicsPP_frames",
        help="Output directory for extracted frames (default: FaceForensicsPP_frames)"
    )
    parser.add_argument(
        "--frame_positions",
        type=float,
        nargs=2,
        default=[0.25, 0.75],
        help="Relative positions (0.0-1.0) to extract frames (default: 0.25 0.75)"
    )
    parser.add_argument(
        "--preserve_structure",
        action="store_true",
        help="Save frames next to videos (default: preserve fake/real structure in output_dir)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)"
    )
    
    args = parser.parse_args()
    
    # Set default workers to CPU count if not specified
    if args.workers is None:
        import multiprocessing
        args.workers = multiprocessing.cpu_count()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Find all videos
    print(f"Scanning for videos in {args.input_dir}...")
    video_files = find_all_videos(args.input_dir)
    
    if not video_files:
        print(f"No .mp4 files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"Found {len(video_files)} video files")
    
    # Create output directory if not preserving structure
    if not args.preserve_structure:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare arguments for workers
    worker_args = [
        (video_path, args.output_dir, tuple(args.frame_positions), args.preserve_structure, args.input_dir)
        for video_path in video_files
    ]
    
    # Process videos with multiprocessing
    successful = 0
    failed = 0
    
    print(f"\nExtracting frames at positions {args.frame_positions}...")
    print(f"Using {args.workers} worker processes")
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_video = {
            executor.submit(process_single_video, args_tuple): args_tuple[0]
            for args_tuple in worker_args
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(video_files), desc="Processing videos") as pbar:
            for future in as_completed(future_to_video):
                success, video_path = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                pbar.update(1)
    
    print(f"\nCompleted:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(video_files)}")
    
    if not args.preserve_structure:
        print(f"\nFrames saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
