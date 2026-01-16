#!/usr/bin/env python3
"""
Dump ELA map to a PNG for visual inspection.

Usage:
    python scripts/dump_ela_map.py --image path/to/image.jpg --out ela_map.png

All parameters match Sherloq defaults. Adjust as needed:
    python scripts/dump_ela_map.py --image img.jpg --out ela.png --quality 75 --scale 50 --contrast 20
"""

import argparse
import base64
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tools.forensic import perform_ela  # noqa: E402


def decode_and_save(ela_map: str, out_path: Path):
    """Decode data:image/png;base64,... to a file."""
    if not ela_map or not ela_map.startswith("data:image/png;base64,"):
        raise ValueError("ELA map missing or not a base64 PNG data URL.")
    b64 = ela_map.split(",", 1)[1]
    data = base64.b64decode(b64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description="Dump ELA map to PNG (Sherloq-compatible).")
    parser.add_argument("--image", required=True, help="Path to source image.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    # Defaults match Sherloq
    parser.add_argument("--quality", type=int, default=75, help="JPEG recompression quality (1-100). Default: 75")
    parser.add_argument("--max-size", type=int, default=0, help="Max side length (0=no resize). Default: 0")
    parser.add_argument("--scale", type=int, default=50, help="Scale factor (1-100). Default: 50")
    parser.add_argument("--contrast", type=int, default=20, help="Contrast adjustment %% (0-100). Default: 20")
    parser.add_argument("--linear", action="store_true", help="Use linear mode (default: non-linear with sqrt)")
    parser.add_argument("--grayscale", action="store_true", help="Output grayscale (default: color RGB)")
    args = parser.parse_args()

    payload = {
        "path": args.image,
        "quality": args.quality,
        "max_size": args.max_size,
        "scale": args.scale,
        "contrast": args.contrast,
        "linear": args.linear,
        "grayscale": args.grayscale,
        "return_map": True,
    }
    result = json.loads(perform_ela(json.dumps(payload)))
    if result.get("status") != "completed":
        raise RuntimeError(f"ELA failed: {result}")
    ela_map = result.get("ela_map")
    if ela_map is None:
        raise RuntimeError("ELA map missing in result.")
    decode_and_save(ela_map, Path(args.out))
    reported_size = result.get("ela_map_size")
    mode = "linear" if args.linear else "non-linear (sqrt)"
    color = "grayscale" if args.grayscale else "color"
    print(f"Saved ELA map to {args.out}")
    print(f"  Size: {reported_size}, Quality: {args.quality}, Scale: {args.scale}, Contrast: {args.contrast}%")
    print(f"  Mode: {mode}, Output: {color}")


if __name__ == "__main__":
    main()
