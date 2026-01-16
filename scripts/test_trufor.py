#!/usr/bin/env python3
"""
Test TruFor forgery detection and localization.

TruFor combines RGB features with Noiseprint++ to detect and localize
image forgeries using a transformer-based fusion architecture.

Requirements:
    - PyTorch (torch, torchvision)
    - timm (for Segformer backbone)
    - yacs (for configuration)
    - TruFor weights at weights/trufor/trufor.pth.tar

Usage:
    python scripts/test_trufor.py --image path/to/image.jpg
    
    # Save localization map to PNG
    python scripts/test_trufor.py --image img.jpg --out localization_map.png
    
    # Use CPU instead of GPU
    python scripts/test_trufor.py --image img.jpg --gpu -1
    
    # Use specific GPU
    python scripts/test_trufor.py --image img.jpg --gpu 1
    
    # Skip localization map generation (faster)
    python scripts/test_trufor.py --image img.jpg --no-map
"""

import argparse
import base64
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tools.forensic import perform_trufor  # noqa: E402


def decode_and_save(localization_map: str, out_path: Path):
    """Decode data:image/png;base64,... to a file."""
    if not localization_map or not localization_map.startswith("data:image/png;base64,"):
        raise ValueError("Localization map missing or not a base64 PNG data URL.")
    b64 = localization_map.split(",", 1)[1]
    data = base64.b64decode(b64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    print(f"✓ Saved localization map to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test TruFor AI-driven forgery detection and localization."
    )
    parser.add_argument("--image", required=True, help="Path to source image.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path for localization map (optional).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device (-1 for CPU, 0+ for GPU). Default: 0",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Don't generate localization map (faster).",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    print("=" * 70)
    print("TruFor Forgery Detection Test")
    print("=" * 70)
    print(f"Image: {image_path}")
    print(f"Device: {'CPU' if args.gpu < 0 else f'GPU {args.gpu}'}")
    print(f"Generate map: {not args.no_map}")
    print("-" * 70)

    # Prepare payload
    payload = {
        "path": str(image_path),
        "gpu": args.gpu,
        "return_map": not args.no_map,
    }

    try:
        print("\nRunning TruFor analysis...")
        print("(This may take a moment, especially on first run as the model loads...)")
        result_json = perform_trufor(json.dumps(payload))
        result = json.loads(result_json)

        if result.get("status") != "completed":
            error_msg = result.get("error", "Unknown error")
            print(f"\n❌ Error: {error_msg}")
            if "note" in result:
                print(f"\nNote: {result['note']}")
            sys.exit(1)

        # Extract results
        manipulation_prob = result.get("manipulation_probability", 0.0)
        detection_score = result.get("detection_score", 0.0)
        localization_map = result.get("localization_map")
        map_size = result.get("localization_map_size")

        # Display results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\nManipulation Probability: {manipulation_prob:.4f} ({manipulation_prob*100:.2f}%)")
        print(f"Detection Score:          {detection_score:.4f} ({detection_score*100:.2f}%)")

        # Interpretation
        print("\n" + "-" * 70)
        print("Interpretation:")
        if manipulation_prob < 0.3:
            interpretation = "Low probability of manipulation (likely authentic)"
        elif manipulation_prob < 0.6:
            interpretation = "Moderate probability of manipulation (uncertain)"
        else:
            interpretation = "High probability of manipulation (likely forged)"
        print(f"  {interpretation}")

        if localization_map and map_size:
            print(f"\nLocalization Map Size: {map_size[0]}x{map_size[1]} pixels")
            if args.out:
                decode_and_save(localization_map, Path(args.out))
            else:
                print("\n💡 Tip: Use --out <path> to save the localization map as PNG")

        print("\n" + "=" * 70)
        print("Analysis complete!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
