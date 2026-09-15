#!/usr/bin/env python3
"""
=============================================================================
SignalScope - Official CLI Prediction Interface (PDF Section 4.1 & 7.1)
Evaluates a single image or directory of images for media authenticity.
=============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.vit_service import analyze_image

def main():
    parser = argparse.ArgumentParser(
        description="SignalScope: SOTA AI-Generated Media Detector & Provenance Engine"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to input image file (JPEG, PNG, WebP)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="efficientnet_b3",
        choices=["efficientnet_b3", "vit_b16", "aarnav_efficientnet"],
        help="Model core architecture to use (default: efficientnet_b3)"
    )
    parser.add_argument(
        "--jpeg-stress",
        action="store_true",
        help="Simulate JPEG compression (q=50) for robustness testing"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON results for automated benchmark scoring"
    )

    args = parser.parse_args()
    img_path = Path(args.image)

    if not img_path.exists():
        print(f"[ERROR] Image path does not exist: {img_path}", file=sys.stderr)
        sys.exit(1)

    import contextlib

    # Run inference and multi-signal forensic pipeline
    if args.json:
        with contextlib.redirect_stdout(sys.stderr):
            result = analyze_image(
                str(img_path),
                simulate_jpeg=args.jpeg_stress,
                model_type=args.model
            )
        # Exclude large base64 heatmap from raw console JSON if desired
        out = {k: v for k, v in result.items() if k != "heatmap_base64"}
        print(json.dumps(out, indent=2))
        return

    result = analyze_image(
        str(img_path),
        simulate_jpeg=args.jpeg_stress,
        model_type=args.model
    )

    # Clean, human-readable terminal output for evaluators
    print("\n" + "=" * 60)
    print("        SIGNALSCOPE FORENSIC VERDICT REPORT")
    print("=" * 60)
    print(f"Target Image:      {img_path.name}")
    print(f"Active Model:      {result['model_name']}")
    print(f"Verdict Title:     {result['verdict_title']}")
    print(f"Verdict Status:    {result['verdict_status'].upper()}")
    print("-" * 60)
    print(f"AI Probability:    {result['raw_fake_pct']}%")
    print(f"Real Probability:  {result['raw_real_pct']}%")
    print(f"Operating Cutoff:  {result['operating_threshold']}% Threshold")
    print("-" * 60)
    print("PROVENANCE & HARDWARE (EXIF & SENSOR):")
    print(f"  Camera Hardware: {result['exif'].get('device_model', 'No EXIF Found')}")
    print(f"  Exposure:        {result['exif'].get('exposure_settings', 'No EXIF Found')}")
    print(f"  Sensor/Lens:     {result['exif'].get('lens_sensor', 'No EXIF Found')}")
    print("-" * 60)
    print("FREQUENCY DOMAIN FFT (SECTION 6):")
    print(f"  Power Decay:     {result['fft_analysis'].get('freq_decay_ratio')} ({'Natural Optics' if result['fft_analysis'].get('is_natural_optics') else 'Anomalous'})")
    print(f"  Grid Artifact:   {'Detected (Periodic lattice)' if result['fft_analysis'].get('grid_artifact_detected') else 'None Detected'}")
    if result.get("calibration_applied"):
        print("-" * 60)
        print("MULTI-SIGNAL CALIBRATION APPLIED:")
        print(f"  Reason: {result.get('calibration_reason')}")
    if result.get("cues"):
        print("-" * 60)
        print("GROUNDED FORENSIC REGION CUES:")
        for cue in result.get("cues", []):
            print(f"  * {cue}")
    print("-" * 60)
    print("SUMMARY:")
    print(f"  {result['summary']}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
