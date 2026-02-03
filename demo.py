#!/usr/bin/env python3
"""
MuSc Industrial Anomaly Detection - CLI Demo

A command-line interface for running zero-shot anomaly detection
without the GUI. Useful for testing, scripting, and headless environments.

Usage:
    python demo.py --input image.png
    python demo.py --input image.png --reference ref_dir/ --output results/
    python demo.py --input images/ --backbone dinov2_vitb14 --threshold 0.9
"""

import argparse
import os
import sys
import glob
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    # Default configuration
    return {
        "datasets": {
            "img_resize": 224,
            "dataset_name": "mvtec_ad",
            "class_name": "ALL",
            "divide_num": 1,
            "data_path": "./data/",
        },
        "device": 0,
        "models": {
            "backbone_name": "dinov2_vitb14",
            "batch_size": 1,
            "feature_layers": [11],
            "pretrained": "openai",
            "r_list": [1],
        },
        "testing": {
            "output_dir": "output_gui",
            "vis": False,
            "vis_type": "single_norm",
            "save_excel": False,
        },
        "thresholds": {
            "image_threshold": 9.0,
            "overlay_threshold": 3.5,
        },
    }


def load_image(image_path: str, target_size: int) -> tuple[np.ndarray, torch.Tensor]:
    """
    Load and preprocess an image for inference.

    Returns:
        Tuple of (original_bgr_image, preprocessed_tensor)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert BGR to RGB and resize
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_size, target_size))

    # Normalize and convert to tensor
    img_norm = img_resized.astype(np.float32) / 255.0
    tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0)

    return img, tensor


def save_heatmap(
    original_img: np.ndarray,
    anomaly_map: np.ndarray,
    output_path: str,
    overlay_alpha: float = 0.5,
) -> None:
    """Save anomaly heatmap overlay on original image."""
    # Normalize anomaly map to 0-255
    if anomaly_map.max() > anomaly_map.min():
        normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
    else:
        normalized = np.zeros_like(anomaly_map)

    heatmap = (normalized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap_colored, (original_img.shape[1], original_img.shape[0]))

    # Create overlay
    overlay = cv2.addWeighted(original_img, 1 - overlay_alpha, heatmap_resized, overlay_alpha, 0)

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, overlay)


def main():
    parser = argparse.ArgumentParser(
        description="MuSc Zero-Shot Anomaly Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect anomalies in a single image
  python demo.py --input sample.png

  # Process a directory of images
  python demo.py --input images/ --output results/

  # Use a specific backbone model
  python demo.py --input sample.png --backbone dinov2_vitl14

  # Adjust detection threshold (1.0-10.0, higher = less sensitive)
  python demo.py --input sample.png --threshold 8.0
        """,
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input image path or directory containing images",
    )
    parser.add_argument(
        "--output", "-o",
        default="output_demo",
        help="Output directory for heatmaps (default: output_demo)",
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--backbone", "-b",
        choices=[
            "dino_deitsmall16", "dino_vitbase8", "dino_vitbase16",
            "dinov2_vitb14", "dinov2_vitl14",
            "ViT-B-16", "ViT-B-32", "ViT-L-14",
            "vit_small_patch16_224.dino", "vit_tiny_patch16_224.augreg_in21k",
        ],
        help="Backbone model (overrides config)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Anomaly threshold 1.0-10.0 (overrides config)",
    )
    parser.add_argument(
        "--image-size", "-s",
        type=int,
        choices=[224, 256, 384, 512],
        help="Image resize dimension (overrides config)",
    )
    parser.add_argument(
        "--device", "-d",
        type=int,
        default=None,
        help="GPU device index, or -1 for CPU (overrides config)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save heatmap images, only print scores",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress information",
    )

    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Apply command-line overrides
    if args.backbone:
        cfg["models"]["backbone_name"] = args.backbone
    if args.threshold:
        cfg["thresholds"]["image_threshold"] = args.threshold
    if args.image_size:
        cfg["datasets"]["img_resize"] = args.image_size
    if args.device is not None:
        cfg["device"] = args.device if args.device >= 0 else "cpu"

    # Determine device
    if cfg["device"] == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cfg['device']}")

    print(f"Device: {device}")
    print(f"Backbone: {cfg['models']['backbone_name']}")
    print(f"Image size: {cfg['datasets']['img_resize']}")
    print(f"Threshold: {cfg['thresholds']['image_threshold']}")
    print()

    # Import MuSc model
    try:
        from models.musc import MuSc
    except ImportError as e:
        print(f"Error: Could not import MuSc model: {e}")
        print("Make sure you're running from the MuSc directory.")
        sys.exit(1)

    # Initialize model
    print("Loading model...")
    start_load = time.time()
    model = MuSc(cfg)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")
    print()

    # Collect input images
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [str(input_path)]
    elif input_path.is_dir():
        image_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]:
            image_paths.extend(glob.glob(str(input_path / ext)))
            image_paths.extend(glob.glob(str(input_path / ext.upper())))
        image_paths = sorted(set(image_paths))
    else:
        print(f"Error: Input path not found: {args.input}")
        sys.exit(1)

    if not image_paths:
        print(f"Error: No images found in {args.input}")
        sys.exit(1)

    print(f"Processing {len(image_paths)} image(s)...")
    print("-" * 60)

    # Process images
    target_size = cfg["datasets"]["img_resize"]
    threshold = cfg["thresholds"]["image_threshold"] / 10.0  # Internal 0-1 scale
    results = []

    for i, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)

        if args.verbose:
            print(f"[{i+1}/{len(image_paths)}] Processing: {img_name}")

        try:
            # Load and preprocess image
            original_img, tensor = load_image(img_path, target_size)
            tensor = tensor.to(device)

            # Run inference
            start_infer = time.time()
            with torch.no_grad():
                anomaly_maps = model.infer_on_images([tensor])
            infer_time = time.time() - start_infer

            # Get max score
            anomaly_map = anomaly_maps[0] if anomaly_maps.ndim > 2 else anomaly_maps
            max_score = float(anomaly_map.max())
            is_anomaly = max_score > threshold

            status = "ANOMALY" if is_anomaly else "OK"
            print(f"{img_name}: {status} (score: {max_score:.4f}, time: {infer_time:.3f}s)")

            results.append({
                "path": img_path,
                "name": img_name,
                "score": max_score,
                "is_anomaly": is_anomaly,
                "inference_time": infer_time,
            })

            # Save heatmap
            if not args.no_save:
                output_path = os.path.join(args.output, f"{Path(img_name).stem}_heatmap.png")
                save_heatmap(original_img, anomaly_map, output_path)
                if args.verbose:
                    print(f"  Saved: {output_path}")

        except Exception as e:
            print(f"{img_name}: ERROR - {e}")
            results.append({
                "path": img_path,
                "name": img_name,
                "score": None,
                "is_anomaly": None,
                "error": str(e),
            })

    # Print summary
    print("-" * 60)
    successful = [r for r in results if r.get("score") is not None]
    anomalies = [r for r in successful if r["is_anomaly"]]

    print(f"\nSummary:")
    print(f"  Total images: {len(results)}")
    print(f"  Processed: {len(successful)}")
    print(f"  Anomalies detected: {len(anomalies)}")
    print(f"  Normal: {len(successful) - len(anomalies)}")

    if successful:
        avg_time = sum(r["inference_time"] for r in successful) / len(successful)
        print(f"  Average inference time: {avg_time:.3f}s")

    if not args.no_save:
        print(f"\nHeatmaps saved to: {args.output}/")

    # Return exit code based on anomaly detection
    sys.exit(1 if anomalies else 0)


if __name__ == "__main__":
    main()
