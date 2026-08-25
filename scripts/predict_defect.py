#!/usr/bin/env python3
"""
MS-ADC Optical Defect Inference Tool
------------------------------------
Runs optical micrograph defect classification on any given image file.
Outputs predictions with confidence scores, probability distributions, and metrology metadata.

Usage:
    python scripts/predict_defect.py --image path/to/die_micrograph.jpg
    python scripts/predict_defect.py --image path/to/die_micrograph.jpg --model models/checkpoint_best.pt
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from PIL import Image

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import torchvision.transforms as T
from src.models.die_vfm import DieVFMClassifier, DIE_DEFECT_CLASSES

def predict_single_image(
    image_path: str,
    model_path: str = "models/checkpoint_best.pt",
    crop_roi: bool = True
) -> dict:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Load image
    raw_img = Image.open(image_path).convert("RGB")
    original_size = raw_img.size

    # Check for XML annotation if available to crop ROI
    xml_path = None
    if crop_roi:
        # Search sibling Annotations folder
        possible_xml = image_path.parents[2] / "Annotations" / image_path.parent.name / f"{image_path.stem}.xml"
        if possible_xml.exists():
            xml_path = possible_xml
            import xml.etree.ElementTree as ET
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                obj = root.find("object")
                if obj is not None:
                    bndbox = obj.find("bndbox")
                    xmin = max(0, int(bndbox.find("xmin").text) - 40)
                    ymin = max(0, int(bndbox.find("ymin").text) - 40)
                    xmax = min(raw_img.width, int(bndbox.find("xmax").text) + 40)
                    ymax = min(raw_img.height, int(bndbox.find("ymax").text) + 40)
                    if xmax > xmin and ymax > ymin:
                        raw_img = raw_img.crop((xmin, ymin, xmax, ymax))
            except Exception:
                pass

    # Initialize model
    classifier = DieVFMClassifier(num_classes=len(DIE_DEFECT_CLASSES))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            classifier.torch_model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            classifier.torch_model.load_state_dict(ckpt["state_dict"])
        classifier.torch_model.eval()

    # Preprocess image
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(raw_img).unsqueeze(0).to(device)

    # Measure latency
    start_time = time.perf_counter()
    with torch.no_grad():
        logits = classifier.torch_model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    pred_idx = probs.index(max(probs))
    predicted_class = DIE_DEFECT_CLASSES[pred_idx]
    confidence = round(probs[pred_idx] * 100, 2)

    prob_distribution = {
        cls: round(p * 100, 2) for cls, p in zip(DIE_DEFECT_CLASSES, probs)
    }

    result = {
        "status": "SUCCESS",
        "image_file": str(image_path),
        "original_resolution": f"{original_size[0]}x{original_size[1]}",
        "roi_cropped": xml_path is not None,
        "predicted_defect": predicted_class,
        "confidence_percentage": confidence,
        "inference_latency_ms": latency_ms,
        "class_probability_distribution": prob_distribution,
        "checkpoint_used": model_path
    }

    return result

def print_inference_report(result: dict):
    print("\n" + "=" * 65)
    print("🔬 MS-ADC METROLOGY DEFECT CLASSIFICATION INFERENCE")
    print("=" * 65)
    print(f"📁 Image: {result['image_file']} ({result['original_resolution']})")
    print(f"🎯 Predicted Defect:  >> {result['predicted_defect'].upper()} <<")
    print(f"⚡ Confidence:        {result['confidence_percentage']}%")
    print(f"⏱️  Inference Latency: {result['inference_latency_ms']} ms")
    print(f"🏷️  ROI Cropped:       {result['roi_cropped']}")
    print("-" * 65)
    print("📊 Class Probabilities Breakdown:")
    for cls_name, prob in sorted(result["class_probability_distribution"].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob / 4)
        print(f"   • {cls_name:<16}: {prob:>6.2f}%  {bar}")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a single optical die micrograph")
    parser.add_argument("--image", type=str, required=True, help="Path to input optical image")
    parser.add_argument("--model", type=str, default="models/checkpoint_best.pt", help="Path to checkpoint file")
    parser.add_argument("--json", action="store_true", help="Print output as raw JSON")
    args = parser.parse_args()

    res = predict_single_image(args.image, model_path=args.model)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print_inference_report(res)
