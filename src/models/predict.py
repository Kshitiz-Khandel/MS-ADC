import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
from PIL import Image

import torch
from src.models.die_vfm import DieVFMClassifier, DIE_DEFECT_CLASSES


def predict_single_image(
    image_path: str,
    weights_path: str = "models/v1.0.0/die_vfm_head.pt"
) -> Dict[str, Any]:
    """
    Runs production inference on a new optical defect micrograph or wafer patch.
    Returns predicted defect class, probability distribution, and cleanroom disposition.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Initialize classifier and load checkpoint
    classifier = DieVFMClassifier(num_classes=6, weights_path=weights_path)
    image = Image.open(image_path).convert("RGB")
    
    # Run classification
    raw_res = classifier.classify_patch(image)

    predicted_class = raw_res.get("predicted_class", DIE_DEFECT_CLASSES[0])
    confidence = raw_res.get("confidence", 0.95)
    all_probs = raw_res.get("all_probabilities", {})

    # Cleanroom semiconductor disposition rule:
    # If confidence < 0.85, flag for Secondary Metrology / Review Station
    if confidence >= 0.85:
        disposition = "AUTO_DEFECT_ROUTED"
    else:
        disposition = "SECONDARY_REVIEW_REQUIRED"

    return {
        "image_path": str(image_path),
        "weights_loaded": str(weights_path),
        "predicted_class": predicted_class,
        "defect_type": predicted_class,
        "confidence": confidence,
        "disposition": disposition,
        "probabilities": all_probs,
        "all_probabilities": all_probs
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MS-ADC Single Image Defect Inference CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input optical micrograph (JPG/PNG)")
    parser.add_argument("--weights", type=str, default="models/v1.0.0/die_vfm_head.pt", help="Path to trained PyTorch weights")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")

    args = parser.parse_args()
    try:
        res = predict_single_image(args.image, args.weights)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("\n" + "=" * 60)
            print("🔬 MS-ADC DIE-LEVEL DEFECT INFERENCE REPORT")
            print("=" * 60)
            print(f"📁 Image Path        : {res['image_path']}")
            print(f"📦 Model Weights     : {res['weights_loaded']}")
            print(f"🏷️ Predicted Defect  : {res['predicted_class'].upper()}")
            print(f"🎯 Confidence Score  : {res['confidence']*100:.2f}%")
            print(f"🚦 Fab Disposition   : {res['disposition']}")
            print("-" * 60)
            print("📊 Class Probabilities Breakdown:")
            for cls, prob in res["probabilities"].items():
                bar = "█" * int(prob * 30)
                print(f"  • {cls:<16}: {prob*100:>6.2f}% | {bar}")
            print("=" * 60 + "\n")
    except Exception as e:
        print(f"❌ Error during inference: {e}", file=sys.stderr)
        sys.exit(1)
