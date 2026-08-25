#!/usr/bin/env python3
"""
MS-ADC Vertex AI Workbench Training Pipeline
---------------------------------------------
End-to-end Deep Vision Foundation Model fine-tuning pipeline on the Kaggle PCB defect dataset.
Fine-tunes a deep convolutional/vision backbone on localized optical defect patches (ROI)
with real-time epoch metrics, validation monitoring, best checkpointing, and evaluation reporting.
"""

import os
import sys
import json
import time
import math
import argparse
import subprocess
from typing import Dict, Any, List, Tuple
from pathlib import Path
from PIL import Image

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models

from src.models.die_vfm import DieVFMClassifier, DIE_DEFECT_CLASSES
from src.models.export_tensorrt import TensorRTExporter
from src.ingestion.dataset_loader import PCBDefectDatasetLoader
from src.utils.metrics import SemiconductorYieldCalculator

def run_workbench_training(
    k_shot: int = 10,
    epochs: int = 25,
    learning_rate: float = 0.001,
    val_ratio: float = 0.2,
    batch_size: int = 16,
    output_dir: str = "models",
    gcs_bucket: str = "semicon-metrology-models",
    data_dir_path: str = "data/pcb_dataset"
) -> Dict[str, Any]:
    print("=" * 82)
    print("🚀 MS-ADC: Deep Vision Metrology Foundation Model Fine-Tuning Pipeline")
    print(f"Defect Classes ({len(DIE_DEFECT_CLASSES)}): {DIE_DEFECT_CLASSES}")
    print(f"Few-Shot Support (K-Shot): {k_shot} samples per class | Total Epochs: {epochs}")
    print(f"Learning Rate: {learning_rate} | Validation Ratio: {val_ratio * 100:.0f}% | Batch Size: {batch_size}")
    print("=" * 82)

    os.makedirs(output_dir, exist_ok=True)
    data_dir = ROOT_DIR / data_dir_path
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/5] Initializing Deep Vision Model on Device: {device}...")

    # Step 1: Ingest optical defect dataset & create 3-way split
    print(f"\n[2/5] Ingesting optical dataset and creating 3-way Stratified Splits...")
    loader = PCBDefectDatasetLoader(data_dir)
    train_paths, val_paths, test_paths = loader.get_stratified_split(k_shot_train=k_shot, val_ratio=val_ratio)

    total_train = sum(len(p) for p in train_paths.values())
    total_val = sum(len(p) for p in val_paths.values())
    total_test = sum(len(p) for p in test_paths.values())

    if total_train == 0:
        print("⚠️ No images found in data directory. Please ensure pcb-defects.zip is extracted to data/pcb_dataset/")
        return {"status": "FAILED", "error": "Dataset missing"}

    print(f"      • Training Set   (K={k_shot}-shot): {total_train} images")
    print(f"      • Validation Set ({val_ratio*100:.0f}% split): {total_val} images")
    print(f"      • Test Set       (Held-out):  {total_test} images")

    print("\n      Pre-loading & cropping localized defect patches (ROI) into memory...")
    def load_split_images(paths_dict):
        items = []
        for class_idx, cls_name in enumerate(DIE_DEFECT_CLASSES):
            for p in paths_dict.get(cls_name, []):
                img = loader.load_and_preprocess_image(p)
                items.append((img, class_idx))
        return items

    train_imgs = load_split_images(train_paths)
    val_imgs = load_split_images(val_paths)
    test_imgs = load_split_images(test_paths)

    # Data Augmentation Transforms for Cleanroom Metrology
    transform_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_eval = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Pre-tensorize Validation & Test sets
    X_val = torch.stack([transform_eval(img) for img, _ in val_imgs]).to(device)
    y_val = torch.tensor([lbl for _, lbl in val_imgs], dtype=torch.long).to(device)

    X_test = torch.stack([transform_eval(img) for img, _ in test_imgs]).to(device)
    y_test = torch.tensor([lbl for _, lbl in test_imgs], dtype=torch.long).to(device)

    # Step 2: Initialize ResNet18 Backbone & Head
    classifier = DieVFMClassifier(num_classes=len(DIE_DEFECT_CLASSES))
    model = classifier.torch_model

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-3
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Step 3: Real Training Loop with Detailed Epoch Metrics
    print(f"\n[3/5] Starting Neural Network Training ({epochs} Epochs)...")
    print("-" * 88)
    print(f"{'Epoch':<10} | {'Train Loss':<12} | {'Train Acc':<11} | {'Val Loss':<11} | {'Val Acc':<10} | {'LR':<10} | {'Status'}")
    print("-" * 88)

    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    epoch_history = []

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_checkpoint_path = os.path.join(output_dir, "checkpoint_best.pt")

    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        
        # 6x data augmentation on the few-shot support set
        aug_tensors, aug_labels = [], []
        for img, lbl in train_imgs:
            for _ in range(6):
                aug_tensors.append(transform_train(img))
                aug_labels.append(lbl)
                
        X_train_batch = torch.stack(aug_tensors).to(device)
        y_train_batch = torch.tensor(aug_labels, dtype=torch.long).to(device)

        # Mini-batch shuffle
        perm = torch.randperm(X_train_batch.size(0))
        epoch_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, X_train_batch.size(0), batch_size):
            indices = perm[i:i + batch_size]
            bx, by = X_train_batch[indices], y_train_batch[indices]

            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * bx.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += bx.size(0)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        train_loss = epoch_loss / total
        train_acc = (correct / total) * 100.0

        # Evaluate on real validation set
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val).item()
            val_preds = val_out.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item() * 100.0

        train_loss_history.append(round(train_loss, 4))
        val_loss_history.append(round(val_loss, 4))
        val_acc_history.append(round(val_acc, 2))
        epoch_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc / 100.0, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc / 100.0, 4)
        })

        # Check for best model improvement
        status_msg = ""
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "val_accuracy": val_acc,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, best_checkpoint_path)
            status_msg = "⭐ Best Model Saved"

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | {train_loss:<12.4f} | {train_acc:<10.1f}% | {val_loss:<11.4f} | {val_acc:<9.1f}% | {current_lr:<10.6f} | {status_msg}")

    total_train_elapsed = round(time.time() - start_train_time, 2)
    print("-" * 88)
    print(f"✅ Training completed in {total_train_elapsed}s | Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")

    # Step 4: Final Evaluation on Held-Out Test Set
    print(f"\n[4/5] Evaluating best checkpoint on held-out Test Set ({total_test} samples)...")
    if os.path.exists(best_checkpoint_path):
        ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    with torch.no_grad():
        test_out = model(X_test)
        test_preds = test_out.argmax(dim=1).cpu().tolist()
        y_test_list = y_test.cpu().tolist()

    eval_metrics = SemiconductorYieldCalculator.calculate_classification_metrics(
        y_true=y_test_list,
        y_pred=test_preds,
        class_names=DIE_DEFECT_CLASSES
    )

    print("\n" + "-" * 65)
    print("📊 CLASSIFICATION EVALUATION METRICS REPORT")
    print("-" * 65)
    print(SemiconductorYieldCalculator.format_classification_report(eval_metrics))
    print("-" * 65)

    # Step 5: ONNX & TensorRT Compilation
    print("\n[5/5] Compiling ONNX computation graph & TensorRT FP16 engine...")
    exporter = TensorRTExporter(target_precision="FP16", max_batch_size=32)
    onnx_path = os.path.join(output_dir, "die_vfm.onnx")
    engine_path = os.path.join(output_dir, "die_vfm_fp16.engine")

    onnx_res = exporter.export_onnx(onnx_path)
    trt_res = exporter.build_tensorrt_engine(onnx_path, engine_path)

    print(f"   ONNX Graph: {onnx_res['status']}")
    print(f"   TensorRT FP16 Latency: {trt_res['benchmarks']['tensorrt_fp16_latency_ms']} ms")
    print(f"   TensorRT Speedup Factor: {trt_res['benchmarks']['speedup_factor']}x")

    # Step 6: Generate Performance Visualizations & Charts
    print("\n📈 Generating Evaluation Plots & Performance Charts...")
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    curve_plot_path = os.path.join(output_dir, "loss_accuracy_curve.png")
    prf1_plot_path = os.path.join(output_dir, "precision_recall_f1.png")

    SemiconductorYieldCalculator.save_confusion_matrix_plot(
        cm=eval_metrics["confusion_matrix"],
        class_names=DIE_DEFECT_CLASSES,
        output_path=cm_plot_path
    )
    SemiconductorYieldCalculator.save_loss_accuracy_curves(
        history=epoch_history,
        output_path=curve_plot_path
    )
    SemiconductorYieldCalculator.save_precision_recall_f1_chart(
        class_metrics=eval_metrics["classes"],
        output_path=prf1_plot_path
    )
    print("   • Saved Confusion Matrix Heatmap: models/confusion_matrix.png")
    print("   • Saved Loss/Accuracy Curves:    models/loss_accuracy_curve.png")
    print("   • Saved Precision/Recall/F1 Bar: models/precision_recall_f1.png")

    # Save metrics JSON & instructions
    metrics_file = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump({
            "epoch_history": epoch_history,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "val_acc_history": val_acc_history,
            "best_val_accuracy": best_val_acc,
            "best_epoch": best_epoch,
            "test_metrics": eval_metrics,
            "tensorrt_benchmarks": trt_res["benchmarks"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }, f, indent=2)

    print(f"\n📦 Artifacts saved to local '{output_dir}/':")
    print(f"   • Best Checkpoint: {best_checkpoint_path}")
    print(f"   • Training Metrics: {metrics_file}")
    print(f"   • ONNX Graph: {onnx_path}")
    print(f"   • TensorRT Engine: {engine_path}")
    print(f"   • Plots: confusion_matrix.png, loss_accuracy_curve.png, precision_recall_f1.png")
    print(f"\n☁️ GCS Cloud Staging Command:")
    print(f"   `gsutil -m cp -r {output_dir}/* gs://{gcs_bucket}/models/`")

    return {
        "status": "SUCCESS",
        "best_val_accuracy": best_val_acc,
        "test_metrics": eval_metrics,
        "tensorrt_benchmarks": trt_res["benchmarks"],
        "training_time_sec": total_train_elapsed
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MS-ADC Vision Foundation Model on Vertex AI Workbench")
    parser.add_argument("--k-shot", type=int, default=10, help="Number of labeled training examples per defect class")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for AdamW optimizer")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio of remaining dataset")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size for training")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to store exported checkpoints and models")
    parser.add_argument("--gcs-bucket", type=str, default="semicon-metrology-models", help="GCS bucket for model artifacts")
    parser.add_argument("--data-dir", type=str, default="data/pcb_dataset", help="Local directory containing Kaggle dataset")
    
    args = parser.parse_args()
    run_workbench_training(
        k_shot=args.k_shot,
        epochs=args.epochs,
        learning_rate=args.lr,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        gcs_bucket=args.gcs_bucket,
        data_dir_path=args.data_dir
    )
