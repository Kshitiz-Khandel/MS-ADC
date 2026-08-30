import os
from typing import List, Dict, Any, Optional
from pathlib import Path


class SemiconductorYieldCalculator:
    """
    Computes metrology defect classification metrics, confusion matrices,
    and generates visual plots for semiconductor cleanroom inspection.
    """

    @staticmethod
    def calculate_classification_metrics(
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str]
    ) -> Dict[str, Any]:
        """Calculates multi-class confusion matrix, precision, recall, F1, and accuracy."""
        n_classes = len(class_names)
        cm = [[0 for _ in range(n_classes)] for _ in range(n_classes)]

        for t, p in zip(y_true, y_pred):
            if 0 <= t < n_classes and 0 <= p < n_classes:
                cm[t][p] += 1

        total_samples = len(y_true)
        correct_samples = sum(cm[i][i] for i in range(n_classes))
        accuracy = (correct_samples / max(1, total_samples)) * 100.0

        per_class_metrics = {}
        precisions, recalls, f1s = [], [], []

        for i, name in enumerate(class_names):
            tp = cm[i][i]
            fp = sum(cm[row][i] for row in range(n_classes)) - tp
            fn = sum(cm[i][col] for col in range(n_classes)) - tp
            support = sum(cm[i][col] for col in range(n_classes))

            prec = (tp / max(1, tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
            rec = (tp / max(1, tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / max(1e-8, prec + rec)) if (prec + rec) > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

            per_class_metrics[name] = {
                "precision": round(prec, 2),
                "recall": round(rec, 2),
                "f1_score": round(f1, 2),
                "support": support
            }

        macro_prec = sum(precisions) / max(1, n_classes)
        macro_rec = sum(recalls) / max(1, n_classes)
        macro_f1 = sum(f1s) / max(1, n_classes)

        return {
            "accuracy": round(accuracy, 2),
            "macro_precision": round(macro_prec, 2),
            "macro_recall": round(macro_rec, 2),
            "macro_f1": round(macro_f1, 2),
            "total_samples": total_samples,
            "classes": per_class_metrics,
            "confusion_matrix": cm
        }

    @staticmethod
    def format_classification_report(metrics: Dict[str, Any]) -> str:
        """Formats classification metrics into a clean terminal report string."""
        lines = [
            f"{'Class':<18} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}",
            "-" * 65
        ]
        for cls_name, vals in metrics.get("classes", {}).items():
            lines.append(
                f"{cls_name:<18} | {vals['precision']:>9.2f}% | {vals['recall']:>9.2f}% | {vals['f1_score']:>9.2f}% | {vals['support']:>8}"
            )
        lines.append("-" * 65)
        lines.append(
            f"{'Overall Accuracy':<18} : {metrics.get('accuracy', 0.0):.2f}% (Macro F1: {metrics.get('macro_f1', 0.0):.2f}%)"
        )
        return "\n".join(lines)

    @staticmethod
    def save_confusion_matrix_plot(
        cm: List[List[int]],
        class_names: List[str],
        output_path: str
    ) -> str:
        """Renders and saves a confusion matrix heatmap using matplotlib."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cm_arr = np.array(cm)

        fig, ax = plt.subplots(figsize=(7, 6))
        cax = ax.matshow(cm_arr, cmap=plt.cm.Blues, alpha=0.85)

        fig.colorbar(cax)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="left", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm_arr[i, j]
                ax.text(j, i, str(val), ha="center", va="center",
                        color="white" if val > cm_arr.max() / 2 else "black",
                        fontweight="bold")

        plt.title("MS-ADC Metrology Defect Confusion Matrix", pad=20, fontsize=12, fontweight="bold")
        plt.xlabel("Predicted Defect Class", labelpad=10, fontsize=10)
        plt.ylabel("Actual Ground Truth Class", labelpad=10, fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path

    @staticmethod
    def save_loss_accuracy_curves(
        history: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """Renders and saves training/validation loss and validation accuracy curves."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        epochs = [h["epoch"] for h in history]
        train_loss = [h.get("train_loss", 0.0) for h in history]
        val_loss = [h.get("val_loss", 0.0) for h in history]
        val_acc = [h.get("val_accuracy", 0.0) * 100.0 if h.get("val_accuracy", 0.0) <= 1.0 else h.get("val_accuracy", 0.0) for h in history]

        fig, ax1 = plt.subplots(figsize=(8, 5))

        color = "tab:red"
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Loss", color=color, fontsize=11)
        l1 = ax1.plot(epochs, train_loss, color="tab:red", linestyle="--", marker="o", label="Train Loss")
        l2 = ax1.plot(epochs, val_loss, color="tab:orange", linestyle="-", marker="s", label="Val Loss")
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("Validation Accuracy (%)", color=color, fontsize=11)
        l3 = ax2.plot(epochs, val_acc, color="tab:blue", linestyle="-", marker="^", label="Val Accuracy")
        ax2.tick_params(axis="y", labelcolor=color)

        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="center right")

        plt.title("MS-ADC Vision Foundation Model: Training Progression Curve", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path

    @staticmethod
    def save_precision_recall_f1_chart(
        class_metrics: Dict[str, Dict[str, float]],
        output_path: str
    ) -> str:
        """Renders a grouped bar chart of Precision, Recall, and F1 per defect class."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        classes = list(class_metrics.keys())
        precisions = [class_metrics[c]["precision"] for c in classes]
        recalls = [class_metrics[c]["recall"] for c in classes]
        f1s = [class_metrics[c]["f1_score"] for c in classes]

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width, precisions, width, label="Precision", color="#4285F4")
        ax.bar(x, recalls, width, label="Recall", color="#34A853")
        ax.bar(x + width, f1s, width, label="F1-Score", color="#FBBC05")

        ax.set_ylabel("Score (%)", fontsize=11)
        ax.set_title("MS-ADC Defect Classification Performance by Class", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path
