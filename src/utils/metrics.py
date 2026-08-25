import math
from typing import Dict, Any, List, Tuple, Optional

class SemiconductorYieldCalculator:
    """
    Semiconductor cleanroom yield and metrology analytics calculator.
    Translates defect classifications and metrology indicators into actionable engineering metrics.
    """

    @staticmethod
    def calculate_defect_density(defective_dies: int, total_dies: int) -> float:
        """Computes defect density (D0) as the ratio of failing dies to total dies."""
        if total_dies <= 0:
            return 0.0
        return float(defective_dies / total_dies)

    @staticmethod
    def calculate_murphy_yield(die_area_cm2: float, defect_density_d0: float) -> float:
        """
        Computes gross die yield using Murphy's Model:
        Y = ((1 - e^{-A * D_0}) / (A * D_0))^2
        """
        if die_area_cm2 <= 0 or defect_density_d0 <= 0:
            return 1.0
        ad0 = die_area_cm2 * defect_density_d0
        numerator = 1.0 - math.exp(-ad0)
        return (numerator / ad0) ** 2

    @staticmethod
    def calculate_seeds_yield(die_area_cm2: float, defect_density_d0: float) -> float:
        """
        Computes gross die yield using Seeds' Model for clustered defects:
        Y = e^{-sqrt(A * D_0)}
        """
        if die_area_cm2 <= 0 or defect_density_d0 <= 0:
            return 1.0
        ad0 = die_area_cm2 * defect_density_d0
        return math.exp(-math.sqrt(ad0))

    @staticmethod
    def calculate_escaped_defect_rate(false_negatives: int, total_true_defects: int) -> float:
        """
        Calculates Escaped Defect Rate (EDR) = False Negatives / Total True Defects.
        Critical for preventing defective dies from entering advanced packaging.
        """
        if total_true_defects <= 0:
            return 0.0
        return float(false_negatives / total_true_defects)

    @staticmethod
    def calculate_classification_metrics(
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str]
    ) -> Dict[str, Any]:
        """
        Computes accuracy, per-class precision/recall/F1, macro/weighted averages, and confusion matrix.
        """
        num_classes = len(class_names)
        total_samples = len(y_true)
        if total_samples == 0:
            return {"accuracy": 0.0, "macro_f1": 0.0, "classes": {}, "confusion_matrix": []}

        # Initialize Confusion Matrix [actual][predicted]
        cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        for actual, pred in zip(y_true, y_pred):
            if 0 <= actual < num_classes and 0 <= pred < num_classes:
                cm[actual][pred] += 1

        # Accuracy
        correct = sum(cm[i][i] for i in range(num_classes))
        accuracy = float(correct / total_samples)

        # Per-class metrics
        class_metrics = {}
        macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
        weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0

        for i, name in enumerate(class_names):
            tp = cm[i][i]
            fn = sum(cm[i][j] for j in range(num_classes)) - tp
            fp = sum(cm[j][i] for j in range(num_classes)) - tp
            support = tp + fn

            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            class_metrics[name] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "support": support
            }

            macro_p += precision
            macro_r += recall
            macro_f1 += f1

            weighted_p += precision * support
            weighted_r += recall * support
            weighted_f1 += f1 * support

        macro_p /= num_classes
        macro_r /= num_classes
        macro_f1 /= num_classes

        weighted_p /= total_samples
        weighted_r /= total_samples
        weighted_f1 /= total_samples

        return {
            "total_samples": total_samples,
            "accuracy": round(accuracy, 4),
            "macro_avg": {
                "precision": round(macro_p, 4),
                "recall": round(macro_r, 4),
                "f1_score": round(macro_f1, 4)
            },
            "weighted_avg": {
                "precision": round(weighted_p, 4),
                "recall": round(weighted_r, 4),
                "f1_score": round(weighted_f1, 4)
            },
            "classes": class_metrics,
            "confusion_matrix": cm,
            "class_names": class_names
        }

    @staticmethod
    def format_classification_report(metrics: Dict[str, Any]) -> str:
        """Formats the evaluation metrics dictionary into a readable terminal table."""
        lines = []
        lines.append(f"{'Class':<18} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
        lines.append("-" * 58)

        for name, data in metrics.get("classes", {}).items():
            p = f"{data['precision'] * 100:.1f}%"
            r = f"{data['recall'] * 100:.1f}%"
            f = f"{data['f1_score'] * 100:.1f}%"
            s = str(data['support'])
            lines.append(f"{name:<18} {p:<10} {r:<10} {f:<10} {s:<8}")

        lines.append("-" * 58)
        acc_pct = f"{metrics.get('accuracy', 0.0) * 100:.2f}%"
        total_s = str(metrics.get('total_samples', 0))
        lines.append(f"{'Accuracy':<18} {'':<10} {'':<10} {acc_pct:<10} {total_s:<8}")

        m_p = f"{metrics.get('macro_avg', {}).get('precision', 0.0) * 100:.1f}%"
        m_r = f"{metrics.get('macro_avg', {}).get('recall', 0.0) * 100:.1f}%"
        m_f = f"{metrics.get('macro_avg', {}).get('f1_score', 0.0) * 100:.1f}%"
        lines.append(f"{'Macro Avg':<18} {m_p:<10} {m_r:<10} {m_f:<10} {total_s:<8}")

        w_p = f"{metrics.get('weighted_avg', {}).get('precision', 0.0) * 100:.1f}%"
        w_r = f"{metrics.get('weighted_avg', {}).get('recall', 0.0) * 100:.1f}%"
        w_f = f"{metrics.get('weighted_avg', {}).get('f1_score', 0.0) * 100:.1f}%"
        lines.append(f"{'Weighted Avg':<18} {w_p:<10} {w_r:<10} {w_f:<10} {total_s:<8}")

        return "\n".join(lines)
