import time
import math
import random
from typing import Dict, Any, List, Tuple
from src.models.die_vfm import DieVFMClassifier, DIE_DEFECT_CLASSES

class VFMFineTuner:
    """
    Lightweight few-shot linear head trainer for NV-DINOv2 representations.
    Demonstrates rapid convergence (<5 minutes) on K-shot labeled cleanroom datasets.
    """
    def __init__(self, model: DieVFMClassifier, learning_rate: float = 0.05):
        self.model = model
        self.lr = learning_rate

    def generate_few_shot_data(self, k_shot: int = 10) -> Tuple[List[List[float]], List[int]]:
        """Generates synthetic K-shot support set (K samples per class)."""
        X_list = []
        y_list = []
        for class_idx in range(len(DIE_DEFECT_CLASSES)):
            for _ in range(k_shot):
                feat = [random.gauss(0, 0.05) for _ in range(self.model.embedding_dim)]
                # Add distinctive signature for target class
                start = class_idx * 10
                for d in range(start, start + 10):
                    feat[d] += 2.0
                X_list.append(feat)
                y_list.append(class_idx)
                
        return X_list, y_list

    def train_epoch(self, X: List[List[float]], y: List[int]) -> float:
        """Executes one SGD epoch over linear head weights."""
        total_loss = 0.0
        num_samples = len(y)
        
        for i in range(num_samples):
            feat = X[i]
            label = y[i]
            
            # Forward
            logits = self.model.predict_logits(feat)
            probs = self.model.softmax(logits)
            
            # Cross-Entropy Loss
            loss = -math.log(probs[label] + 1e-8)
            total_loss += loss
            
            # Gradient: dL/dlogits = probs - y_onehot
            grad_logits = [probs[j] - (1.0 if j == label else 0.0) for j in range(self.model.num_classes)]
            
            # Update W and b
            for d in range(self.model.embedding_dim):
                for j in range(self.model.num_classes):
                    self.model.weights[d][j] -= self.lr * feat[d] * grad_logits[j]
            for j in range(self.model.num_classes):
                self.model.bias[j] -= self.lr * grad_logits[j]
            
        return float(total_loss / num_samples)

    def run_training(self, k_shot: int = 10, epochs: int = 15) -> Dict[str, Any]:
        start_time = time.time()
        X, y = self.generate_few_shot_data(k_shot=k_shot)
        
        loss_history = []
        for epoch in range(epochs):
            avg_loss = self.train_epoch(X, y)
            loss_history.append(round(avg_loss, 4))
            
        elapsed_sec = time.time() - start_time
        
        # Compute final accuracy
        correct = 0
        for i in range(len(y)):
            logits = self.model.predict_logits(X[i])
            pred = logits.index(max(logits))
            if pred == y[i]:
                correct += 1
        accuracy = float(correct / len(y)) * 100.0
        
        return {
            "k_shot": k_shot,
            "total_samples": len(y),
            "epochs": epochs,
            "final_loss": loss_history[-1],
            "accuracy_pct": round(accuracy, 2),
            "training_time_sec": round(elapsed_sec, 3),
            "loss_history": loss_history
        }
