import os
import time
import math
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from src.models.die_vfm import DieVFMClassifier, DIE_DEFECT_CLASSES

class VFMFineTuner:
    """
    Few-shot linear probe optimizer for Vision Foundation Model representations.
    Utilizes AdamW optimization and Cosine Annealing learning rate schedule for fast, stable convergence.
    Includes validation monitoring and best-model checkpointing.
    """
    def __init__(self, model: DieVFMClassifier, learning_rate: float = 0.01):
        self.model = model
        self.lr = learning_rate

    def generate_few_shot_data(self, k_shot: int = 10) -> Tuple[List[List[float]], List[int]]:
        """Generates synthetic K-shot support set (K samples per class)."""
        X_list = []
        y_list = []
        for class_idx in range(len(DIE_DEFECT_CLASSES)):
            for _ in range(k_shot):
                feat = [random.gauss(0, 0.05) for _ in range(self.model.embedding_dim)]
                start = class_idx * 10
                for d in range(start, start + 10):
                    feat[d] += 2.0
                X_list.append(feat)
                y_list.append(class_idx)
                
        return X_list, y_list

    def evaluate(self, X: List[List[float]], y: List[int]) -> Tuple[float, float, List[int]]:
        """Computes average loss, accuracy, and predicted labels on given dataset."""
        if not y:
            return 0.0, 0.0, []

        total_loss = 0.0
        correct = 0
        predictions = []

        for i in range(len(y)):
            logits = self.model.predict_logits(X[i])
            probs = self.model.softmax(logits)
            pred = logits.index(max(logits))
            predictions.append(pred)

            loss = -math.log(probs[y[i]] + 1e-8)
            total_loss += loss

            if pred == y[i]:
                correct += 1

        avg_loss = total_loss / len(y)
        accuracy = (correct / len(y)) * 100.0
        return float(avg_loss), float(accuracy), predictions

    def train_with_validation(
        self,
        X_train: List[List[float]],
        y_train: List[int],
        X_val: Optional[List[List[float]]] = None,
        y_val: Optional[List[int]] = None,
        epochs: int = 35,
        checkpoint_dir: Optional[str] = "models"
    ) -> Dict[str, Any]:
        """
        Trains linear probe with validation monitoring and saves checkpoint_best.pt.
        """
        start_time = time.time()
        train_loss_history = []
        val_loss_history = []
        val_acc_history = []

        best_val_acc = -1.0
        best_val_loss = float("inf")
        best_epoch = 0
        best_checkpoint_path = None

        os.makedirs(checkpoint_dir or "models", exist_ok=True)

        if self.model.use_pytorch:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            device = self.model.device
            head = self.model.torch_head.to(device)
            head.train()
            optimizer = optim.AdamW(head.parameters(), lr=self.lr, weight_decay=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = nn.CrossEntropyLoss()

            X_tr_t = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_tr_t = torch.tensor(y_train, dtype=torch.long).to(device)

            for epoch in range(1, epochs + 1):
                optimizer.zero_grad()
                logits = head(X_tr_t)
                loss = criterion(logits, y_tr_t)
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss = float(loss.item())
                train_loss_history.append(round(train_loss, 4))

                # Sync into CPU python weights
                with torch.no_grad():
                    self.model.weights = head.weight.t().cpu().tolist()
                    self.model.bias = head.bias.cpu().tolist()

                if X_val and y_val:
                    val_loss, val_acc, _ = self.evaluate(X_val, y_val)
                    val_loss_history.append(round(val_loss, 4))
                    val_acc_history.append(round(val_acc, 2))

                    if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                        best_val_acc = val_acc
                        best_val_loss = val_loss
                        best_epoch = epoch
                        best_path = os.path.join(checkpoint_dir or "models", "checkpoint_best.pt")
                        best_checkpoint_path = self.model.save_checkpoint(
                            best_path,
                            epoch=epoch,
                            val_accuracy=val_acc,
                            metadata={"train_loss": train_loss, "val_loss": val_loss}
                        )
                else:
                    if train_loss < best_val_loss:
                        best_val_loss = train_loss
                        best_epoch = epoch
                        best_path = os.path.join(checkpoint_dir or "models", "checkpoint_best.pt")
                        best_checkpoint_path = self.model.save_checkpoint(
                            best_path,
                            epoch=epoch,
                            metadata={"train_loss": train_loss}
                        )
        else:
            # Fallback pure python SGD
            for epoch in range(1, epochs + 1):
                total_loss = 0.0
                for i in range(len(y_train)):
                    feat = X_train[i]
                    label = y_train[i]
                    logits = self.model.predict_logits(feat)
                    probs = self.model.softmax(logits)
                    loss = -math.log(probs[label] + 1e-8)
                    total_loss += loss
                    grad_logits = [probs[j] - (1.0 if j == label else 0.0) for j in range(self.model.num_classes)]
                    for d in range(self.model.embedding_dim):
                        for j in range(self.model.num_classes):
                            self.model.weights[d][j] -= self.lr * feat[d] * grad_logits[j]
                    for j in range(self.model.num_classes):
                        self.model.bias[j] -= self.lr * grad_logits[j]

                train_loss = total_loss / max(1, len(y_train))
                train_loss_history.append(round(train_loss, 4))

                if X_val and y_val:
                    val_loss, val_acc, _ = self.evaluate(X_val, y_val)
                    val_loss_history.append(round(val_loss, 4))
                    val_acc_history.append(round(val_acc, 2))
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_epoch = epoch
                        best_path = os.path.join(checkpoint_dir or "models", "checkpoint_best.pt")
                        best_checkpoint_path = self.model.save_checkpoint(
                            best_path, epoch=epoch, val_accuracy=val_acc
                        )

        elapsed_sec = time.time() - start_time
        final_train_loss, final_train_acc, _ = self.evaluate(X_train, y_train)

        final_path = os.path.join(checkpoint_dir or "models", "model_final.pt")
        self.model.save_checkpoint(
            final_path,
            epoch=epochs,
            val_accuracy=best_val_acc if X_val else final_train_acc,
            metadata={"train_loss_history": train_loss_history, "val_loss_history": val_loss_history}
        )

        return {
            "k_shot": len(y_train) // max(1, len(DIE_DEFECT_CLASSES)),
            "train_samples": len(y_train),
            "val_samples": len(y_val) if y_val else 0,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc if X_val else round(final_train_acc, 2),
            "final_train_accuracy": round(final_train_acc, 2),
            "final_train_loss": round(final_train_loss, 4),
            "best_checkpoint_path": best_checkpoint_path,
            "training_time_sec": round(elapsed_sec, 3),
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "val_acc_history": val_acc_history
        }

    def train_custom_dataset(self, X: List[List[float]], y: List[int], epochs: int = 20) -> Dict[str, Any]:
        res = self.train_with_validation(X, y, epochs=epochs)
        return {
            "k_shot": res["k_shot"],
            "total_samples": res["train_samples"],
            "epochs": res["epochs"],
            "final_loss": res["final_train_loss"],
            "accuracy_pct": res["final_train_accuracy"],
            "training_time_sec": res["training_time_sec"],
            "loss_history": res["train_loss_history"]
        }

    def run_training(self, k_shot: int = 10, epochs: int = 15) -> Dict[str, Any]:
        X, y = self.generate_few_shot_data(k_shot=k_shot)
        return self.train_custom_dataset(X, y, epochs=epochs)
