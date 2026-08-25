import os
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

DIE_DEFECT_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

class PCBDefectDatasetLoader:
    """
    Ingests and organizes raw optical microscopy micrographs from the Kaggle PCB Defect dataset.
    Normalizes image sizes (224x224 RGB) and splits them into Train, Validation, and Test sets.
    """
    def __init__(self, data_dir: Optional[Path] = None, target_size: Tuple[int, int] = (224, 224)):
        self.data_dir = data_dir or (Path(__file__).parent.parent.parent / "data" / "pcb_dataset")
        self.target_size = target_size
        self.classes = DIE_DEFECT_CLASSES

    def discover_image_files(self) -> Dict[str, List[Path]]:
        """
        Recursively discovers all image files and categorizes them by defect class.
        Matches exact parent folder names (e.g. Missing_hole, Short, Spurious_copper).
        """
        discovered: Dict[str, List[Path]] = {cls: [] for cls in self.classes}
        
        if not self.data_dir.exists():
            return discovered

        # Sort classes by length descending so spurious_copper matches before spur
        sorted_classes = sorted(self.classes, key=lambda c: len(c), reverse=True)

        for file_path in self.data_dir.rglob("*"):
            if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                parent_name = file_path.parent.name.lower().replace("-", "_").replace(" ", "_")
                stem_name = file_path.stem.lower().replace("-", "_").replace(" ", "_")

                for cls in sorted_classes:
                    if parent_name == cls or parent_name == f"{cls}s" or cls in stem_name or cls in parent_name:
                        discovered[cls].append(file_path)
                        break

        # Sort for reproducible ordering
        for cls in self.classes:
            discovered[cls].sort()

        return discovered

    def load_and_preprocess_image(self, image_path: Path) -> Image.Image:
        """Loads and converts image to standard 224x224 RGB PIL image."""
        img = Image.open(image_path).convert("RGB")
        return img.resize(self.target_size, Image.Resampling.BILINEAR)

    def get_stratified_split(
        self,
        k_shot_train: int = 10,
        val_ratio: float = 0.2,
        seed: int = 42
    ) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]], Dict[str, List[Path]]]:
        """
        Partitions discovered images into a 3-way split:
        - Train: K-shot support set (e.g., 10 samples per class)
        - Validation: val_ratio of remaining samples (for model checkpointing)
        - Test: Held-out unseen test samples (for final evaluation metrics)
        """
        all_files = self.discover_image_files()
        train_split: Dict[str, List[Path]] = {}
        val_split: Dict[str, List[Path]] = {}
        test_split: Dict[str, List[Path]] = {}

        rng = random.Random(seed)

        for cls, paths in all_files.items():
            shuffled = paths[:]
            rng.shuffle(shuffled)

            # 1. Train set (K-shot)
            if len(shuffled) >= k_shot_train:
                train_split[cls] = shuffled[:k_shot_train]
                remaining = shuffled[k_shot_train:]
            else:
                train_split[cls] = shuffled[:]
                remaining = []

            # 2. Validation & Test sets from remaining
            if remaining:
                val_count = max(1, int(len(remaining) * val_ratio)) if len(remaining) > 1 else 0
                val_split[cls] = remaining[:val_count]
                test_split[cls] = remaining[val_count:]
            else:
                val_split[cls] = []
                test_split[cls] = []

        return train_split, val_split, test_split

    def get_k_shot_split(self, k_shot: int = 10) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
        """Backward-compatible helper returning (train_split, test_split)."""
        train_split, val_split, test_split = self.get_stratified_split(k_shot_train=k_shot, val_ratio=0.0)
        return train_split, test_split
