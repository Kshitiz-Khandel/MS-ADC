import os
import random
import xml.etree.ElementTree as ET
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
    Ingests and organizes optical microscopy micrographs from the Kaggle PCB Defect dataset.
    Extracts high-resolution localized defect patches (ROI) using bounding box annotations
    and normalizes image sizes (224x224 RGB) for few-shot Vision Foundation Model inspection.
    """
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        target_size: Tuple[int, int] = (224, 224),
        crop_padding: int = 40
    ):
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent.parent.parent / "data" / "pcb_dataset"
        self.target_size = target_size
        self.crop_padding = crop_padding
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
            if file_path.is_file() and file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
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

    def find_annotation_xml(self, image_path: Path) -> Optional[Path]:
        """Locates corresponding Pascal VOC XML annotation file for given image."""
        # Check standard Kaggle hierarchy: PCB_DATASET/Annotations/<Class>/<stem>.xml
        parent_class = image_path.parent.name
        candidates = [
            image_path.parents[1] / "Annotations" / parent_class / f"{image_path.stem}.xml",
            image_path.parents[2] / "Annotations" / parent_class / f"{image_path.stem}.xml",
            image_path.parent / f"{image_path.stem}.xml",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def load_and_preprocess_image(self, image_path: Path) -> Image.Image:
        """
        Loads optical micrograph and crops the high-resolution localized defect patch (ROI).
        If bounding box annotation exists, crops the defect region with padding.
        Returns standardized 224x224 RGB image patch.
        """
        img = Image.open(image_path).convert("RGB")
        xml_path = self.find_annotation_xml(image_path)

        if xml_path and xml_path.exists():
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                obj = root.find("object")
                if obj is not None:
                    bndbox = obj.find("bndbox")
                    if bndbox is not None:
                        xmin = max(0, int(bndbox.find("xmin").text) - self.crop_padding)
                        ymin = max(0, int(bndbox.find("ymin").text) - self.crop_padding)
                        xmax = min(img.width, int(bndbox.find("xmax").text) + self.crop_padding)
                        ymax = min(img.height, int(bndbox.find("ymax").text) + self.crop_padding)

                        if xmax > xmin and ymax > ymin:
                            img = img.crop((xmin, ymin, xmax, ymax))
            except Exception:
                pass

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
