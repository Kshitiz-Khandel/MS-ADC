import os
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
    Normalizes image sizes (224x224 RGB) and splits them into K-shot support and query/test sets.
    """
    def __init__(self, data_dir: Optional[Path] = None, target_size: Tuple[int, int] = (224, 224)):
        self.data_dir = data_dir or (Path(__file__).parent.parent.parent / "data" / "pcb_dataset")
        self.target_size = target_size
        self.classes = DIE_DEFECT_CLASSES

    def discover_image_files(self) -> Dict[str, List[Path]]:
        """
        Recursively discovers all image files and categorizes them by defect class.
        Handles variations in Kaggle directory naming (e.g. Missing_hole, Short, open_circuit).
        """
        discovered: Dict[str, List[Path]] = {cls: [] for cls in self.classes}
        
        if not self.data_dir.exists():
            return discovered

        for file_path in self.data_dir.rglob("*"):
            if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                parent_name = file_path.parent.name.lower()
                stem_name = file_path.stem.lower()

                for cls in self.classes:
                    if cls in parent_name or cls in stem_name or cls.replace("_", "") in parent_name:
                        discovered[cls].append(file_path)
                        break

        return discovered

    def load_and_preprocess_image(self, image_path: Path) -> Image.Image:
        """Loads and converts image to standard 224x224 RGB PIL image."""
        img = Image.open(image_path).convert("RGB")
        return img.resize(self.target_size, Image.Resampling.BILINEAR)

    def get_k_shot_split(self, k_shot: int = 10) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
        """Partitions discovered images into K-shot training support set and testing set."""
        all_files = self.discover_image_files()
        train_split: Dict[str, List[Path]] = {}
        test_split: Dict[str, List[Path]] = {}

        for cls, paths in all_files.items():
            if len(paths) >= k_shot:
                train_split[cls] = paths[:k_shot]
                test_split[cls] = paths[k_shot:]
            else:
                train_split[cls] = paths[:]
                test_split[cls] = []

        return train_split, test_split
