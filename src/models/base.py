from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


class DefectClassifierInterface(ABC):
    """
    Abstract Base Interface for MS-ADC Micro Metrology Defect Classifiers.
    Enforces standardized contracts across PyTorch, TensorRT, and Cloud vision models.
    """

    @abstractmethod
    def predict_logits(self, features_or_image: Any) -> List[float]:
        """Runs forward inference on image or feature embedding to compute class logits."""
        pass

    @abstractmethod
    def classify_patch(self, image_data: Any) -> Dict[str, Any]:
        """Classifies an optical microscopy die crop and returns structured prediction."""
        pass

    @abstractmethod
    def extract_features(self, image: Any) -> List[float]:
        """Extracts fixed-dimension embedding representation from an optical die crop."""
        pass

    @abstractmethod
    def save_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        epoch: Optional[int] = None,
        val_accuracy: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Serializes model weights and training metadata."""
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Loads serialized weights from disk."""
        pass
