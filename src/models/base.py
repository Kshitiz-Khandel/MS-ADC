import abc
from typing import Dict, Any, List, Optional

class DefectClassifierInterface(abc.ABC):
    """
    Abstract interface for automated semiconductor defect classification models.
    Enables plug-and-play swapping between PyTorch, TensorRT, Triton, and cloud inference endpoints.
    """

    @abc.abstractmethod
    def extract_features(self, image_data: Any) -> List[float]:
        """Extracts dense visual feature representations from an image patch."""
        pass

    @abc.abstractmethod
    def classify_patch(self, image_data: Any) -> Dict[str, Any]:
        """
        Classifies an input optical or e-beam die patch.
        Returns:
            Dict containing predicted_class, confidence, and class probabilities.
        """
        pass
