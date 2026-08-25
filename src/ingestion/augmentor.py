import math
import random
from typing import List, Dict, Any, Tuple, Optional

class CleanroomDataAugmentor:
    """
    Applies minority-class synthetic data augmentation for semiconductor die micrographs and wafer maps.
    Synthesizes realistic defect variations to prevent model overfitting on sparse excursion events.
    """
    def __init__(self, rotation_angles: Optional[List[int]] = None):
        self.rotation_angles = rotation_angles or [0, 90, 180, 270]

    def augment_matrix(self, matrix: List[List[int]], flip_horizontal: bool = True, flip_vertical: bool = False) -> List[List[int]]:
        """Applies spatial transforms (reflections, rotations) on a 2D wafer/die matrix."""
        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0
        new_matrix = [row[:] for row in matrix]

        if flip_horizontal:
            new_matrix = [row[::-1] for row in new_matrix]

        if flip_vertical:
            new_matrix = new_matrix[::-1]

        return new_matrix

    def inject_defect_cluster(
        self,
        matrix: List[List[int]],
        defect_code: int = 2,
        cluster_radius: int = 3,
        center_coords: Optional[Tuple[int, int]] = None
    ) -> List[List[int]]:
        """Synthesizes localized defect clusters to augment rare spatial defect types."""
        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0
        new_matrix = [row[:] for row in matrix]

        if center_coords is None:
            cr = random.randint(cluster_radius, max(cluster_radius, rows - cluster_radius - 1))
            cc = random.randint(cluster_radius, max(cluster_radius, cols - cluster_radius - 1))
        else:
            cr, cc = center_coords

        for r in range(max(0, cr - cluster_radius), min(rows, cr + cluster_radius + 1)):
            for c in range(max(0, cc - cluster_radius), min(cols, cc + cluster_radius + 1)):
                dist = math.sqrt((r - cr) ** 2 + (c - cc) ** 2)
                if dist <= cluster_radius and new_matrix[r][c] > 0:
                    if random.random() < 0.75:
                        new_matrix[r][c] = defect_code

        return new_matrix

    def augment_feature_vector(self, feature_vector: List[float], noise_std: float = 0.02) -> List[float]:
        """Applies Gaussian feature jittering in representation space for few-shot linear head training."""
        augmented = [x + random.gauss(0, noise_std) for x in feature_vector]
        norm = math.sqrt(sum(x ** 2 for x in augmented)) + 1e-8
        return [x / norm for x in augmented]
