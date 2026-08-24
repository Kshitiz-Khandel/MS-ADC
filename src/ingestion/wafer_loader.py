import io
import base64
import math
import random
from typing import Dict, Any, Tuple, Optional, List

class WM811KWaferLoader:
    """
    Ingests and transforms raw WM-811K 2D spatial wafer matrices (0=Air, 1=Pass, 2=Defect)
    into normalized RGB visual assets for multimodal Vision-Language Models (Gemini 2.0).
    Computes mathematical cleanroom yield KPIs (D0 Defect Density, Die Yield).
    """
    
    COLOR_MAP = {
        0: (0, 0, 0),       # Background air (Black)
        1: (40, 180, 40),   # Passing functional die (Green)
        2: (230, 30, 30)    # Defective failing die (Red)
    }

    def __init__(self, target_resolution: Tuple[int, int] = (224, 224)):
        self.target_res = target_resolution

    def compute_kpis(self, wafer_matrix: List[List[int]]) -> Dict[str, float]:
        """Calculates semiconductor Defect Density (D0) and Gross Die Yield."""
        total_dies = 0
        defective_dies = 0
        
        for row in wafer_matrix:
            for val in row:
                if val > 0:
                    total_dies += 1
                if val == 2:
                    defective_dies += 1
        
        if total_dies == 0:
            return {"total_dies": 0, "defective_dies": 0, "defect_density_d0": 0.0, "die_yield_pct": 100.0}
        
        d0 = float(defective_dies / total_dies)
        yield_pct = float((1.0 - d0) * 100.0)
        
        return {
            "total_dies": total_dies,
            "defective_dies": defective_dies,
            "defect_density_d0": round(d0, 4),
            "die_yield_pct": round(yield_pct, 2)
        }

    def render_base64_representation(self, wafer_matrix: List[List[int]]) -> str:
        """Encodes wafer spatial matrix into a lightweight visual representation."""
        # Simulated PNG base64 header for zero-dependency portability
        raw_repr = f"WAFER_GRID_{len(wafer_matrix)}x{len(wafer_matrix[0])}"
        return base64.b64encode(raw_repr.encode("utf-8")).decode("utf-8")

    def process_wafer(
        self,
        wafer_matrix: List[List[int]],
        lot_id: str,
        wafer_index: int,
        failure_type: str = "Unknown"
    ) -> Dict[str, Any]:
        """Transforms a raw wafer matrix into a fully structured inspection payload."""
        kpis = self.compute_kpis(wafer_matrix)
        b64_str = self.render_base64_representation(wafer_matrix)
        
        return {
            "lot_id": lot_id,
            "wafer_index": wafer_index,
            "failure_type": failure_type,
            "kpis": kpis,
            "image_base64": b64_str,
            "gcs_uri": f"gs://semicon-metrology-raw/{lot_id}/wafer_{wafer_index}_map.png",
            "resolution": list(self.target_res)
        }

    def generate_synthetic_wafer(
        self,
        pattern: str = "Center",
        size: int = 50
    ) -> List[List[int]]:
        """Generates realistic synthetic 300mm wafer spatial bin maps for unit testing and CI."""
        matrix = [[0 for _ in range(size)] for _ in range(size)]
        radius = size // 2
        center = (size // 2, size // 2)
        
        for r in range(size):
            for c in range(size):
                dist = math.sqrt((r - center[0])**2 + (c - center[1])**2)
                if dist <= radius:
                    matrix[r][c] = 1  # Good die
                    
                    if pattern == "Center" and dist < radius * 0.4:
                        matrix[r][c] = 2
                    elif pattern == "Edge-Ring" and dist > radius * 0.75:
                        matrix[r][c] = 2
                    elif pattern == "Scratch" and abs(r - c) <= 1 and size // 4 <= r <= 3 * size // 4:
                        matrix[r][c] = 2
                    elif pattern == "Random" and random.random() < 0.05:
                        matrix[r][c] = 2

        return matrix
