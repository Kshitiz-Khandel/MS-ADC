import math
from typing import Dict, Any, List

class SemiconductorYieldCalculator:
    """
    Domain math library (Comp 5) translating defect classifications and metrology
    indicators into actionable business and engineering yields.
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
