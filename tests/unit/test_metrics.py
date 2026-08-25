import unittest
from src.utils.metrics import SemiconductorYieldCalculator
from src.models.die_vfm import DIE_DEFECT_CLASSES

class TestSemiconductorMetricsSuite(unittest.TestCase):
    def test_defect_density_calculation(self):
        d0 = SemiconductorYieldCalculator.calculate_defect_density(5, 100)
        self.assertAlmostEqual(d0, 0.05, places=4)
        
        d0_zero = SemiconductorYieldCalculator.calculate_defect_density(0, 0)
        self.assertEqual(d0_zero, 0.0)

    def test_murphy_perfect_yield(self):
        # When defect density D0 = 0, yield must be 1.0 (100%)
        y = SemiconductorYieldCalculator.calculate_murphy_yield(1.5, 0.0)
        self.assertEqual(y, 1.0)
        
        y_zero_area = SemiconductorYieldCalculator.calculate_murphy_yield(0.0, 0.5)
        self.assertEqual(y_zero_area, 1.0)

    def test_murphy_standard_calculations(self):
        # A = 1.5 cm2, D0 = 0.5 defects/cm2 -> AD0 = 0.75
        # Expected: ((1 - e^-0.75) / 0.75)^2 = 0.4949
        y = SemiconductorYieldCalculator.calculate_murphy_yield(1.5, 0.5)
        self.assertAlmostEqual(y, 0.4949, places=3)

    def test_seeds_standard_calculations(self):
        # A = 1.5 cm2, D0 = 0.5 defects/cm2 -> AD0 = 0.75
        # Expected: e^-sqrt(0.75) = 0.4206
        y = SemiconductorYieldCalculator.calculate_seeds_yield(1.5, 0.5)
        self.assertAlmostEqual(y, 0.4206, places=3)
        
        y_zero = SemiconductorYieldCalculator.calculate_seeds_yield(1.5, 0.0)
        self.assertEqual(y_zero, 1.0)

    def test_escaped_defect_rate_boundaries(self):
        self.assertEqual(SemiconductorYieldCalculator.calculate_escaped_defect_rate(0, 50), 0.0)
        self.assertAlmostEqual(SemiconductorYieldCalculator.calculate_escaped_defect_rate(5, 50), 0.10, places=4)
        self.assertEqual(SemiconductorYieldCalculator.calculate_escaped_defect_rate(5, 0), 0.0)

    def test_classification_metrics_and_report(self):
        y_true = [0, 1, 2, 3, 4, 5, 0, 1]
        y_pred = [0, 1, 2, 3, 4, 5, 0, 0] # One misprediction on index 7 (true 1, pred 0)
        metrics = SemiconductorYieldCalculator.calculate_classification_metrics(y_true, y_pred, DIE_DEFECT_CLASSES)
        
        self.assertEqual(metrics["total_samples"], 8)
        self.assertAlmostEqual(metrics["accuracy"], 7 / 8, places=3)
        self.assertIn("missing_hole", metrics["classes"])
        self.assertEqual(len(metrics["confusion_matrix"]), len(DIE_DEFECT_CLASSES))

        report_str = SemiconductorYieldCalculator.format_classification_report(metrics)
        self.assertIn("missing_hole", report_str)
        self.assertIn("Accuracy", report_str)

if __name__ == "__main__":
    unittest.main()
