import unittest
import tempfile
import os

from src.models.fine_tune_vfm import run_training_pipeline, run_experiment_progression

class TestExperimentTracking(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_run_training_pipeline_execution(self):
        res = run_training_pipeline()
        self.assertIn("model", res)
        self.assertIn("final_metrics", res)
        self.assertGreaterEqual(res["final_metrics"]["train_acc"], 0.0)
