import unittest
import tempfile
import os
import glob
from pathlib import Path

from src.models.fine_tune_vfm import run_training_pipeline


class TestExperimentTracking(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.models_dir = os.path.join(self.temp_dir.name, "models")
        self.data_dir = os.path.join(self.temp_dir.name, "pcb_data")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_tensorboard_and_mlflow_integration(self):
        res = run_training_pipeline(
            version="v1.0.0-track-test",
            epochs=2,
            output_dir=self.models_dir,
            data_dir_path=self.data_dir,
            use_tracking=True
        )
        self.assertEqual(res["version"], "v1.0.0-track-test")
        self.assertTrue(os.path.exists(res["version_output_dir"]))

        # Check TensorBoard runs directory
        tb_dir = os.path.join("runs", "v1.0.0-track-test")
        self.assertTrue(os.path.exists(tb_dir))
        event_files = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))
        self.assertGreaterEqual(len(event_files), 1)


if __name__ == "__main__":
    unittest.main()
