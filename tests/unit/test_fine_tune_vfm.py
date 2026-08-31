import unittest
import tempfile
import os

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from src.models.fine_tune_vfm import (
    FineTuneConfig,
    SyntheticDataset,
    train_head,
    run_training_pipeline
)
from src.models.die_vfm import DieVFMClassifier

class TestFineTuneVFM(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = FineTuneConfig(
            epochs=2,
            batch_size=4,
            learning_rate=0.01,
            num_samples=16,
            num_classes=6,
            checkpoint_dir=self.temp_dir.name
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_synthetic_dataset_generation(self):
        dataset = SyntheticDataset(num_samples=12, num_classes=6)
        self.assertEqual(len(dataset), 12)
        sample = dataset[0]
        self.assertIn("image", sample)
        self.assertIn("label", sample)
        self.assertGreaterEqual(sample["label"], 0)
        self.assertLess(sample["label"], 6)

    def test_train_head_pure_python(self):
        model = DieVFMClassifier(num_classes=6, embedding_dim=512)
        dataset = SyntheticDataset(num_samples=16, num_classes=6)
        metrics = train_head(model, dataset, self.config)
        self.assertIn("train_loss", metrics)
        self.assertIn("train_acc", metrics)
        self.assertGreaterEqual(metrics["train_acc"], 0.0)

    def test_run_training_pipeline_end_to_end(self):
        res = run_training_pipeline(self.config)
        self.assertIn("model", res)
        self.assertIn("final_metrics", res)
        self.assertIn("checkpoint_path", res)
        self.assertTrue(os.path.exists(res["checkpoint_path"]))
