import unittest
import tempfile
import os

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from src.models.die_vfm import DieVFMClassifier, DIE_DEFECT_CLASSES
from src.models.export_tensorrt import TensorRTExporter

class TestDieVFM(unittest.TestCase):
    def setUp(self):
        self.classifier = DieVFMClassifier(num_classes=6, embedding_dim=512)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_predict_logits_shape(self):
        logits = self.classifier.predict_logits("dummy_path.png")
        self.assertEqual(len(logits), 6)

    def test_classify_patch(self):
        pred = self.classifier.classify_patch("dummy_path.png")
        self.assertIn("predicted_class", pred)
        self.assertIn(pred["predicted_class"], DIE_DEFECT_CLASSES)
        self.assertGreaterEqual(pred["confidence"], 0.0)
        self.assertLessEqual(pred["confidence"], 1.0)
        self.assertEqual(len(pred["all_probabilities"]), 6)

    def test_extract_features(self):
        feats = self.classifier.extract_features("dummy_path.png")
        self.assertEqual(len(feats), 512)

    def test_save_and_load_checkpoint(self):
        ckpt_path = os.path.join(self.temp_dir.name, "test_checkpoint.json")
        saved_path = self.classifier.save_checkpoint(ckpt_path, epoch=5, val_accuracy=98.5)
        self.assertTrue(os.path.exists(saved_path))
