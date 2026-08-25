import os
import unittest
from pathlib import Path
from src.ingestion.wafer_loader import WM811KWaferLoader
from src.ingestion.micro_batcher import DynamicMicroBatcher
from src.ingestion.augmentor import CleanroomDataAugmentor
from src.ingestion.dataset_loader import PCBDefectDatasetLoader
from src.models.die_vfm import DieVFMClassifier, DIE_DEFECT_CLASSES
from src.models.base import DefectClassifierInterface
from src.models.fine_tune_vfm import VFMFineTuner
from src.models.export_tensorrt import TensorRTExporter

class TestVisionAndIngestionSuite(unittest.TestCase):
    def setUp(self):
        self.wafer_loader = WM811KWaferLoader(target_resolution=(224, 224))
        self.batcher = DynamicMicroBatcher(max_batch_size=16, max_latency_ms=25.0)
        self.augmentor = CleanroomDataAugmentor()
        self.classifier = DieVFMClassifier()
        self.exporter = TensorRTExporter()

    def test_classifier_implements_interface(self):
        self.assertTrue(issubclass(DieVFMClassifier, DefectClassifierInterface))
        self.assertIsInstance(self.classifier, DefectClassifierInterface)

    def test_synthetic_wafer_map_generation_and_kpis(self):
        center_matrix = self.wafer_loader.generate_synthetic_wafer(pattern="Center", size=50)
        kpis = self.wafer_loader.compute_kpis(center_matrix)
        self.assertGreater(kpis["total_dies"], 0)
        self.assertGreater(kpis["defective_dies"], 0)
        self.assertGreaterEqual(kpis["defect_density_d0"], 0.05)
        self.assertLessEqual(kpis["die_yield_pct"], 95.0)

    def test_wafer_rendering_and_base64_packaging(self):
        matrix = self.wafer_loader.generate_synthetic_wafer(pattern="Edge-Ring", size=50)
        res = self.wafer_loader.process_wafer(matrix, lot_id="LOT-TEST-99", wafer_index=1, failure_type="Edge-Ring")
        self.assertEqual(res["lot_id"], "LOT-TEST-99")
        self.assertEqual(res["failure_type"], "Edge-Ring")
        self.assertTrue(len(res["image_base64"]) > 10)
        self.assertEqual(res["resolution"], [224, 224])

    def test_cleanroom_data_augmentor_matrix_and_features(self):
        matrix = self.wafer_loader.generate_synthetic_wafer(pattern="Scratch", size=30)
        flipped = self.augmentor.augment_matrix(matrix, flip_horizontal=True, flip_vertical=True)
        self.assertEqual(len(flipped), 30)
        self.assertEqual(len(flipped[0]), 30)

        clustered = self.augmentor.inject_defect_cluster(matrix, defect_code=2, cluster_radius=2)
        self.assertEqual(len(clustered), 30)

        feat = [0.1] * 768
        jittered = self.augmentor.augment_feature_vector(feat)
        self.assertEqual(len(jittered), 768)

    def test_dynamic_micro_batching_stream(self):
        mock_stream = [{"die_id": f"die_{i}", "lot_id": "LOT-1"} for i in range(45)]
        batches = list(self.batcher.batch_stream(mock_stream))
        
        self.assertEqual(len(batches), 3)
        self.assertEqual(len(batches[0]), 16)
        self.assertEqual(len(batches[1]), 16)
        self.assertEqual(len(batches[2]), 13)

    def test_die_vfm_few_shot_fine_tuning_convergence(self):
        trainer = VFMFineTuner(self.classifier, learning_rate=0.05)
        res = trainer.run_training(k_shot=10, epochs=15)
        
        self.assertEqual(res["k_shot"], 10)
        self.assertEqual(res["total_samples"], 60)
        self.assertLess(res["final_loss"], res["loss_history"][0])
        self.assertGreaterEqual(res["accuracy_pct"], 90.0)

    def test_checkpoint_save_and_load(self):
        ckpt_path = "models/test_checkpoint.pt"
        saved_path = self.classifier.save_checkpoint(ckpt_path, epoch=5, val_accuracy=98.5)
        self.assertTrue(os.path.exists(saved_path))

        new_classifier = DieVFMClassifier()
        metadata = new_classifier.load_checkpoint(saved_path)
        self.assertIsNotNone(metadata)
        if os.path.exists(saved_path):
            os.remove(saved_path)

    def test_dataset_loader_stratified_split(self):
        loader = PCBDefectDatasetLoader()
        train_s, val_s, test_s = loader.get_stratified_split(k_shot_train=5, val_ratio=0.2)
        self.assertIn("missing_hole", train_s)
        self.assertIn("spurious_copper", train_s)

    def test_tensorrt_fp16_export_and_latency_sla(self):
        onnx_res = self.exporter.export_onnx("models/test.onnx")
        self.assertEqual(onnx_res["status"], "ONNX_EXPORT_SUCCESS")
        
        trt_res = self.exporter.build_tensorrt_engine("models/test.onnx", "models/test.engine")
        self.assertEqual(trt_res["status"], "TENSORRT_ENGINE_BUILT")
        self.assertLess(trt_res["benchmarks"]["tensorrt_fp16_latency_ms"], 50.0)
        self.assertGreater(trt_res["benchmarks"]["speedup_factor"], 3.0)

if __name__ == "__main__":
    unittest.main()
