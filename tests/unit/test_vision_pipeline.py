import unittest
from src.ingestion.wafer_loader import WM811KWaferLoader
from src.ingestion.micro_batcher import DynamicMicroBatcher
from src.models.die_vfm import DieVFMClassifier, DIE_DEFECT_CLASSES
from src.models.fine_tune_vfm import VFMFineTuner
from src.models.export_tensorrt import TensorRTExporter

class TestVisionAndIngestionSuite(unittest.TestCase):
    def setUp(self):
        self.wafer_loader = WM811KWaferLoader(target_resolution=(224, 224))
        self.batcher = DynamicMicroBatcher(max_batch_size=16, max_latency_ms=25.0)
        self.classifier = DieVFMClassifier()
        self.exporter = TensorRTExporter()

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

    def test_tensorrt_fp16_export_and_latency_sla(self):
        onnx_res = self.exporter.export_onnx("models/test.onnx")
        self.assertEqual(onnx_res["status"], "ONNX_EXPORT_SUCCESS")
        
        trt_res = self.exporter.build_tensorrt_engine("models/test.onnx", "models/test.engine")
        self.assertEqual(trt_res["status"], "TENSORRT_ENGINE_BUILT")
        self.assertLess(trt_res["benchmarks"]["tensorrt_fp16_latency_ms"], 50.0)
        self.assertGreater(trt_res["benchmarks"]["speedup_factor"], 3.0)

if __name__ == "__main__":
    unittest.main()
