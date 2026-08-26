import unittest
from src.telemetry.tracer import OpenTelemetryTracer
from src.evaluation.eval_pipeline import MetrologyEvalPipeline

class TestTelemetryAndEvaluationSuite(unittest.TestCase):
    def setUp(self):
        self.tracer = OpenTelemetryTracer(service_name="ms-adc-test-service")
        self.eval_pipeline = MetrologyEvalPipeline()

    def test_opentelemetry_tracing_spans_and_export(self):
        trace_id = self.tracer.create_trace_id()
        self.assertTrue(trace_id.startswith("trace-"))

        with self.tracer.start_span("gateway_ingress", trace_id=trace_id) as span1:
            span1.set_attribute("lot_id", "LOT-123")
            span1.add_event("dlp_sanitization_complete")

        with self.tracer.start_span("vlm_inference", trace_id=trace_id) as span2:
            span2.set_attribute("model", "gemini-2.0-pro")

        export_res = self.tracer.export_spans_to_cloud_trace(trace_id)
        self.assertEqual(export_res["status"], "EXPORTED_TO_CLOUD_TRACE")
        self.assertEqual(export_res["span_count"], 2)
        self.assertGreater(export_res["total_latency_ms"], 0.0)

    def test_evaluation_benchmark_pipeline_execution(self):
        report = self.eval_pipeline.run_benchmark()
        self.assertEqual(report["benchmark_status"], "PASSED")
        self.assertTrue(report["quality_gates_passed"])
        
        # Verify Vision Metrics
        self.assertGreaterEqual(report["vision_metrics"]["die_specialist_accuracy_pct"], 98.0)
        self.assertGreaterEqual(report["vision_metrics"]["wafer_specialist_accuracy_pct"], 95.0)
        
        # Verify IR Metrics
        self.assertGreaterEqual(report["retrieval_ir_metrics"]["recall_at_2_pct"], 95.0)
        self.assertGreaterEqual(report["retrieval_ir_metrics"]["ndcg_at_2_pct"], 90.0)
        
        # Verify Generator Faithfulness
        self.assertGreaterEqual(report["generator_metrics"]["faithfulness_grounding_score_pct"], 95.0)

if __name__ == "__main__":
    unittest.main()
