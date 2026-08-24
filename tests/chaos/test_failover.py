import unittest
from src.orchestrator.circuit_breaker import CircuitBreaker, CircuitState
from src.orchestrator.agent import MetrologyCoordinatorAgent

class TestChaosAndFailoverSuite(unittest.TestCase):
    def setUp(self):
        self.cb = CircuitBreaker(failure_threshold=3, recovery_time_s=0.5)
        self.agent = MetrologyCoordinatorAgent()

    def test_simulated_cloud_outage_circuit_breaker_transition(self):
        def failing_cloud_call():
            raise TimeoutError("Simulated Cloud 504 Gateway Timeout")

        def fallback_edge_model():
            return {"macro_defect": "Center", "macro_confidence": 0.88, "fallback": True}

        # 1. First 2 failures: Circuit remains CLOSED, returns fallback
        res1, status1 = self.cb.execute(failing_cloud_call, fallback_edge_model)
        self.assertEqual(self.cb.state, CircuitState.CLOSED)
        self.assertEqual(status1, "FALLBACK_TRIGGERED: Simulated Cloud 504 Gateway Timeout")

        res2, status2 = self.cb.execute(failing_cloud_call, fallback_edge_model)
        self.assertEqual(self.cb.state, CircuitState.CLOSED)

        # 2. 3rd failure: Reaches failure threshold, trips OPEN
        res3, status3 = self.cb.execute(failing_cloud_call, fallback_edge_model)
        self.assertEqual(self.cb.state, CircuitState.OPEN)

        # 3. Subsequent calls immediately bypass cloud and execute edge fallback
        res4, status4 = self.cb.execute(failing_cloud_call, fallback_edge_model)
        self.assertEqual(status4, "CIRCUIT_OPEN_FALLBACK")
        self.assertTrue(res4["fallback"])

    def test_chaos_empty_images_fault_tolerance(self):
        payload = {
            "engineer_ticket": "Lot-999 missing images due to network socket drop.",
            "lot_info": {
                "lot_id": "LOT-CHAOS-999",
                "chamber": "300mm_RIE_Etch_Chamber_3",
                "images": []
            }
        }
        # Pipeline must not crash; falls back gracefully
        res = self.agent.process_inspection(payload, user_identity="chaos_monkey")
        self.assertEqual(res["circuit_breaker_status"], "PRIMARY_SUCCESS")
        self.assertIn("inspection_id", res)

if __name__ == "__main__":
    unittest.main()
