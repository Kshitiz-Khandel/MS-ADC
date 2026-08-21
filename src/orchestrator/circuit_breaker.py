import time
from enum import Enum
from typing import Callable, Any, Tuple

class CircuitState(Enum):
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing, fast fallback
    HALF_OPEN = "HALF_OPEN"# Testing recovery

class CircuitBreaker:
    """
    Protects downstream systems from cascading failures and timeouts (Comp 21).
    Falls back to fast edge models if primary multimodal VLM SLA (>2.5s) is breached.
    """
    def __init__(self, failure_threshold: int = 3, recovery_time_s: float = 5.0):
        self.failure_threshold = failure_threshold
        self.recovery_time_s = recovery_time_s
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = CircuitState.CLOSED

    def execute(self, primary_fn: Callable[[], Any], fallback_fn: Callable[[], Any]) -> Tuple[Any, str]:
        current_time = time.time()

        if self.state == CircuitState.OPEN:
            if current_time - self.last_failure_time > self.recovery_time_s:
                self.state = CircuitState.HALF_OPEN
            else:
                return fallback_fn(), "CIRCUIT_OPEN_FALLBACK"

        try:
            result = primary_fn()
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            return result, "PRIMARY_SUCCESS"
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = current_time
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
            return fallback_fn(), f"FALLBACK_TRIGGERED: {str(e)}"
