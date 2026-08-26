import time
import uuid
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

class MetrologySpan:
    """Represents a single distributed execution span within an inspection trace."""
    def __init__(self, name: str, trace_id: str, parent_id: Optional[str] = None):
        self.name = name
        self.trace_id = trace_id
        self.span_id = f"span-{uuid.uuid4().hex[:8]}"
        self.parent_id = parent_id
        self.start_time = time.time()
        self.end_time = 0.0
        self.duration_ms = 0.0
        self.attributes = {}
        self.events = []
        self.status = "UNSET"

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })

    def finish(self, status: str = "OK"):
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000.0, 3)
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events
        }

class OpenTelemetryTracer:
    """
    Distributed tracing coordinator instrumenting spans across Gateway, DLP Sanitizer,
    Wafer VLM, Die VFM, and Vertex AI Vector Search. Exports trace spans to Cloud Trace.
    """
    def __init__(self, service_name: str = "ms-adc-metrology-gateway"):
        self.service_name = service_name
        self.active_spans: List[MetrologySpan] = []

    def create_trace_id(self) -> str:
        return f"trace-{uuid.uuid4().hex[:16]}"

    @contextmanager
    def start_span(self, name: str, trace_id: str, parent_id: Optional[str] = None):
        span = MetrologySpan(name=name, trace_id=trace_id, parent_id=parent_id)
        span.set_attribute("service.name", self.service_name)
        try:
            yield span
            span.finish(status="OK")
        except Exception as e:
            span.finish(status=f"ERROR: {str(e)}")
            raise e
        finally:
            self.active_spans.append(span)

    def export_spans_to_cloud_trace(self, trace_id: str) -> Dict[str, Any]:
        """Simulates exporting collected spans to Google Cloud Trace API."""
        spans_for_trace = [s.to_dict() for s in self.active_spans if s.trace_id == trace_id]
        total_latency = sum(s["duration_ms"] for s in spans_for_trace)
        
        return {
            "status": "EXPORTED_TO_CLOUD_TRACE",
            "trace_id": trace_id,
            "span_count": len(spans_for_trace),
            "total_latency_ms": round(total_latency, 3),
            "spans": spans_for_trace
        }
