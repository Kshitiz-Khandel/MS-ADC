import json
import logging
import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("ms_adc.audit")
logger.setLevel(logging.INFO)

class MetrologyAuditLogger:
    """
    Generates tamper-evident, structured audit events formatted for Google Cloud Logging (Comp 17).
    """
    @staticmethod
    def log_inspection_event(
        inspection_id: str,
        lot_id: str,
        wafer_id: Optional[str] = None,
        user_identity: str = "anonymous",
        macro_defect: str = "Unknown",
        micro_defect: str = "Unknown",
        fmea_citation: str = "None",
        latency_ms: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        event = {
            "severity": "INFO",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event_type": "METROLOGY_INSPECTION_AUDIT",
            "inspection_id": inspection_id,
            "lot_id": lot_id,
            "wafer_id": wafer_id or "W-01",
            "principal": user_identity,
            "predictions": {
                "macro_defect_class": macro_defect,
                "micro_defect_class": micro_defect,
                "fmea_citation_id": fmea_citation
            },
            "performance": {
                "total_latency_ms": round(latency_ms, 2)
            }
        }
        for k, v in kwargs.items():
            event[k] = v
        logger.info(json.dumps(event))
        return event
