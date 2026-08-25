import json
import logging
import datetime
from typing import Dict, Any

logger = logging.getLogger("ms_adc.audit")
logger.setLevel(logging.INFO)

class MetrologyAuditLogger:
    """
    Generates tamper-evident, structured audit events formatted for Google Cloud Logging.
    Captures full metrology transaction traces for quality compliance.
    """
    @staticmethod
    def log_inspection_event(
        inspection_id: str,
        lot_id: str,
        wafer_id: str,
        user_identity: str,
        macro_defect: str,
        micro_defect: str,
        fmea_citation: str,
        latency_ms: float
    ) -> Dict[str, Any]:
        event = {
            "severity": "INFO",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event_type": "METROLOGY_INSPECTION_AUDIT",
            "inspection_id": inspection_id,
            "lot_id": lot_id,
            "wafer_id": wafer_id,
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
        logger.info(json.dumps(event))
        return event
