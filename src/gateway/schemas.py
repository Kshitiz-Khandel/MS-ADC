import os
from enum import Enum
from typing import List, Optional, Dict, Any

class ChamberType(str, Enum):
    RIE_ETCH = "300mm_RIE_Etch_Chamber_3"
    LITHO_SCANNER = "300mm_Immersion_Litho_Track_2"
    CMP_PLATEN = "300mm_CMP_Platen_1"
    UNKNOWN = "unknown_chamber"

class MacroDefectType(str, Enum):
    CENTER = "Center"
    DONUT = "Donut"
    EDGE_RING = "Edge-Ring"
    EDGE_LOC = "Edge-Loc"
    SCRATCH = "Scratch"
    RANDOM = "Random"
    UNKNOWN = "Unknown"

class MicroDefectType(str, Enum):
    SHORT = "Short"
    OPEN = "Open_circuit"
    PARTICLE = "Particle"
    VOID = "Void"
    LINE_COLLAPSE = "Line_collapse"
    SPURIOUS_COPPER = "Spurious_copper"
    UNKNOWN = "Unknown"

try:
    from pydantic import BaseModel, Field, ValidationError

    class LotInfo(BaseModel):
        lot_id: str = Field(..., example="LOT-123")
        chamber: str = Field(..., example="300mm_RIE_Etch_Chamber_3")
        recipe_id: Optional[str] = Field(None, example="RECIPE-OXIDE-ETCH-994")
        images: List[str] = Field(
            default_factory=list,
            description="List of GCS image URIs or base64 strings containing wafer maps and SEM die micrographs."
        )

    class InspectionRequest(BaseModel):
        engineer_ticket: str = Field(
            ...,
            example="Lot-123 failed metal-1 resistance test after Etch Chamber 3. Investigate if this is a tool-level chamber excursion or isolated particle defects."
        )
        lot_info: LotInfo
        metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class InspectionResponse(BaseModel):
        inspection_id: str
        timestamp: str
        lot_id: str
        chamber: str
        macro_defect: str
        macro_confidence: float
        micro_defect: str
        micro_confidence: float
        fmea_citations: List[Dict[str, Any]]
        recommended_action: str
        tool_call_trace: List[Dict[str, Any]]
        execution_latency_ms: float
        circuit_breaker_status: str
        agent_framework: str = "Google_Agent_Development_Kit_2.0"

except ImportError:
    class ValidationError(Exception):
        pass

    class LotInfo:
        def __init__(self, lot_id: str, chamber: str, recipe_id: Optional[str] = None, images: Optional[List[str]] = None):
            if not lot_id or not chamber:
                raise ValidationError("lot_id and chamber are required in LotInfo")
            self.lot_id = lot_id
            self.chamber = chamber
            self.recipe_id = recipe_id
            self.images = images or []

    class InspectionRequest:
        def __init__(self, engineer_ticket: str, lot_info: Any, metadata: Optional[Dict[str, Any]] = None):
            if not engineer_ticket or not lot_info:
                raise ValidationError("engineer_ticket and lot_info are required in InspectionRequest")
            if isinstance(lot_info, dict):
                lot_info = LotInfo(**lot_info)
            self.engineer_ticket = engineer_ticket
            self.lot_info = lot_info
            self.metadata = metadata or {}

    class InspectionResponse:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
