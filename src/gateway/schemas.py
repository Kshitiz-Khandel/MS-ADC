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

class ValidationError(ValueError):
    pass

try:
    from pydantic import BaseModel, Field, field_validator

    class LotInfo(BaseModel):
        lot_id: str = Field(..., description="Manufacturing lot ID")
        chamber: str = Field(..., description="Tool chamber ID")
        recipe_id: Optional[str] = Field(None, description="Chemical recipe ID")
        images: List[str] = Field(
            default_factory=list,
            description="List of GCS image URIs (gs://), local paths, or base64 strings containing wafer maps and SEM die micrographs."
        )

        @field_validator("lot_id", "chamber")
        @classmethod
        def check_non_empty(cls, v):
            if v is None or not str(v).strip():
                raise ValidationError("Field cannot be empty or None")
            return v

    class CompositeInspectionRequest(BaseModel):
        engineer_ticket: str = Field(..., description="Engineering problem description")
        lot_info: LotInfo
        metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

        @field_validator("engineer_ticket")
        @classmethod
        def check_ticket(cls, v):
            if v is None or not str(v).strip():
                raise ValidationError("engineer_ticket cannot be empty or None")
            return v

    class WaferInspectionRequest(BaseModel):
        lot_id: str = Field(..., example="LOT-WAFER-001")
        chamber: str = Field(..., example="300mm_RIE_Etch_Chamber_3")
        image_uri: str = Field(..., example="gs://semicon-raw/LOT-WAFER-001/wafer_map.png")
        engineer_ticket: Optional[str] = Field(default="Routine 300mm macro wafer map inspection.")

    class DieInspectionRequest(BaseModel):
        lot_id: str = Field(..., example="LOT-DIE-001")
        chamber: str = Field(..., example="300mm_RIE_Etch_Chamber_3")
        image_uri: str = Field(..., example="gs://semicon-raw/LOT-DIE-001/die_sem_01.png")
        engineer_ticket: Optional[str] = Field(default="Sub-micron SEM die defect inspection.")

    # Compatibility alias
    InspectionRequest = CompositeInspectionRequest

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
    class LotInfo:
        def __init__(self, lot_id: str, chamber: str, recipe_id: Optional[str] = None, images: Optional[List[str]] = None):
            if not lot_id or not chamber:
                raise ValidationError("lot_id and chamber are required in LotInfo")
            self.lot_id = lot_id
            self.chamber = chamber
            self.recipe_id = recipe_id
            self.images = images or []

    class CompositeInspectionRequest:
        def __init__(self, engineer_ticket: str, lot_info: Any, metadata: Optional[Dict[str, Any]] = None):
            if not engineer_ticket:
                raise ValidationError("engineer_ticket cannot be empty")
            if isinstance(lot_info, dict):
                lot_info = LotInfo(**lot_info)
            elif not lot_info:
                raise ValidationError("lot_info is required")
            self.engineer_ticket = engineer_ticket
            self.lot_info = lot_info
            self.metadata = metadata or {}

    class WaferInspectionRequest:
        def __init__(self, lot_id: str, chamber: str, image_uri: str, engineer_ticket: str = ""):
            self.lot_id = lot_id
            self.chamber = chamber
            self.image_uri = image_uri
            self.engineer_ticket = engineer_ticket

    class DieInspectionRequest:
        def __init__(self, lot_id: str, chamber: str, image_uri: str, engineer_ticket: str = ""):
            self.lot_id = lot_id
            self.chamber = chamber
            self.image_uri = image_uri
            self.engineer_ticket = engineer_ticket

    InspectionRequest = CompositeInspectionRequest

    class InspectionResponse:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
