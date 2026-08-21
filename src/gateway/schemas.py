from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class WaferInspectionRequest(BaseModel):
    lot_id: str = Field(..., example="LOT-882", description="Semiconductor manufacturing lot batch ID")
    wafer_id: str = Field(..., example="W-14", description="Wafer slot index inside lot")
    tool_chamber: str = Field(..., example="300mm_RIE_Etch_Chamber_3", description="Cleanroom process chamber")
    wafer_map_data: Optional[str] = Field(None, description="Base64 encoded wafer map or array representation")
    operator_notes: Optional[str] = Field(None, description="Cleanroom engineer observation notes")

class DieInspectionRequest(BaseModel):
    lot_id: str = Field(..., example="LOT-882")
    die_id: str = Field(..., example="D-128")
    patch_image_b64: str = Field(..., description="Base64 encoded micro optical die image")

class CompositeInspectionRequest(BaseModel):
    lot_id: str = Field(..., example="LOT-882")
    wafer_id: str = Field(..., example="W-14")
    tool_chamber: str = Field(..., example="300mm_RIE_Etch_Chamber_3")
    wafer_map_b64: Optional[str] = None
    die_patch_b64: Optional[str] = None
    operator_notes: Optional[str] = None

class FMEACitation(BaseModel):
    doc_id: str
    section_title: str
    tool_chamber: str
    similarity_score: float

class InspectionResponse(BaseModel):
    inspection_id: str
    timestamp: str
    lot_id: str
    wafer_id: str
    macro_defect: str
    macro_confidence: float
    micro_defect: Optional[str] = None
    micro_confidence: Optional[float] = None
    fmea_citations: List[FMEACitation] = []
    recommended_action: str
    execution_latency_ms: float
    circuit_breaker_status: str

class HealthCheckResponse(BaseModel):
    status: str
    environment: str
    timestamp: str
    version: str
