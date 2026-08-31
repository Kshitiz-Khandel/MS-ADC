from fastapi import APIRouter, Depends, HTTPException, status
from src.gateway.schemas import (
    CompositeInspectionRequest,
    WaferInspectionRequest,
    DieInspectionRequest,
    InspectionResponse,
    WaferInspectionResponse,
    DieInspectionResponse
)
from src.gateway.auth import verify_cleanroom_token
from src.orchestrator.agent import MetrologyCoordinatorAgent

router = APIRouter(prefix="/v1", tags=["Metrology Inspection v1"])
agent = MetrologyCoordinatorAgent()

@router.post("/inspect", response_model=InspectionResponse)
async def composite_inspection(
    request: CompositeInspectionRequest,
    principal: str = Depends(verify_cleanroom_token)
):
    """
    Full hierarchical multimodal inspection: Macro Wafer Map + Micro Die Patch + Grounded FMEA RAG Retrieval.
    """
    try:
        return agent.process_inspection(request.model_dump(), user_identity=principal)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/inspect/wafer", response_model=WaferInspectionResponse)
async def inspect_wafer(
    request: WaferInspectionRequest,
    principal: str = Depends(verify_cleanroom_token)
):
    """
    Inspects 300mm macro wafer-bin maps for spatial defect pattern classification (WM-811K).
    Executes ONLY the Wafer Specialist without running unrelated die models.
    """
    try:
        return agent.inspect_wafer_only(request.model_dump(), user_identity=principal)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/inspect/die", response_model=DieInspectionResponse)
async def inspect_die(
    request: DieInspectionRequest,
    principal: str = Depends(verify_cleanroom_token)
):
    """
    Inspects high-magnification sub-micron SEM die micrographs for micro-defect classification (NV-DINOv2).
    Executes ONLY the Die Specialist without running unrelated wafer models.
    """
    try:
        return agent.inspect_die_only(request.model_dump(), user_identity=principal)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
