from fastapi import APIRouter, Depends, HTTPException, status
from src.gateway.schemas import WaferInspectionRequest, DieInspectionRequest, CompositeInspectionRequest, InspectionResponse
from src.gateway.auth import verify_cleanroom_token
from src.orchestrator.agent import MetrologyCoordinatorAgent

router = APIRouter(prefix="/v1", tags=["Metrology Inspection v1"])
agent = MetrologyCoordinatorAgent()

@router.post("/inspect/wafer", response_model=InspectionResponse)
async def inspect_wafer(
    request: WaferInspectionRequest,
    principal: str = Depends(verify_cleanroom_token)
):
    """
    Inspects 300mm macro wafer-bin maps for spatial defect pattern classification (WM-811K).
    """
    try:
        return agent.process_inspection(request.model_dump(), user_identity=principal)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/inspect", response_model=InspectionResponse)
async def composite_inspection(
    request: CompositeInspectionRequest,
    principal: str = Depends(verify_cleanroom_token)
):
    """
    Full hierarchical multimodal inspection: Macro Wafer Map + Micro Die Patch + FMEA RAG Retrieval.
    """
    try:
        return agent.process_inspection(request.model_dump(), user_identity=principal)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
