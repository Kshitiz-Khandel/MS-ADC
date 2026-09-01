import datetime
from typing import Optional

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    class FastAPI:
        def __init__(self, title: str = "", description: str = "", version: str = "", docs_url: str = "", redoc_url: str = ""):
            self.title = title
            self.description = description
            self.version = version
            self.routes = {}

        def add_middleware(self, *args, **kwargs):
            pass

        def include_router(self, router):
            self.routes.update(getattr(router, "routes", {}))

        def get(self, path: str, response_model: any = None, tags: Optional[list] = None):
            def decorator(func):
                self.routes[("GET", path)] = func
                return func
            return decorator

    class CORSMiddleware:
        pass

from src.gateway.routes_v1 import router as v1_router
from src.gateway.schemas import HealthCheckResponse
from config.settings import settings

app = FastAPI(
    title="MS-ADC: Multimodal Semiconductor Defect Classification API",
    description="Enterprise Multi-Agent API for 300mm silicon cleanroom metrology, Gemini VLM spatial reasoning, and FMEA RAG retrieval.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router)

@app.get("/healthz", response_model=HealthCheckResponse, tags=["System Health"])
async def health_check():
    """
    Liveness and readiness probe for Cloud Run autoscaling health monitor.
    """
    return {
        "status": "HEALTHY",
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "version": "1.0.0"
    }
