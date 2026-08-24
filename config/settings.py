import os
from typing import Dict, Any

try:
    from pydantic_settings import BaseSettings
    class Settings(BaseSettings):
        PROJECT_ID: str = "semicon-metrology-sandbox"
        ENVIRONMENT: str = "sandbox"
        REGION: str = "us-central1"
        BIGQUERY_DATASET: str = "semicon_metrology"
        DLP_SANITIZATION_ENABLED: bool = True
        PROMPT_GUARD_ENABLED: bool = True
        CIRCUIT_BREAKER_ENABLED: bool = True
        WEBHOOKS_ENABLED: bool = True
        VLM_TIMEOUT_SECONDS: float = 2.5
        CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 3
        CIRCUIT_BREAKER_RECOVERY_TIME_S: float = 10.0
        API_V1_PREFIX: str = "/v1"
        AUTH_SECRET_KEY: str = "semicon-metrology-jwt-secret-key-prod-01"

        class Config:
            env_file = ".env"
            extra = "ignore"
    settings = Settings()
except ImportError:
    class MockSettings:
        PROJECT_ID: str = os.getenv("PROJECT_ID", "semicon-metrology-sandbox")
        ENVIRONMENT: str = os.getenv("ENVIRONMENT", "sandbox")
        REGION: str = os.getenv("REGION", "us-central1")
        BIGQUERY_DATASET: str = os.getenv("BIGQUERY_DATASET", "semicon_metrology")
        DLP_SANITIZATION_ENABLED: bool = True
        PROMPT_GUARD_ENABLED: bool = True
        CIRCUIT_BREAKER_ENABLED: bool = True
        WEBHOOKS_ENABLED: bool = True
        VLM_TIMEOUT_SECONDS: float = 2.5
        CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 3
        CIRCUIT_BREAKER_RECOVERY_TIME_S: float = 10.0
        API_V1_PREFIX: str = "/v1"
        AUTH_SECRET_KEY: str = "semicon-metrology-jwt-secret-key-prod-01"
    settings = MockSettings()
