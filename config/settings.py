import os
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field

    class Settings(BaseSettings):
        """
        Central Configuration Registry for MS-ADC Platform.
        Manages environment flags, model checkpoint paths, and cloud storage targets.
        """
        # Environment Profiles & Cloud Targets
        PROJECT_ID: str = Field(default="semicon-metrology-sandbox", env="PROJECT_ID")
        ENVIRONMENT: str = Field(default="PRODUCTION", env="ENVIRONMENT")
        REGION: str = Field(default="us-central1", env="REGION")
        BIGQUERY_DATASET: str = Field(default="semicon_metrology", env="BIGQUERY_DATASET")

        # Security & Guardrail Feature Flags
        DLP_SANITIZATION_ENABLED: bool = Field(default=True, env="DLP_SANITIZATION_ENABLED")
        PROMPT_GUARD_ENABLED: bool = Field(default=True, env="PROMPT_GUARD_ENABLED")
        CIRCUIT_BREAKER_ENABLED: bool = Field(default=True, env="CIRCUIT_BREAKER_ENABLED")
        WEBHOOKS_ENABLED: bool = Field(default=True, env="WEBHOOKS_ENABLED")
        VLM_TIMEOUT_SECONDS: float = Field(default=2.5, env="VLM_TIMEOUT_SECONDS")
        CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=3, env="CIRCUIT_BREAKER_FAILURE_THRESHOLD")
        CIRCUIT_BREAKER_RECOVERY_TIME_S: float = Field(default=10.0, env="CIRCUIT_BREAKER_RECOVERY_TIME_S")
        API_V1_PREFIX: str = Field(default="/v1", env="API_V1_PREFIX")
        AUTH_SECRET_KEY: str = Field(default="semicon-metrology-jwt-secret-key-prod-01", env="AUTH_SECRET_KEY")

        # Model Versioning & Storage Structure
        MODEL_VERSION: str = Field(default="v1.0.0", env="MODEL_VERSION")
        LOCAL_MODEL_DIR: str = Field(default="models", env="LOCAL_MODEL_DIR")
        GCS_MODEL_BUCKET: str = Field(default="semicon-metrology-few-shot-seeds", env="GCS_MODEL_BUCKET")

        # Concrete Local & GCS File Paths
        @property
        def local_checkpoint_path(self) -> str:
            return f"{self.LOCAL_MODEL_DIR}/die_vfm_head.pt"

        @property
        def local_trt_engine_path(self) -> str:
            return f"{self.LOCAL_MODEL_DIR}/die_vfm_fp16.engine"

        @property
        def versioned_local_dir(self) -> str:
            return f"{self.LOCAL_MODEL_DIR}/{self.MODEL_VERSION}"

        @property
        def versioned_local_checkpoint_path(self) -> str:
            return f"{self.LOCAL_MODEL_DIR}/{self.MODEL_VERSION}/die_vfm_head.pt"

        @property
        def gcs_versioned_path(self) -> str:
            """Resolves to: gs://semicon-metrology-few-shot-seeds/models/v1.0.0/"""
            return f"gs://{self.GCS_MODEL_BUCKET}/models/{self.MODEL_VERSION}/"

        @property
        def checkpoint_url(self) -> str:
            """Resolves to: gs://semicon-metrology-few-shot-seeds/models/v1.0.0/die_vfm_head.pt"""
            return f"gs://{self.GCS_MODEL_BUCKET}/models/{self.MODEL_VERSION}/die_vfm_head.pt"

        # Cleanroom Decision Thresholds
        defect_density_threshold: float = 0.05
        vfm_confidence_threshold: float = 0.90
        micro_batch_size: int = 16
        fmea_top_k: int = 2

        class Config:
            env_file = ".env"
            extra = "allow"

    settings = Settings()

except ImportError:
    class MockSettings:
        PROJECT_ID: str = os.getenv("PROJECT_ID", "semicon-metrology-sandbox")
        ENVIRONMENT: str = os.getenv("ENVIRONMENT", "PRODUCTION")
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

        # Model Versioning
        MODEL_VERSION: str = os.getenv("MODEL_VERSION", "v1.0.0")
        LOCAL_MODEL_DIR: str = os.getenv("LOCAL_MODEL_DIR", "models")
        GCS_MODEL_BUCKET: str = os.getenv("GCS_MODEL_BUCKET", "semicon-metrology-few-shot-seeds")

        @property
        def local_checkpoint_path(self) -> str:
            return f"{self.LOCAL_MODEL_DIR}/die_vfm_head.pt"

        @property
        def local_trt_engine_path(self) -> str:
            return f"{self.LOCAL_MODEL_DIR}/die_vfm_fp16.engine"

        @property
        def versioned_local_dir(self) -> str:
            return f"{self.LOCAL_MODEL_DIR}/{self.MODEL_VERSION}"

        @property
        def versioned_local_checkpoint_path(self) -> str:
            return f"{self.LOCAL_MODEL_DIR}/{self.MODEL_VERSION}/die_vfm_head.pt"

        @property
        def gcs_versioned_path(self) -> str:
            return f"gs://{self.GCS_MODEL_BUCKET}/models/{self.MODEL_VERSION}/"

        @property
        def checkpoint_url(self) -> str:
            return f"gs://{self.GCS_MODEL_BUCKET}/models/{self.MODEL_VERSION}/die_vfm_head.pt"

        defect_density_threshold: float = 0.05
        vfm_confidence_threshold: float = 0.90
        micro_batch_size: int = 16
        fmea_top_k: int = 2

    settings = MockSettings()
