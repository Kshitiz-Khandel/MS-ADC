MS-ADC/
├── .gitignore                           # Enterprise ignore rules
├── README.md                            # Architecture & Quickstart
├── pyproject.toml                       # Python dependencies & build config
├── cloudbuild.yaml                      # Comp 25: Automated CI/CD Pipeline
│
├── config/                              # Externalized Configuration (Comp 27, 30)
│   ├── prompts.yaml                     # Versioned Prompt Registry (Comp 27)
│   └── settings.py                      # Pydantic BaseSettings & Feature Flags (Comp 30)
│
├── data/                                # Data Assets & Playbooks
│   ├── fmea_corpus/                     # Comp 2: FMEA Markdown SOP Playbooks
│   │   ├── 300mm_rie_plasma_etch_fmea.md
│   │   ├── 300mm_photolithography_fmea.md
│   │   └── 300mm_cmp_planarization_fmea.md
│   └── lots/                            # Sample Lot Packages (LOT-882, LOT-883...)
│       └── LOT-882/
│           ├── metadata.json
│           ├── wafer_map.png
│           └── die_zoom.jpg
│
├── docs/                                # Scoping & Architectural Docs (Comp 6-12)
│   ├── scoping.md                       # Comp 6, 7, 8: Problem Def & Scope
│   ├── architecture/
│   │   ├── system_design.md             # Comp 9: Architecture & Sequence Flows
│   │   └── directory_structure.md       # Canonical Directory Contract
│   ├── adr/
│   │   └── 001_vfm_fewshot_adaptation.md # Comp 10: ADR on DINOv2 selection
│   ├── runbooks/
│   │   └── cleanroom_troubleshooting.md # Comp 12: Operational Runbooks
│   └── prr/
│       └── definition_of_done.md        # Comp 8: DoD Checklist & Metrics
│
├── src/                                 # Core Application Source Code
│   ├── gateway/                         # API Gateway Tier (Comp 11, 31, 32)
│   │   ├── __init__.py
│   │   ├── app.py                       # FastAPI Application Instance & Middleware
│   │   ├── routes_v1.py                 # Contract-First REST Endpoints (/v1/inspect)
│   │   ├── schemas.py                   # Pydantic v2 Request/Response Models
│   │   ├── auth.py                      # OAuth2 / IAM Verification (Comp 13)
│   │   └── webhooks.py                  # Outbound Fab Event Webhooks (Comp 32)
│   │
│   ├── orchestrator/                    # Multi-Agent Coordination (Comp 1, 21)
│   │   ├── __init__.py
│   │   ├── agent.py                     # LangGraph Coordinator & Tool-Calling Loop
│   │   ├── router.py                    # Autonomous Vision Triage Router
│   │   └── circuit_breaker.py           # Circuit Breaker & Fallback Guard (Comp 21)
│   │
│   ├── models/                          # Vision Inference Tier (Comp 1, 3, 22, 23, 29)
│   │   ├── __init__.py
│   │   ├── base.py                      # DefectClassifierInterface (Comp 29)
│   │   ├── wafer_vlm.py                 # Gemini 2.0 Wafer Specialist (Comp 1)
│   │   ├── die_vfm.py                   # NV-DINOv2 ViT Specialist (Comp 3)
│   │   ├── fine_tune_vfm.py             # Few-Shot Linear Head Trainer (Comp 3)
│   │   ├── export_tensorrt.py           # TensorRT ONNX Optimizer (Comp 23)
│   │   └── batch_processor.py           # Dynamic Batching Worker (Comp 22)
│   │
│   ├── rag/                             # RAG Knowledge Engine (Comp 2, 24)
│   │   ├── __init__.py
│   │   ├── indexer.py                   # Vertex AI Vector Search Ingestion
│   │   ├── fmea_retriever.py            # ANN Semantic Vector Search Retriever
│   │   └── cache.py                     # Prompt & Embedding Cache (Comp 24)
│   │
│   ├── ingestion/                       # Data Ingestion & Augmentation (Comp 2, 5)
│   │   ├── __init__.py
│   │   ├── dataset_loader.py            # WM-811K & PCB-DATASET Loader
│   │   └── augmentor.py                 # Minority Defect Class Augmentation (Comp 5)
│   │
│   ├── security/                        # Enterprise Security & Compliance (Comp 15, 16, 17)
│   │   ├── __init__.py
│   │   ├── dlp_sanitizer.py             # Cloud DLP Masking Middleware (Comp 15)
│   │   ├── prompt_guard.py              # Prompt Injection & Adversarial Defense (Comp 16)
│   │   └── audit_logger.py              # Tamper-Evident Cloud Logging (Comp 17)
│   │
│   ├── telemetry/                       # Observability (Comp 19)
│   │   ├── __init__.py
│   │   └── tracer.py                    # OpenTelemetry Cloud Trace Instrumentation
│   │
│   ├── evaluation/                      # LLM Ops & Benchmarking (Comp 4)
│   │   ├── __init__.py
│   │   ├── eval_pipeline.py             # LLM-as-a-Judge & Benchmark Suite
│   │   └── golden_dataset.json          # 20 Ground-Truth Inspection Cases
│   │
│   └── utils/                           # Domain Math & Metrics (Comp 5)
│       ├── __init__.py
│       └── metrics.py                   # Defect Density D0, Yield Loss, SLA Breaches
│
├── terraform/                           # 100% Codified Infrastructure as Code (Comp 14, 18, 26)
│   ├── versions.tf
│   ├── variables.tf
│   ├── main.tf
│   ├── vpc.tf                           # Comp 14: Private Subnet & Cloud NAT
│   ├── bigquery.tf                      # Comp 26: Partitioned Metrology Warehouse
│   ├── gcs.tf                           # Encrypted Object Storage Buckets
│   ├── iam.tf                           # Comp 13: Least Privilege Service Accounts
│   ├── secret_manager.tf                # Secret Manager
│   └── cloud_run.tf                     # Comp 18: Multi-Replica HA Cloud Run
│
└── tests/                               # Comprehensive Automated Test Suite (Comp 20, 28)
    ├── __init__.py
    ├── conftest.py                      # Global Fixtures & Mock Providers
    ├── unit/                            # Unit Tests (>80% Code Coverage)
    │   ├── test_infra_config.py         # Terraform & IaC Tests
    │   ├── test_models.py               # VLM & VFM Classifier Tests
    │   ├── test_rag_retriever.py        # Vector Search & Cache Tests
    │   ├── test_dlp_security.py         # DLP & Prompt Guard Tests
    │   └── test_metrics.py              # Yield & Defect Density Tests
    ├── integration/                     # E2E Multi-Agent Integration Tests
    │   └── test_inspection_pipeline.py  # Full Ingestion -> VLM -> RAG -> Output Flow
    └── chaos/                           # Resilience & Chaos Tests (Comp 20)
        └── test_failover.py             # Simulated Timeout & Circuit Breaker Tests



