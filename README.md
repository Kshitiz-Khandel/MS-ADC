# MS-ADC: Multimodal Semiconductor Defect Classification & Root-Cause Analysis Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Pydantic%20v2-green.svg)](https://fastapi.tiangolo.com/)
[![Google Cloud](https://img.shields.io/badge/Cloud-Google%20Cloud%20Vertex%20AI-4285F4.svg)](https://cloud.google.com/vertex-ai)
[![NVIDIA AI](https://img.shields.io/badge/Vision-NV--DINOv2%20%7C%20TensorRT-76B900.svg)](https://developer.nvidia.com/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

An enterprise-grade **Multi-Agent AI Metrology System** for advanced 300mm semiconductor cleanrooms. MS-ADC bridges ultra-fast edge vision inference with cloud multimodal reasoning to accelerate lot excursion root-cause diagnosis from **6 hours down to under 30 seconds**.

---

## 🏗️ System Architecture

```
                                 INCOMING LOT METROLOGY (LOT-882)
                                                 │
                                                 ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────┐
 │                      ENTERPRISE API GATEWAY & SECURITY PERIMETER                          │
 │  • FastAPI REST Endpoints (`/v1/inspect/wafer`, `/v1/inspect/die`)                        │
 │  • Cloud DLP Sanitization (Proprietary recipe & geometry tokenization)                    │
 │  • OAuth2 / OIDC Least Privilege IAM Role Verification                                    │
 └───────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                 │
                                                 ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────┐
 │                       LANGGRAPH CENTRAL COORDINATOR AGENT                                 │
 │  • Structured Tool Calling & Dynamic Vision Routing                                       │
 │  • Circuit Breaker & Failover Recovery Logic (<2.5s SLA timeout)                          │
 └───────────────────────┬───────────────────────────────────────────┬───────────────────────┘
                         │                                           │
                         ▼ (Macro Wafer Map)                         ▼ (Micro Die Micrograph)
 ┌───────────────────────────────────────┐   ┌───────────────────────────────────────────────┐
 │       WAFER VLM SPECIALIST            │   │            DIE VFM SPECIALIST                 │
 │     (Gemini 2.0 Multimodal)           │   │      (NV-DINOv2 ViT + TensorRT Engine)        │
 ├───────────────────────────────────────┤   ├───────────────────────────────────────────────┤
 │ • 9 WM-811K Spatial Defect Classes    │   │ • 6 Micro-Defect Classes (Short, Void, Open)  │
 │ • Concentric & Rim Clustering Logic   │   │ • Sub-50ms Edge Execution Latency             │
 │ • Zero-Hallucination Pydantic Schema  │   │ • Few-Shot Support Representation             │
 └───────────────────┬───────────────────┘   └───────────────────────┬───────────────────────┘
                     │                                               │
                     └───────────────────────┬───────────────────────┘
                                             │ (Aggregated Defect Signature)
                                             ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────┐
 │                          FMEA RAG KNOWLEDGE ENGINE                                        │
 │  • Vertex AI Vector Search (768-dim dense embeddings via `text-embedding-004`)            │
 │  • Prompt & Embedding In-Memory Caching (Comp 24)                                         │
 │  • Indexed SEMI-E10 Equipment Manuals (300mm Plasma Etch, Lithography, CMP)               │
 └───────────────────────────────────────────┬───────────────────────────────────────────────┘
                                             │
                                             ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────┐
 │                        METROLOGY AUDIT & ACTIONABLE DISPATCH                              │
 │  • BigQuery Yield Warehouse Table Logging (`semicon_metrology.lot_inspections`)           │
 │  • OpenTelemetry Distributed Spans Exported to Cloud Trace                                │
 │  • Real-Time Webhook Alert Callback Dispatched to Cleanroom Dashboard                     │
 └───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Assessed Competencies Mapping (32/32 Covered)

| Pillar | Competencies Demonstrated | Implementation Path |
| :--- | :--- | :--- |
| **1. AI/ML Engineering** | Multi-Agent Systems, RAG Pipelines, Model Tuning, LLM Ops, Domain KPIs | `src/orchestrator/`, `src/rag/`, `src/models/` |
| **2. Scoping & Design** | Problem Def, Scoping Boundaries, DoD, System Diagrams, ADRs, OpenAPI 3.0 | `docs/scoping.md`, `docs/adr/`, `docs/architecture/` |
| **3. Security & Privacy** | OAuth2/IAM, VPC Isolation, Cloud DLP, Prompt Injection Guard, Audit Logs | `src/security/`, `terraform/vpc.tf`, `terraform/iam.tf` |
| **4. Reliability & Trace** | High Availability, OpenTelemetry Distributed Tracing, Chaos Testing, Circuit Breakers | `src/telemetry/`, `src/orchestrator/circuit_breaker.py` |
| **5. Cost & Efficiency** | Dynamic Batching, TensorRT Compilation, Vertex AI Context & Embedding Caching | `src/models/export_tensorrt.py`, `src/rag/cache.py` |
| **6. Operations & CI/CD** | Cloud Build CI/CD, Terraform IaC (100%), Versioned Prompt Registries, Pytest | `cloudbuild.yaml`, `terraform/`, `config/prompts.yaml` |
| **7. Extensibility** | Base Interface Abstractions, Feature Flags, Contract-First API, Webhooks | `src/models/base.py`, `src/gateway/webhooks.py` |

---

## 🚀 Quickstart & Local Development

### 1. Clone & Environment Setup
```bash
git clone https://github.com/Kshitiz-Khandel/MS-ADC.git
cd MS-ADC

python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### 2. Run Data Ingestion & Few-Shot Setup
```bash
# Ingests WM-811K wafer maps and PCB micro-die datasets
python -m src.ingestion.dataset_loader --sample-size 50
```

### 3. Launch Local FastAPI Gateway
```bash
uvicorn src.gateway.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation will be available at `http://localhost:8000/docs`.

### 4. Run Test Suite & Evaluation
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 📄 License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
