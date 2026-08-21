# MS-ADC System Design & Architecture Specification

**Document Version:** 1.0.0  
**Author:** Aditya Sharma (Lead AI/ML Engineer)  
**Co-Author:** Kshitiz Khandelwal (Lead Architect)  
**Competencies Demonstrated:** Comp 9 (System Design Artifacts)  

---

## 1. High-Level Architecture Overview

MS-ADC is an enterprise multi-agent system designed for semiconductor automated defect classification (ADC) and root-cause analysis (RCA). It operates across two visual inspection tiers (Macro Wafer Maps and Micro Die Micrographs) coordinated by a central LangGraph reasoning agent and grounded in verified FMEA maintenance manuals via Vertex AI Vector Search.

```mermaid
flowchart TD
    subgraph ClientLayer ["Client & Ingestion Layer"]
        MES["Fab MES / Metrology Tool Track"]
        Engineer["Cleanroom Process Engineer Web UI"]
    end

    subgraph SecurityPerimeter ["Security & Gateway Perimeter (GCP Cloud Run)"]
        Gateway["FastAPI Gateway (/v1/inspect)"]
        DLP["Cloud DLP Redaction Middleware"]
        Auth["OAuth2 / IAM Role Guard"]
    end

    subgraph AgentOrchestration ["Agent Orchestration Tier (LangGraph)"]
        Coordinator["Coordinator Agent (Gemini 2.0 Multimodal)"]
        Router{"Vision Triage Router"}
        CB["Circuit Breaker & Fallback Guard"]
    end

    subgraph VisionSpecialists ["Vision Inference Tier"]
        WaferVLM["Wafer VLM Specialist\n(Gemini 2.0 / Pydantic Schema)"]
        DieVFM["Die VFM Specialist\n(NV-DINOv2 ViT + TensorRT <50ms)"]
    end

    subgraph KnowledgeTier ["RAG Knowledge & Storage Tier"]
        VectorDB[("Vertex AI Vector Search\n(SEMI-E10 FMEA Playbooks)")]
        BQ[("BigQuery Metrology Warehouse\n(semicon_metrology.lot_inspections)")]
        GCS[("Cloud Storage Buckets\n(Raw Wafers & Golden Templates)")]
        Trace["Cloud Trace (OpenTelemetry)"]
    end

    MES -->|HTTP Multi-Part| Gateway
    Engineer -->|HTTP Multi-Part| Gateway
    Gateway --> Auth --> DLP --> Coordinator

    Coordinator --> Router
    Router -->|Wafer Map| WaferVLM
    Router -->|Die Micrograph| DieVFM
    WaferVLM -->|Macro Defect JSON| Coordinator
    DieVFM -->|Micro Defect Code| Coordinator

    Coordinator -->|Synthesized Query| VectorDB
    VectorDB -->|Cited FMEA SOP Chunks| Coordinator

    Coordinator -->|Audit Log| BQ
    Coordinator -->|Telemetry Spans| Trace
    Coordinator -->|JSON Report| Gateway
```

---

## 2. Sequence Flow Specification

The interaction sequence below details the zero-hallucination execution loop for an inspection request:

```mermaid
sequenceDiagram
    autonumber
    actor Engineer as Cleanroom Engineer
    participant Gateway as FastAPI Gateway
    participant DLP as Cloud DLP Filter
    participant Agent as LangGraph Coordinator
    participant WaferVLM as Wafer VLM (Gemini 2.0)
    participant DieVFM as Die VFM (NV-DINOv2)
    participant RAG as Vertex AI Vector Search
    participant BQ as BigQuery Warehouse

    Engineer->>Gateway: POST /v1/inspect (lot_id="LOT-882", wafer_map.png, die.jpg)
    Gateway->>DLP: Inspect & Redact proprietary recipe codes
    DLP-->>Gateway: Sanitized Request Context
    Gateway->>Agent: Dispatch Inspection Payload
    par Parallel Vision Analysis
        Agent->>WaferVLM: Classify Macro Wafer Map (Pydantic Schema)
        WaferVLM-->>Agent: {"macro_defect": "Center", "confidence": 0.962}
        Agent->>DieVFM: Extract Embeddings & Classify Patch (<50ms)
        DieVFM-->>Agent: {"micro_defect": "Short", "confidence": 0.987}
    end
    Agent->>RAG: Vector Search ("Center Defect + Short in 300mm Etch Chamber 3")
    RAG-->>Agent: Top-1 Chunk (FMEA-SOP-ETCH-300-CH3 Sec 4.2)
    Agent->>BQ: Async Log Audit Record (lot_id, defects, fmea_id, latency)
    Agent-->>Gateway: Standardized InspectionResponse JSON
    Gateway-->>Engineer: Render Diagnostic Report with Actionable SOP Steps
```

---

## 3. Data Schema & Contracts

### A. Metrology Warehouse Schema (`BigQuery`)
```sql
CREATE TABLE `semicon_metrology.lot_inspections` (
    inspection_id STRING NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    lot_id STRING NOT NULL,
    wafer_id STRING NOT NULL,
    tool_chamber STRING NOT NULL,
    macro_defect_class STRING NOT NULL,
    macro_confidence FLOAT64 NOT NULL,
    micro_defect_class STRING,
    micro_confidence FLOAT64,
    fmea_citation_id STRING NOT NULL,
    severity STRING NOT NULL,
    total_latency_ms FLOAT64 NOT NULL
);
```

### B. REST API Interfaces
* `POST /v1/inspect/wafer`: Ingests single 2D wafer-bin maps for macro spatial classification.
* `POST /v1/inspect/die`: Ingests high-resolution optical/e-beam micrographs for $<50\text{ms}$ edge classification.
* `POST /v1/inspect`: Composite endpoint executing full hierarchical inspection and FMEA root-cause retrieval.
