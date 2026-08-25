                                OUR COLLABORATIVE PR ROADMAP
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 🚀 STEP 1: PR #1 — Infrastructure as Code (IaC) & Cloud Landing Zone                   │
│    • Branch: feat/infra-iac (Author: Kshitiz)                                        │
│    • Files: terraform/main.tf, vpc.tf, bigquery.tf, gcs.tf, iam.tf           │
│    • Competencies: Comp 13 (IAM), Comp 14 (VPC Security), Comp 26 (Terraform IaC)     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 📦 STEP 2: PR #2 — Data Ingestion & Few-Shot NV-DINOv2 VFM                             │
│    • Branch: feat/data-and-vfm (Author: Aditya)                                      │
│    • Files: dataset_loader.py, augmentor.py, fine_tune_vfm.py, export_tensorrt │
│    • Competencies: Comp 2 (Data Eng), Comp 3 (Model Tuning), Comp 23 (TensorRT)       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 📚 STEP 3: PR #3 — FMEA RAG Knowledge Engine                                           │
│    • Branch: feat/rag-knowledge-engine (Author: Kshitiz)                             │
│    • Files: data/fmea_corpus/*.md, indexer.py, fmea_retriever.py, cache.py     │
│    • Competencies: Comp 2 (Vector Search RAG), Comp 24 (Context Caching)               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 🤖 STEP 4: PR #4 — Multi-Agent Coordinator, Gateway & Security Guardrails              │
│    • Branch: feat/agent-gateway-security (Author: Kshitiz & Aditya)                  │
│    • Files: wafer_vlm.py, agent.py, circuit_breaker.py, dlp_sanitizer.py, UI   │
│    • Competencies: Comp 1 (Multi-Agent), Comp 11, 15, 16, 17, 21, 27, 29, 31, 32      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 📈 STEP 5: PR #5 — OpenTelemetry Observability, CI/CD & Automated Evaluation           │
│    • Branch: feat/telemetry-ci-eval (Author: Aditya & Kshitiz)                       │
│    • Files: tracer.py, cloudbuild.yaml, eval_pipeline.py, tests/               │
│    • Competencies: Comp 4 (Eval), Comp 12, 18, 19, 20, 25, 28, 30                     │
└────────────────────────────────────────────────────────────────────────────────────────┘



🤝 Feature Branch & PR Assignment Matrix

PR #
Feature Branch
Primary Author
Reviewer
Scope & Files Covered
PR #1
feat/infra-iac
Kshitiz
Aditya
terraform/ (VPC, BQ, GCS, IAM, Cloud Run), tests/unit/test_infra_config.py
PR #2
feat/data-and-vfm
Aditya
Kshitiz
src/ingestion/, src/models/fine_tune_vfm.py, die_vfm.py, export_tensorrt.py, tests/unit/test_models.py
PR #3
feat/rag-knowledge-engine
Kshitiz
Aditya
data/fmea_corpus/, src/rag/, tests/unit/test_rag_retriever.py
PR #4
feat/agent-gateway-security
Kshitiz & Aditya
Both
src/models/wafer_vlm.py, src/gateway/, src/orchestrator/, src/security/, tests/unit/test_dlp_security.py
PR #5
feat/telemetry-ci-eval
Aditya & Kshitiz
Both
src/telemetry/, cloudbuild.yaml, src/evaluation/, tests/chaos/, docs/runbooks/