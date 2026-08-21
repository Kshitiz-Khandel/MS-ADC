# Project Scoping Document: MS-ADC Semiconductor AI System

**Project Title:** Multimodal Semiconductor Defect Classification & Root-Cause Analysis Agent  
**Client / Domain:** Cleanroom 300mm Metrology & Yield Engineering  
**Lead Authors:** Kshitiz Khandelwal & Aditya Sharma  
**Reviewer:** Anshul Solanki  

---

## 1. Problem Definition (Comp 6)
Modern 300mm semiconductor fabs face two severe bottlenecks in automated defect classification (ADC):
1. **CNN Brittleness:** Traditional convolutional neural networks (CNNs) require thousands of labeled images per defect class, cannot generalize to out-of-distribution defects, and lack semantic reasoning capabilities.
2. **Excursion Diagnosis Latency:** When a yield excursion occurs, cleanroom process engineers spend an average of **6 hours** manually correlating spatial wafer maps, optical microscope images, and paper/PDF equipment manuals to identify the faulty chamber root cause.

## 2. Technical Scope & Boundaries (Comp 7)
### In Scope:
* **Hierarchical Multimodal Vision:** Integration of Gemini 2.0 Multimodal VLM for macro wafer-bin maps (9 WM-811K classes) and few-shot NV-DINOv2 VFM for micro-die optical defects (6 classes).
* **FMEA RAG Knowledge Engine:** Semantic indexing and sub-second retrieval across 300mm Etch, Lithography, and CMP equipment troubleshooting SOPs in Vertex AI Vector Search.
* **Enterprise Security & Observability:** Cloud DLP redaction of proprietary fab recipe IP, OAuth2 IAM authentication, and end-to-end OpenTelemetry distributed tracing to Cloud Trace.
* **100% Codified Infrastructure:** Complete Terraform HCL modules for VPC, Cloud Run, BigQuery, GCS, and Secret Manager.

### Out of Scope:
* Real-time physical hardware control, wafer robotic arm manipulation, or automated chemical stop-valve shutdown in live production fab tools.
* Sub-millisecond edge PLC (Programmable Logic Controller) hardware control loops.

## 3. Stakeholder Alignment & Definition of Done (Comp 8)
* **Die Classification Accuracy:** $\ge 98.0\%$ accuracy on the micro-die evaluation test dataset using few-shot NV-DINOv2 representations.
* **Wafer Reasoning Quality:** $\ge 95.0\%$ diagnostic accuracy and 100% schema-valid JSON output citing verified FMEA document chunks.
* **Inference Latency Target:** $<500\text{ms}$ for micro-die edge classification and $<3.0\text{s}$ for end-to-end VLM reasoning + RAG retrieval.
* **Test Code Coverage:** Maintain $\ge 80\%$ automated pytest code coverage across the application backend.
