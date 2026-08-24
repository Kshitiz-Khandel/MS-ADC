# Cleanroom Operational Troubleshooting Runbook (MS-ADC)

**Audience:** Cleanroom On-Call SREs, Metrology Systems Engineers, Fab Technicians  
**Platform:** Multimodal Semiconductor Automated Defect Classification & Root-Cause Agent (MS-ADC)  
**SLA Target:** <3.0s total diagnosis latency | Zero production line stoppage

---

## 1. Incident Management Workflows

### Incident 1: Circuit Breaker Tripped (`OPEN` State)
* **Description:** Cloud Gemini 2.0 VLM endpoint experienced latency >2.5s or network partition. The circuit breaker tripped `OPEN` and automatically redirected all traffic to the local NVIDIA TensorRT Edge classifier.
* **Triage Steps:**
  1. Inspect Google Cloud Logging for HTTP 504 Gateway Timeouts:
     ```bash
     gcloud logging read "resource.type=cloud_run_revision AND jsonPayload.status=504" --limit=20
     ```
  2. Verify edge fallback model health on the local metrology node:
     ```bash
     curl -s http://localhost:8080/v1/health | jq .
     ```
  3. Once cloud latency normalizes (<500ms), manually reset the Circuit Breaker:
     ```bash
     curl -X POST http://localhost:8080/v1/admin/circuit-breaker/reset        -H "Authorization: Bearer $(gcloud auth print-identity-token)"
     ```

---

### Incident 2: GPU Out-Of-Memory (OOM) on TensorRT Inference Node
* **Description:** Burst cassette inspection (25 wafers / 50,000 dies) saturated GPU VRAM.
* **Triage Steps:**
  1. Check real-time GPU memory utilization:
     ```bash
     nvidia-smi --query-gpu=memory.used,memory.total --format=csv
     ```
  2. Throttle micro-batching ceiling in `config/settings.py`:
     ```python
     # Reduce batch size from 32 to 16 to fit within 8GB VRAM
     DYNAMIC_MICRO_BATCH_SIZE = 16
     ```
  3. Restart the container worker to flush CUDA memory caches.

---

## 2. Standard Operating Procedures (SOPs)

### SOP: Onboarding a New Defect Class (Few-Shot Adaptation)
When process engineers discover a novel defect signature (e.g., *"Slurry Scratch"*):
1. **Curate Image Exemplars:** Place 5–10 verified optical/SEM image crops in GCS:
   ```bash
   gsutil cp ./new_slurry_crops/*.png gs://semicon-metrology-raw-storage/few_shot_seeds/slurry_scratch/
   ```
2. **Execute Few-Shot Linear Head Adaptation:**
   ```bash
   python3 src/models/fine_tune_vfm.py --k-shot 10 --epochs 15 --new-class "slurry_scratch"
   ```
3. **Re-export TensorRT Engine:**
   ```bash
   python3 src/models/export_tensorrt.py --precision FP16 --output-engine models/die_vfm_fp16.engine
   ```
4. **Deploy to Metrology Node:** The edge container hot-reloads the new engine without downtime.
