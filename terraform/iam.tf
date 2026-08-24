resource "google_service_account" "gateway_sa" {
  account_id   = "${local.name_prefix}-gw-sa"
  display_name = "MS-ADC API Gateway Service Account"
  description  = "Service account for Cloud Run API Gateway with least privilege permissions"
}

resource "google_service_account" "agent_sa" {
  account_id   = "${local.name_prefix}-agent-sa"
  display_name = "MS-ADC Multi-Agent Orchestrator Service Account"
  description  = "Service account for LangGraph reasoning agent and model inference"
}

# Gateway IAM Bindings
resource "google_project_iam_member" "gateway_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.gateway_sa.email}"
}

resource "google_project_iam_member" "gateway_trace" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.gateway_sa.email}"
}

# Agent IAM Bindings (Least Privilege access to BigQuery, Vertex AI, Storage, DLP)
resource "google_project_iam_member" "agent_vertex_ai" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.agent_sa.email}"
}

resource "google_project_iam_member" "agent_bq_editor" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.agent_sa.email}"
}

resource "google_project_iam_member" "agent_storage_viewer" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.agent_sa.email}"
}

resource "google_project_iam_member" "agent_dlp_user" {
  project = var.project_id
  role    = "roles/dlp.user"
  member  = "serviceAccount:${google_service_account.agent_sa.email}"
}
