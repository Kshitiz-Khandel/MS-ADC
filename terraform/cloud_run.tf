resource "google_cloud_run_v2_service" "api_gateway" {
  name     = "${local.name_prefix}-api"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"

  template {
    service_account = google_service_account.gateway_sa.email

    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/ms-adc-repo/api-gateway:latest"

      resources {
        limits = {
          cpu    = "2000m"
          memory = "4Gi"
        }
        cpu_idle = false
      }

      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }
      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "BIGQUERY_DATASET"
        value = google_bigquery_dataset.metrology_warehouse.dataset_id
      }

      startup_probe {
        http_get {
          path = "/healthz"
          port = 8000
        }
        initial_delay_seconds = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/healthz"
          port = 8000
        }
        period_seconds    = 15
        timeout_seconds   = 5
        failure_threshold = 3
      }
    }

    vpc_access {
      network_interfaces {
        network    = google_compute_network.metrology_vpc.id
        subnetwork = google_compute_subnetwork.metrology_private_subnet.id
      }
      egress = "ALL_TRAFFIC"
    }
  }

  labels = local.common_labels
}
