# ==============================================================================
# Google Cloud Vertex AI Vector Search (Matching Engine) for FMEA RAG
# ==============================================================================

resource "google_vertex_ai_index" "fmea_vector_index" {
  region       = var.region
  display_name = "${local.name_prefix}-fmea-vector-index"
  description  = "768-dim dense embedding index for SEMI-E10 FMEA troubleshooting playbooks"

  metadata {
    contents_delta_uri = "gs://${google_storage_bucket.raw_metrology_bucket.name}/vector_index_data/"
    config {
      dimensions                  = 768
      approximate_neighbors_count = 10
      distance_measure_type       = "COSINE_DISTANCE"
      algorithm_config {
        tree_ah_config {
          leaf_node_embedding_count    = 500
          leaf_nodes_to_search_percent = 10
        }
      }
    }
  }

  labels = local.common_labels
}

resource "google_vertex_ai_index_endpoint" "fmea_index_endpoint" {
  region       = var.region
  display_name = "${local.name_prefix}-fmea-index-endpoint"
  description  = "Private endpoint for real-time sub-millisecond FMEA vector retrieval"
  public_endpoint_enabled = false
  network      = google_compute_network.cleanroom_vpc.id

  labels = local.common_labels
}

resource "google_vertex_ai_index_endpoint_deployed_index" "fmea_deployed_index" {
  index_endpoint = google_vertex_ai_index_endpoint.fmea_index_endpoint.id
  index          = google_vertex_ai_index.fmea_vector_index.id
  deployed_index_id = "fmea_playbooks_deployed"
  display_name      = "fmea-playbooks-deployed-v1"

  automatic_resources {
    min_replica_count = 1
    max_replica_count = 3
  }
}
