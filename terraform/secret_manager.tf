resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "${local.name_prefix}-gemini-api-key"

  replication {
    auto {}
  }

  labels = local.common_labels
}

resource "google_secret_manager_secret" "webhook_auth_token" {
  secret_id = "${local.name_prefix}-webhook-auth-token"

  replication {
    auto {}
  }

  labels = local.common_labels
}
