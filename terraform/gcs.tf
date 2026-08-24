resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "google_storage_bucket" "wafer_raw_images" {
  name                        = "${local.name_prefix}-wafer-raw-${random_id.bucket_suffix.hex}"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  labels = local.common_labels
}

resource "google_storage_bucket" "fmea_knowledge_corpus" {
  name                        = "${local.name_prefix}-fmea-corpus-${random_id.bucket_suffix.hex}"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  labels = local.common_labels
}
