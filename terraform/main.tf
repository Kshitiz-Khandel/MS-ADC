provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

locals {
  name_prefix = "ms-adc-${var.environment}"
  common_labels = {
    project     = "ms-adc"
    environment = var.environment
    managed_by  = "terraform"
    domain      = "semiconductor-metrology"
  }
}
