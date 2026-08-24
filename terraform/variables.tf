variable "project_id" {
  description = "The Google Cloud Project ID"
  type        = string
  default     = "semicon-metrology-sandbox"
}

variable "region" {
  description = "The primary Google Cloud region for infrastructure deployment"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment (sandbox, staging, prod)"
  type        = string
  default     = "sandbox"
}

variable "vpc_subnet_cidr" {
  description = "CIDR range for the private metrology subnet"
  type        = string
  default     = "10.10.0.0/24"
}

variable "bigquery_dataset_id" {
  description = "Dataset ID for the BigQuery metrology warehouse"
  type        = string
  default     = "semicon_metrology"
}
