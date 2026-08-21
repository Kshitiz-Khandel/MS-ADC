import os
import re
from pathlib import Path
import pytest

TF_DIR = Path(__file__).parent.parent.parent / "terraform"

def test_terraform_files_exist():
    required_files = [
        "versions.tf", "variables.tf", "main.tf", "vpc.tf", 
        "bigquery.tf", "gcs.tf", "iam.tf", "secret_manager.tf", "cloud_run.tf"
    ]
    for filename in required_files:
        filepath = TF_DIR / filename
        assert filepath.exists(), f"Missing required Terraform file: {filename}"
        assert filepath.stat().st_size > 0, f"Terraform file {filename} is empty"

def test_vpc_private_google_access_enabled():
    vpc_content = (TF_DIR / "vpc.tf").read_text()
    assert "private_ip_google_access = true" in vpc_content, "VPC subnet must have Private Google Access enabled"
    assert "google_compute_router_nat" in vpc_content, "VPC must have a Cloud NAT Gateway configured"

def test_gcs_uniform_bucket_level_access():
    gcs_content = (TF_DIR / "gcs.tf").read_text()
    assert "uniform_bucket_level_access = true" in gcs_content, "GCS buckets must enforce uniform bucket-level access"
    assert "versioning" in gcs_content, "GCS buckets must have object versioning enabled"

def test_bigquery_partitioning_and_clustering():
    bq_content = (TF_DIR / "bigquery.tf").read_text()
    assert 'time_partitioning' in bq_content, "BigQuery table must be partitioned by day"
    assert 'clustering = ["lot_id", "tool_chamber", "macro_defect_class"]' in bq_content, "BigQuery table must be clustered on query keys"

def test_iam_least_privilege_roles():
    iam_content = (TF_DIR / "iam.tf").read_text()
    # Check dedicated service accounts
    assert "gateway_sa" in iam_content, "Must define dedicated Gateway Service Account"
    assert "agent_sa" in iam_content, "Must define dedicated Agent Service Account"
    # Ensure no admin / wildcard roles
    assert "roles/owner" not in iam_content, "Wildcard owner role forbidden"
    assert "roles/editor" not in iam_content, "Wildcard editor role forbidden"
    # Verify specific least privilege roles
    assert "roles/bigquery.dataEditor" in iam_content
    assert "roles/aiplatform.user" in iam_content
    assert "roles/storage.objectViewer" in iam_content
    assert "roles/dlp.user" in iam_content

def test_cloud_run_health_probes():
    cr_content = (TF_DIR / "cloud_run.tf").read_text()
    assert "startup_probe" in cr_content, "Cloud Run service must define startup health probe"
    assert "liveness_probe" in cr_content, "Cloud Run service must define liveness health probe"
    assert "min_instance_count = 1" in cr_content, "Cloud Run service must maintain min 1 instance for HA"
