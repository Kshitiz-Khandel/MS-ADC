import os
import unittest
from pathlib import Path

TF_DIR = Path(__file__).parent.parent.parent / "terraform"

class TestInfraConfigSuite(unittest.TestCase):
    def test_terraform_files_exist(self):
        required_files = [
            "versions.tf", "variables.tf", "main.tf", "vpc.tf", 
            "bigquery.tf", "gcs.tf", "iam.tf", "secret_manager.tf", "cloud_run.tf"
        ]
        for filename in required_files:
            filepath = TF_DIR / filename
            self.assertTrue(filepath.exists(), f"Missing required Terraform file: {filename}")
            self.assertGreater(filepath.stat().st_size, 0, f"Terraform file {filename} is empty")

    def test_vpc_private_google_access_enabled(self):
        vpc_content = (TF_DIR / "vpc.tf").read_text()
        self.assertIn("private_ip_google_access = true", vpc_content, "VPC subnet must have Private Google Access enabled")
        self.assertIn("google_compute_router_nat", vpc_content, "VPC must have a Cloud NAT Gateway configured")

    def test_gcs_uniform_bucket_level_access(self):
        gcs_content = (TF_DIR / "gcs.tf").read_text()
        self.assertIn("uniform_bucket_level_access = true", gcs_content, "GCS buckets must enforce uniform bucket-level access")
        self.assertIn("versioning", gcs_content, "GCS buckets must have object versioning enabled")

    def test_bigquery_partitioning_and_clustering(self):
        bq_content = (TF_DIR / "bigquery.tf").read_text()
        self.assertIn('time_partitioning', bq_content, "BigQuery table must be partitioned by day")
        self.assertIn('clustering = ["lot_id", "tool_chamber", "macro_defect_class"]', bq_content, "BigQuery table must be clustered on query keys")

    def test_iam_least_privilege_roles(self):
        iam_content = (TF_DIR / "iam.tf").read_text()
        self.assertIn("gateway_sa", iam_content, "Must define dedicated Gateway Service Account")
        self.assertIn("agent_sa", iam_content, "Must define dedicated Agent Service Account")
        self.assertNotIn("roles/owner", iam_content, "Wildcard owner role forbidden")
        self.assertNotIn("roles/editor", iam_content, "Wildcard editor role forbidden")
        self.assertIn("roles/bigquery.dataEditor", iam_content)
        self.assertIn("roles/aiplatform.user", iam_content)
        self.assertIn("roles/storage.objectViewer", iam_content)
        self.assertIn("roles/dlp.user", iam_content)

    def test_cloud_run_health_probes(self):
        cr_content = (TF_DIR / "cloud_run.tf").read_text()
        self.assertIn("startup_probe", cr_content, "Cloud Run service must define startup health probe")
        self.assertIn("liveness_probe", cr_content, "Cloud Run service must define liveness health probe")
        self.assertIn("min_instance_count = 1", cr_content, "Cloud Run service must maintain min 1 instance for HA")

if __name__ == "__main__":
    unittest.main()
