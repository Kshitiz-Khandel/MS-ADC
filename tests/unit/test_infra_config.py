import unittest
from pathlib import Path

class TestInfrastructureConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent.parent
        cls.terraform_dir = cls.repo_root / "terraform"

    def test_terraform_files_exist(self):
        expected_files = [
            "main.tf", "versions.tf", "variables.tf", "vpc.tf",
            "gcs.tf", "bigquery.tf", "iam.tf", "secret_manager.tf", "cloud_run.tf"
        ]
        for f in expected_files:
            file_path = self.terraform_dir / f
            self.assertTrue(file_path.exists(), f"Missing Terraform config file: {f}")

    def test_versions_tf_contains_required_providers(self):
        versions_file = self.terraform_dir / "versions.tf"
        content = versions_file.read_text()
        self.assertIn("google", content)
        self.assertIn("hashicorp/google", content)
        self.assertIn("~> 5.20.0", content)

    def test_vpc_configuration_isolated(self):
        vpc_file = self.terraform_dir / "vpc.tf"
        content = vpc_file.read_text()
        self.assertIn("google_compute_network", content)
        self.assertIn("metrology_vpc", content)
        self.assertIn("google_compute_router_nat", content)

    def test_bigquery_metrology_table_partitioning(self):
        bq_file = self.terraform_dir / "bigquery.tf"
        content = bq_file.read_text()
        self.assertIn("google_bigquery_dataset", content)
        self.assertIn("metrology_warehouse", content)
        self.assertIn("time_partitioning", content)
        self.assertIn("clustering", content)

    def test_iam_service_accounts_defined(self):
        iam_file = self.terraform_dir / "iam.tf"
        content = iam_file.read_text()
        self.assertIn("gateway_sa", content)
        self.assertIn("agent_sa", content)

    def test_cloud_run_service_configured(self):
        cloud_run_file = self.terraform_dir / "cloud_run.tf"
        content = cloud_run_file.read_text()
        self.assertIn("google_cloud_run_v2_service", content)
        self.assertIn("api_gateway", content)
        self.assertIn("startup_probe", content)

if __name__ == "__main__":
    unittest.main()
