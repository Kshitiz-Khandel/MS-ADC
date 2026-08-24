resource "google_compute_network" "metrology_vpc" {
  name                    = "${local.name_prefix}-vpc"
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"
  description             = "Isolated VPC network for semiconductor metrology inference and data processing"
}

resource "google_compute_subnetwork" "metrology_private_subnet" {
  name                     = "${local.name_prefix}-private-subnet"
  ip_cidr_range            = var.vpc_subnet_cidr
  region                   = var.region
  network                  = google_compute_network.metrology_vpc.id
  private_ip_google_access = true

  log_config {
    aggregation_interval = "INTERVAL_5_SEC"
    flow_sampling        = 0.5
    metadata             = "INCLUDE_ALL_METADATA"
  }
}

resource "google_compute_router" "nat_router" {
  name    = "${local.name_prefix}-nat-router"
  network = google_compute_network.metrology_vpc.name
  region  = var.region
}

resource "google_compute_router_nat" "nat_gateway" {
  name                               = "${local.name_prefix}-nat-gw"
  router                             = google_compute_router.nat_router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}
