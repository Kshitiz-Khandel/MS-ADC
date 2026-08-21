resource "google_bigquery_dataset" "metrology_warehouse" {
  dataset_id                  = var.bigquery_dataset_id
  friendly_name               = "Semiconductor Metrology Yield & Defect Warehouse"
  description                 = "Stores audit trails, model predictions, defect metrics, and root-cause diagnoses"
  location                    = var.region
  delete_contents_on_destroy  = false

  labels = local.common_labels
}

resource "google_bigquery_table" "lot_inspections" {
  dataset_id = google_bigquery_dataset.metrology_warehouse.dataset_id
  table_id   = "lot_inspections"

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }

  clustering = ["lot_id", "tool_chamber", "macro_defect_class"]

  schema = <<EOF
[
  {
    "name": "inspection_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Unique inspection UUID"
  },
  {
    "name": "timestamp",
    "type": "TIMESTAMP",
    "mode": "REQUIRED",
    "description": "UTC timestamp of the inspection execution"
  },
  {
    "name": "lot_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Semiconductor manufacturing lot batch ID (e.g. LOT-882)"
  },
  {
    "name": "wafer_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Wafer index inside the lot (e.g. W-14)"
  },
  {
    "name": "tool_chamber",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Cleanroom process station (e.g. 300mm_RIE_Etch_Chamber_3)"
  },
  {
    "name": "macro_defect_class",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Wafer VLM predicted spatial cluster (e.g. Center, Donut, Scratch)"
  },
  {
    "name": "macro_confidence",
    "type": "FLOAT64",
    "mode": "REQUIRED",
    "description": "Confidence score of the macro wafer classification [0.0 - 1.0]"
  },
  {
    "name": "micro_defect_class",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "Die VFM predicted micro defect (e.g. Short, Open, Mouse-bite)"
  },
  {
    "name": "micro_confidence",
    "type": "FLOAT64",
    "mode": "NULLABLE",
    "description": "Confidence score of the micro die classification [0.0 - 1.0]"
  },
  {
    "name": "fmea_citation_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Referenced FMEA document and section citation (e.g. FMEA-SOP-ETCH-300-CH3#4.2)"
  },
  {
    "name": "severity",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Excursion severity code (LOW, MEDIUM, HIGH, CRITICAL_EXCURSION)"
  },
  {
    "name": "total_latency_ms",
    "type": "FLOAT64",
    "mode": "REQUIRED",
    "description": "Total end-to-end execution latency in milliseconds"
  }
]
EOF

  labels = local.common_labels
}
