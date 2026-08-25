#!/bin/bash
set -e

# ==============================================================================
# MS-ADC: Kaggle PCB Micro-Defect Dataset Downloader
# Downloads and unpacks the 6-class optical die proxy dataset for DINOv2 few-shot training.
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/pcb_dataset"
ZIP_PATH="${PROJECT_ROOT}/data/pcb-defects.zip"

echo "========================================================================"
echo "📥 MS-ADC: Downloading Kaggle PCB Defect Dataset (akhatova/pcb-defects)"
echo "========================================================================"

# 1. Setup Kaggle Authentication
mkdir -p ~/.kaggle
export KAGGLE_API_TOKEN="KGAT_490a7ce5512c4f0e92eb3b0c0726843a"
echo "${KAGGLE_API_TOKEN}" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token

# 2. Download Dataset Zip Archive
mkdir -p "${PROJECT_ROOT}/data"
echo "\n[1/3] Downloading archive via Kaggle API curl..."
curl -L -o "${ZIP_PATH}" \
  -H "Authorization: Bearer ${KAGGLE_API_TOKEN}" \
  https://www.kaggle.com/api/v1/datasets/download/akhatova/pcb-defects

# 3. Extract Archive
echo "\n[2/3] Unpacking ${ZIP_PATH} -> ${DATA_DIR}..."
mkdir -p "${DATA_DIR}"
unzip -q -o "${ZIP_PATH}" -d "${DATA_DIR}"

# 4. Clean up zip
rm -f "${ZIP_PATH}"

echo "\n[3/3] Dataset download and extraction complete!"
echo "Files ready in: ${DATA_DIR}"
echo "========================================================================"
