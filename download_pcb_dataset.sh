#!/bin/bash
set -e

# ==============================================================================
# MS-ADC: Kaggle PCB Micro-Defect Dataset Downloader
# Downloads and unpacks the 6-class optical die dataset for VFM fine-tuning.
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/pcb_dataset"
ZIP_PATH="${PROJECT_ROOT}/data/pcb-defects.zip"

echo "========================================================================"
echo "📥 MS-ADC: Downloading Kaggle PCB Defect Dataset (akhatova/pcb-defects)"
echo "========================================================================"

# 1. Setup Kaggle Authentication securely from environment or existing config
mkdir -p ~/.kaggle

if [ -n "${KAGGLE_API_TOKEN}" ]; then
  echo "${KAGGLE_API_TOKEN}" > ~/.kaggle/access_token
  chmod 600 ~/.kaggle/access_token
  AUTH_HEADER="Authorization: Bearer ${KAGGLE_API_TOKEN}"
elif [ -n "${KAGGLE_USERNAME}" ] && [ -n "${KAGGLE_KEY}" ]; then
  cat <<EOF > ~/.kaggle/kaggle.json
{"username":"${KAGGLE_USERNAME}","key":"${KAGGLE_KEY}"}
EOF
  chmod 600 ~/.kaggle/kaggle.json
elif [ -f ~/.kaggle/kaggle.json ] || [ -f ~/.kaggle/access_token ]; then
  echo "Using existing Kaggle credentials from ~/.kaggle"
else
  echo "⚠️ Notice: No KAGGLE_API_TOKEN or KAGGLE_USERNAME/KAGGLE_KEY environment variables found."
  echo "Please set KAGGLE_API_TOKEN or configure ~/.kaggle/kaggle.json before downloading."
fi

# 2. Download Dataset Zip Archive if not already present
mkdir -p "${PROJECT_ROOT}/data"
if [ ! -d "${DATA_DIR}" ] || [ -z "$(ls -A "${DATA_DIR}" 2>/dev/null)" ]; then
  if [ -n "${AUTH_HEADER}" ]; then
    echo -e "\n[1/3] Downloading archive via Kaggle API curl..."
    curl -L -o "${ZIP_PATH}" \
      -H "${AUTH_HEADER}" \
      https://www.kaggle.com/api/v1/datasets/download/akhatova/pcb-defects
  elif command -v kaggle >/dev/null 2>&1; then
    echo -e "\n[1/3] Downloading archive via kaggle CLI..."
    kaggle datasets download -d akhatova/pcb-defects -p "${PROJECT_ROOT}/data"
  else
    echo "❌ Error: Kaggle CLI or KAGGLE_API_TOKEN required to download dataset."
    exit 1
  fi

  # 3. Extract Archive
  echo -e "\n[2/3] Unpacking ${ZIP_PATH} -> ${DATA_DIR}..."
  mkdir -p "${DATA_DIR}"
  unzip -q -o "${ZIP_PATH}" -d "${DATA_DIR}"

  # 4. Clean up zip
  rm -f "${ZIP_PATH}"
  echo -e "\n[3/3] Dataset download and extraction complete!"
else
  echo "✅ Dataset already present at ${DATA_DIR}, skipping download."
fi

echo "Files ready in: ${DATA_DIR}"
echo "========================================================================"
