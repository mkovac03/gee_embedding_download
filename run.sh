#!/bin/bash

# Activate your environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gee_data_download

echo "==== Starting validation and download: $(date) ===="

## Step 1: Validate and delete corrupted TIFFs
#echo "[1/2] Validating and cleaning up TIFFs..."
#python validate.py
#VALIDATION_EXIT=$?
#
#if [ $VALIDATION_EXIT -ne 0 ]; then
#  echo "❌ Validation script exited with error code $VALIDATION_EXIT"
#  exit 1
#fi

# Step 2: Download missing tiles
echo "[2/2] Downloading missing tiles..."
python download.py
DOWNLOAD_EXIT=$?

if [ $DOWNLOAD_EXIT -ne 0 ]; then
  echo "❌ Download script exited with error code $DOWNLOAD_EXIT"
  exit 1
fi

echo "✅ Done at $(date)"
