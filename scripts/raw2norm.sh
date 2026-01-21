#!/bin/bash
#SBATCH --job-name=raw2norm_sample_name          # Job name
#SBATCH --mail-type=BEGIN,END,FAIL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=your_email_address           # Where to send mail
#SBATCH --cpus-per-task=4                        # Number of CPU cores
#SBATCH --ntasks=1                               # Number of tasks
#SBATCH --partition=compute                      # Partition (default is all if you don't specify)
#SBATCH --mem=32G                                # Amount of memory in GB
#SBATCH --time=02:00:00                          # Time Limit D-HH:MM:SS
#SBATCH --output=raw2norm_sample_name_%j.log           # Standard output and error log

set -euo pipefail

IN_ZARR="/path/to/zarr/sample_name.codex_raw.zarr"
OUT_ZARR="/path/to/zarr/sample_name.codex_norm.zarr"
CONDA_ENV="env_name"
SCRIPT="/path/to/raw2norm.py"

# Parameters (can be modified)
P_LOW=1
P_HIGH=99
TRANSFORM="asinh"
ASINH_C=5
MAX_SAMPLES=500000      # speed-friendly, still robust
OUT_DTYPE="float32"     # safe default
CLEVEL=1                # faster I/O than clevel=3

source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

python "${SCRIPT}" \
  --in_zarr  "${IN_ZARR}" \
  --out_zarr "${OUT_ZARR}" \
  --p_low "${P_LOW}" \
  --p_high "${P_HIGH}" \
  --transform "${TRANSFORM}" \
  --c "${ASINH_C}" \
  --minmax \
  --max_samples_per_channel "${MAX_SAMPLES}" \
  --out_dtype "${OUT_DTYPE}" \
  --clevel "${CLEVEL}" \
  --overwrite

echo "[INFO] normalization finished successfully."
