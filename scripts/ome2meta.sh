#!/bin/bash
#SBATCH --job-name=ome2meta_sample_name          # Job name
#SBATCH --mail-type=BEGIN,END,FAIL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=your_email_address           # Where to send mail
#SBATCH --cpus-per-task=4                        # Number of CPU cores
#SBATCH --ntasks=1                               # Number of tasks
#SBATCH --partition=compute                      # Partition (default is all if you don't specify)
#SBATCH --mem=32G                                # Amount of memory in GB
#SBATCH --time=01:00:00                          # Time Limit D-HH:MM:SS
#SBATCH --output=ome2meta_sample_name_%j.log     # Standard output and error log

set -euo pipefail

OME_TIFF="/path/to/sample_name.ome.tiff"
OUTDIR="/path/to/meta"
CONDA_ENV="env_name"
SCRIPT="/path/to/ome2meta.py"

source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

mkdir -p "${OUTDIR}"

python "${SCRIPT}" "${OME_TIFF}" --outdir "${OUTDIR}"
