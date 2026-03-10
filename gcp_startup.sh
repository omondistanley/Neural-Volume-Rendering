#!/usr/bin/env bash
# gcp_startup.sh
# Usage:
#  - Edit USER_GCS_BUCKET to point to your GCS bucket (must already exist)
#  - Run this script on the DL VM after SSH: bash gcp_startup.sh

set -euo pipefail
USER_GCS_BUCKET="my-assignment3-bucket-$(whoami)"   # <-- EDIT ME
MATERIALS_GCS_PATH="gs://${USER_GCS_BUCKET}/data/materials.zip"
REPO_DIR="$HOME/assignment3"
ENV_NAME="l3d"
CONDA_ENV_FILE="environment.yml"

echo "Starting GCP startup tasks"
# show GPU
nvidia-smi || true

# Make a directory for the repo
mkdir -p "$REPO_DIR"
cd "$REPO_DIR"

# If you cloned repository already, skip. Otherwise instruct how to clone.
if [ ! -d assignment3-main ]; then
  echo "Please clone your assignment repo into $REPO_DIR, or use git clone now."
  echo "Example: git clone https://github.com/<your-repo>.git ."
  # pause - user may want to clone manually
fi

# Fetch big data from GCS (materials.zip)
if gsutil ls "$MATERIALS_GCS_PATH" >/dev/null 2>&1; then
  echo "Found materials archive in GCS. Downloading..."
  gsutil cp "$MATERIALS_GCS_PATH" . || true
  if [ -f materials.zip ]; then
    echo "Unzipping materials.zip to assignment3-main/data/"
    mkdir -p assignment3-main/data
    unzip -o materials.zip -d assignment3-main/data || true
  fi
else
  echo "No materials.zip found at $MATERIALS_GCS_PATH — skip download."
fi

# Create conda env if environment.yml exists
if command -v conda >/dev/null 2>&1 && [ -f assignment3-main/$CONDA_ENV_FILE ]; then
  echo "Creating conda environment $ENV_NAME from environment.yml (may take a while)."
  conda env list | grep "^$ENV_NAME" || conda env create -f assignment3-main/$CONDA_ENV_FILE
  echo "Activating env $ENV_NAME"
  # shellcheck disable=SC1091
  source $(conda info --base)/etc/profile.d/conda.sh
  conda activate "$ENV_NAME"
else
  echo "Conda not found or environment.yml missing. You may need to manually install Conda or edit this script."
fi

# quick checks
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# optional quick smoke run (won't fully run if many TODOs remain)
echo "Running a small smoke test to render a single camera frame (may error if functions unimplemented)"
python assignment3-main/volume_rendering_main.py --config-name=box || true

echo "Startup script complete. Remember to run jobs inside tmux/screen and upload checkpoints to GCS."

# Helpful reminders
cat <<EOF
Reminders:
 - Upload checkpoints frequently: gsutil cp model_checkpoint.pt gs://$USER_GCS_BUCKET/checkpoints/
 - To copy results back: gsutil cp images/*.gif gs://$USER_GCS_BUCKET/results/
EOF
