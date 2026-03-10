This file explains two recommended ways to run the assignment on Google Cloud Platform (GCP):

1) Using a Deep Learning VM and a startup script (`gcp_startup.sh`) — fastest to get going.
2) Using Docker with NVIDIA Container Toolkit (`Dockerfile.gpu`) — best for reproducibility.

Both approaches assume you have a GCP project, `gcloud` installed locally, and GPU quota in your chosen zone.

Files added:
- `gcp_startup.sh` — a startup script you can run on a DL VM to fetch data from a GCS bucket, prepare the conda env, and run a smoke test.
- `Dockerfile.gpu` — Dockerfile tuned for CUDA 11.0 + PyTorch + PyTorch3D, useful for building an image to run on GCP VMs or Vertex AI.

Quick usage (Deep Learning VM):
1. Upload your large data to a GCS bucket (e.g. `gs://my-bucket/data/materials.zip`).
2. Create a DL VM (see commands in README or earlier notes). SSH into it.
3. Copy `gcp_startup.sh` to the VM and run it (it will pull data from your GCS bucket, create env, and run a small test). Edit variables at the top of the script before running.

Quick usage (Docker):
1. Build the image locally or on the VM that has Docker + nvidia runtime:
   docker build -t assignment3:gpu -f Dockerfile.gpu .
2. Run the container with GPU access and mount repo/data:
   docker run --gpus all -v $(pwd):/workspace -w /workspace --ipc=host assignment3:gpu bash -c "conda activate l3d && python -c 'print(\"ready\")'"

Verification steps included in both scripts:
- `nvidia-smi`
- `python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"`
- A small smoke run of `python volume_rendering_main.py --config-name=box` to produce a minimal output (it may crash later if TODOs aren't implemented, but will exercise ray/camera setup). 

Edit the variables in the scripts to fit your GCS bucket, paths, and chosen commands.
