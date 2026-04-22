# MI300X CAMEL Full Pipeline Runbook

This guide is intended for running the full-size CAMEL pipeline on an AMD MI300X server after SSH login.

## 1) SSH into server

    ssh YOUR_USER@YOUR_SERVER_IP

## 2) Install base packages (Ubuntu)

    sudo apt update
    sudo apt install -y git python3 python3-venv python3-pip build-essential

## 3) Clone and enter repository

    git clone https://github.com/YOUR_ORG/vortex-codec.git
    cd vortex-codec

If repo already exists:

    cd /path/to/vortex-codec

## 4) Verify ROCm and GPU visibility

    which rocm-smi
    rocm-smi
    python3 -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('hip=', torch.version.hip)"

Expected:
- rocm-smi shows MI300X device info
- torch reports cuda_available=True and hip not None

## 5) Make full pipeline script executable

    chmod +x scripts/full_pipeline_mi300x_camel.sh

## 6) Run full-size pipeline (train + full-file eval + full-file baselines)

    FULL_VORTEX=1 SAMPLE_MB=999999 VORTEX_MB=999999 EVAL_BATCH_SIZE=128 ./scripts/full_pipeline_mi300x_camel.sh

What this does:
- Creates or reuses local virtual environment
- Installs Python dependencies
- Downloads and prepares full CAMEL data (server profile)
- Trains with configs/camel_mi300x.yaml
- Runs benchmark evaluation and saves JSON

## 7) Faster rerun mode (skip setup and download)

Use this if dependencies and data are already present:

    SKIP_SETUP=1 SKIP_DOWNLOAD=1 FULL_VORTEX=1 SAMPLE_MB=999999 VORTEX_MB=999999 EVAL_BATCH_SIZE=128 ./scripts/full_pipeline_mi300x_camel.sh

## 8) Run in background (recommended for long jobs)

    nohup env FULL_VORTEX=1 SAMPLE_MB=999999 VORTEX_MB=999999 EVAL_BATCH_SIZE=128 ./scripts/full_pipeline_mi300x_camel.sh > full_pipeline.out 2>&1 &
    echo $!

## 9) Monitor progress

    tail -f full_pipeline.out

In another shell:

    tail -f logs/mi300x_camel_*/train.log
    tail -f logs/mi300x_camel_*/eval.log
    watch -n 2 rocm-smi

## 10) Output locations

- Results JSON:
  - results/camel_mi300x_results.json
- Run logs:
  - logs/mi300x_camel_TIMESTAMP/

## 11) Copy result back to local machine

Run from your local machine:

    scp YOUR_USER@YOUR_SERVER_IP:/path/to/vortex-codec/results/camel_mi300x_results.json .

## 12) One-shot copy-paste block

If you want a single block after SSH:

    cd /path/to/vortex-codec
    chmod +x scripts/full_pipeline_mi300x_camel.sh
    nohup env FULL_VORTEX=1 SAMPLE_MB=999999 VORTEX_MB=999999 EVAL_BATCH_SIZE=128 ./scripts/full_pipeline_mi300x_camel.sh > full_pipeline.out 2>&1 &
    echo "PID: $!"
    tail -f full_pipeline.out

## 13) Troubleshooting: CUDA torch accidentally installed on MI300X

Symptom:
- Pipeline prints torch with +cuXXX and then fails with CUDA/ROCm not available.

Quick recovery:

    cd /path/to/vortex-codec
    rm -rf .venv
    python3 -m pip install --upgrade pip setuptools wheel
    python3 -m pip install -r <(grep -E -v '^[[:space:]]*torch([[:space:]]|[<>=!~]|$)' requirements.txt)
    python3 -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('hip=', torch.version.hip)"
    SKIP_SETUP=1 SKIP_DOWNLOAD=1 FULL_VORTEX=1 SAMPLE_MB=999999 VORTEX_MB=999999 EVAL_BATCH_SIZE=128 ./scripts/full_pipeline_mi300x_camel.sh

Notes:
- On MI300X, expected check output is cuda_available=True and hip not None.
- If you still see +cu in torch version, the active environment is not using ROCm torch.
