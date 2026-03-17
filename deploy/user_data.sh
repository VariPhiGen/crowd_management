#!/bin/bash
# =============================================================================
# Crimenabi Crowd Pipeline — EC2 Bootstrap Script (user_data.sh)
# =============================================================================
# Runs automatically when the EC2 instance first boots.
# Placeholders are replaced by launch_ec2.py before encoding:
#   __S3_BUCKET__   → crimenabi-data-variphi
#   __S3_CODE__     → code/crowd_management
#   __CAMERA_ARG__  → --process  OR  --process-camera cam_1
# =============================================================================

set -euo pipefail
exec > /var/log/crimenabi_bootstrap.log 2>&1
echo "=== Bootstrap started: $(date) ==="

S3_BUCKET="__S3_BUCKET__"
CAMERA_ARG="__CAMERA_ARG__"
GITHUB_REPO="https://github.com/VariPhiGen/crowd_management.git"
WORKDIR="/home/ubuntu/crowd_management"

# ── 1. System updates + NVIDIA driver + CUDA ──────────────────────────────────
echo "--- [1/6] Installing system packages and NVIDIA drivers..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q \
    build-essential git wget curl unzip \
    python3.10 python3.10-venv python3-pip \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    ffmpeg

# Install NVIDIA driver (Ubuntu 22.04 supports 525 for CUDA 12.x)
apt-get install -y -q ubuntu-drivers-common
ubuntu-drivers autoinstall || true

# Install CUDA 12.1 toolkit
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update -q
apt-get install -y -q cuda-toolkit-12-1
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> /etc/environment
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> /etc/environment
source /etc/environment

echo "--- NVIDIA/CUDA install done: $(date)"

# ── 2. Python virtual environment ─────────────────────────────────────────────
echo "--- [2/6] Setting up Python virtual environment..."
python3.10 -m venv /opt/crimenabi_venv
source /opt/crimenabi_venv/bin/activate

pip install --upgrade pip -q

# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# Project dependencies
pip install \
    ultralytics \
    easyocr \
    opencv-python-headless \
    boto3 \
    flask \
    pandas \
    numpy \
    scipy \
    -q

echo "--- Python deps done: $(date)"

# ── 3. Clone code from GitHub ─────────────────────────────────────────────────
echo "--- [3/6] Cloning code from GitHub..."
apt-get install -y -q git
git clone "$GITHUB_REPO" "$WORKDIR"
chmod +x "$WORKDIR"/deploy/*.sh 2>/dev/null || true
echo "--- Code cloned: $(date)"

# ── 4. Verify GPU is available ─────────────────────────────────────────────────
echo "--- [4/6] Verifying GPU..."
nvidia-smi || echo "WARNING: nvidia-smi failed — CUDA may not be ready yet"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# ── 5. Run the pipeline ────────────────────────────────────────────────────────
echo "--- [5/6] Running pipeline: python main.py ${CAMERA_ARG}"
cd "$WORKDIR"
python main.py ${CAMERA_ARG} \
    --model yolov8m.pt \
    2>&1 | tee /var/log/crimenabi_pipeline.log
echo "--- Pipeline finished: $(date)"

# ── 6. Upload results to S3 ───────────────────────────────────────────────────
echo "--- [6/6] Uploading results to S3..."
DATESTAMP=$(date +%Y%m%d-%H%M)
aws s3 sync "$WORKDIR/output/" "s3://${S3_BUCKET}/output/${DATESTAMP}/" \
    --exclude "tmp_s3/*" \
    --region ap-northeast-1
echo "--- Results uploaded to s3://${S3_BUCKET}/output/${DATESTAMP}/"
echo "--- Upload done: $(date)"

# ── Self-terminate ─────────────────────────────────────────────────────────────
echo "=== Bootstrap finished: $(date) — self-terminating ==="
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances \
    --instance-ids "$INSTANCE_ID" \
    --region ap-northeast-1
