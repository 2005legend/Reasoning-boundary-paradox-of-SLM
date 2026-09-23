#!/usr/bin/env bash
# One-time setup on a fresh EC2 GPU instance (Deep Learning Base AMI with Single CUDA, Ubuntu 24.04, x86).
# torch 2.13 from PyPI is built for CUDA 13.0 and needs NVIDIA driver >= 580 (the 20260721 AMI ships 595).
#   bash scripts/setup_instance.sh            # 0.5B only
#   SIZES=0.5B,1.5B bash scripts/setup_instance.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HOME/venv}"
SIZES="${SIZES:-0.5B}"

echo "== GPU / driver =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "== system packages =="
sudo apt-get update -qq
sudo apt-get install -y -qq tmux htop >/dev/null

if ! command -v uv >/dev/null 2>&1; then
  echo "== installing uv (Python env manager) =="
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "== Python 3.11 venv at $VENV =="
[[ -d "$VENV" ]] || uv venv --python 3.11 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -r "$REPO/requirements.txt"

echo "== CUDA check =="
python - <<'EOF'
import torch
assert torch.cuda.is_available(), (
    "torch cannot see the GPU. torch 2.13 (CUDA 13.0) needs NVIDIA driver >= 580; compare the "
    "driver_version printed at the top. Relaunch with the newest 'Deep Learning Base AMI with "
    "Single CUDA (Ubuntu 24.04)', x86.")
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) on {torch.cuda.get_device_name(0)}, "
      f"bf16={torch.cuda.is_bf16_supported()}")
EOF

echo "== downloading models + datasets ($SIZES) =="
python "$REPO/scripts/prefetch.py" --sizes "$SIZES"

echo "== offline tests =="
(cd "$REPO" && python -m pytest -q tests -p no:cacheprovider)

# Without the CLI, checkpoint/result backups to S3 are silently skipped.
if ! command -v aws >/dev/null 2>&1; then
  echo "== installing AWS CLI =="
  sudo snap install aws-cli --classic >/dev/null || echo "WARNING: AWS CLI install failed; S3 backups will be skipped."
  hash -r
fi
echo "== AWS CLI: $(aws --version 2>&1) =="
aws sts get-caller-identity --query Arn --output text \
  || echo "WARNING: no AWS credentials. Attach the IAM role (README A6) or S3 sync will fail."

grep -q "source $VENV/bin/activate" "$HOME/.bashrc" || echo "source $VENV/bin/activate" >> "$HOME/.bashrc"
echo
echo "Setup complete. Next: bash scripts/start_jupyter.sh   (README Part E)"
