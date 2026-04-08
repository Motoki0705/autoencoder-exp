#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODELS=(
  "dino_resnet50_autoencoder"
  "dinov3_vits16_autoencoder"
  "dinov3_vitl16_autoencoder"
  "groundingdino_swin_t_autoencoder"
  "deformable_detr_resnet50_autoencoder"
)

for model_cfg in "${MODELS[@]}"; do
  echo "=== Running model=${model_cfg} ==="
  uv run python -m src.train \
    model="${model_cfg}" \
    experiment.version="${model_cfg}" \
    "$@"
done
