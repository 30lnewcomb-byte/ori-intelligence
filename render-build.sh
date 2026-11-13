#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CACHE_ROOT="${XDG_CACHE_HOME:-/opt/render/project/.cache}/ori-tensorflow-models"
CACHE_VERSION="ori-models-v1"
CACHE_DIR="${CACHE_ROOT}/${CACHE_VERSION}"
MODEL_ROOT="services/intelligence"
CORE_MODEL="${MODEL_ROOT}/models/ori_core.keras"
LM_DIR="${MODEL_ROOT}/artifacts/ori-small"
CODER_DIR="${MODEL_ROOT}/artifacts/ori-coder"
CAD_DIR="${MODEL_ROOT}/artifacts/ori-3d"

mkdir -p "${MODEL_ROOT}/models" "${MODEL_ROOT}/artifacts"

echo "Installing Ori TensorFlow runtime dependencies..."
pip install -r services/intelligence/requirements.txt

fingerprint() {
  sha256sum \
    services/intelligence/tensorflow_core/train_core.py \
    services/intelligence/tensorflow_core/train_ori.py \
    services/intelligence/tensorflow_core/train_specialists.py \
    services/intelligence/tensorflow_core/model.py \
    services/intelligence/tensorflow_core/ori_model.py \
    services/intelligence/tensorflow_core/specialized_workers.py \
    services/intelligence/tensorflow_core/data/ori_training_expanded.jsonl \
    | sha256sum | cut -d' ' -f1
}

MODEL_FINGERPRINT="$(fingerprint)"

if [[ -f "${CACHE_DIR}/fingerprint" && "$(cat "${CACHE_DIR}/fingerprint")" == "${MODEL_FINGERPRINT}" \
      && -f "${CACHE_DIR}/models/ori_core.keras" \
      && -f "${CACHE_DIR}/artifacts/ori-small/config.json" \
      && -f "${CACHE_DIR}/artifacts/ori-small/model.weights.h5" \
      && -f "${CACHE_DIR}/artifacts/ori-small/ori_tokenizer.model" \
      && -f "${CACHE_DIR}/artifacts/ori-coder/config.json" \
      && -f "${CACHE_DIR}/artifacts/ori-coder/model.weights.h5" \
      && -f "${CACHE_DIR}/artifacts/ori-3d/config.json" \
      && -f "${CACHE_DIR}/artifacts/ori-3d/model.weights.h5" ]]; then
  echo "Restoring cached Ori models (${CACHE_VERSION})."
  rm -rf "${MODEL_ROOT}/models/ori_core.keras" "${MODEL_ROOT}/artifacts/ori-small" "${MODEL_ROOT}/artifacts/ori-coder" "${MODEL_ROOT}/artifacts/ori-3d"
  mkdir -p "${MODEL_ROOT}/models" "${MODEL_ROOT}/artifacts"
  cp "${CACHE_DIR}/models/ori_core.keras" "${CORE_MODEL}"
  cp -a "${CACHE_DIR}/artifacts/ori-small" "${LM_DIR}"
  cp -a "${CACHE_DIR}/artifacts/ori-coder" "${CODER_DIR}"
  cp -a "${CACHE_DIR}/artifacts/ori-3d" "${CAD_DIR}"
  echo "Cached models restored; no training required."
  exit 0
fi

echo "No matching trained model cache found. Training Ori models once for fingerprint ${MODEL_FINGERPRINT}..."

python services/intelligence/tensorflow_core/train_core.py \
  --output "${CORE_MODEL}" \
  --epochs 30 \
  --batch-size 8 \
  --validation-split 0.2

python services/intelligence/tensorflow_core/train_ori.py \
  --data services/intelligence/tensorflow_core/data/ori_training_expanded.jsonl \
  --output "${LM_DIR}" \
  --epochs 20 \
  --batch-size 8 \
  --validation-split 0.2

python services/intelligence/tensorflow_core/train_specialists.py \
  --tokenizer "${LM_DIR}/ori_tokenizer.model" \
  --output-root services/intelligence/artifacts \
  --code-epochs 10 \
  --cad-epochs 10

echo "Saving trained Ori models to Render's persistent build cache..."
rm -rf "${CACHE_DIR}"
mkdir -p "${CACHE_DIR}/models" "${CACHE_DIR}/artifacts"
cp "${CORE_MODEL}" "${CACHE_DIR}/models/ori_core.keras"
cp -a "${LM_DIR}" "${CACHE_DIR}/artifacts/ori-small"
cp -a "${CODER_DIR}" "${CACHE_DIR}/artifacts/ori-coder"
cp -a "${CAD_DIR}" "${CACHE_DIR}/artifacts/ori-3d"
printf '%s\n' "${MODEL_FINGERPRINT}" > "${CACHE_DIR}/fingerprint"

echo "Ori model training complete and cached. Future normal deploys will restore these artifacts instead of retraining."
