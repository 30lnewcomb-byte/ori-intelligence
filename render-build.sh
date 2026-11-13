#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CACHE_ROOT="${XDG_CACHE_HOME:-/opt/render/project/.cache}/ori-intelligence-models"
CACHE_VERSION="ori-models-v1"
CACHE_DIR="${CACHE_ROOT}/${CACHE_VERSION}"
CORE_MODEL="models/ori_core.keras"
LM_DIR="artifacts/ori-small"

mkdir -p models artifacts

echo "Installing Ori Intelligence runtime dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

restore_cached_models() {
  echo "Restoring Ori model artifacts from Render's persistent build cache..."
  mkdir -p models artifacts
  cp "${CACHE_DIR}/models/ori_core.keras" "${CORE_MODEL}"
  rm -rf "${LM_DIR}"
  cp -a "${CACHE_DIR}/artifacts/ori-small" "${LM_DIR}"
  for model_name in ori-coder ori-3d; do
    if [[ -d "${CACHE_DIR}/artifacts/${model_name}" ]]; then
      rm -rf "artifacts/${model_name}"
      cp -a "${CACHE_DIR}/artifacts/${model_name}" "artifacts/${model_name}"
    fi
  done
  echo "Ori Core and Ori Small restored; no retraining required."
}

if [[ -f "${CACHE_DIR}/models/ori_core.keras" \
      && -f "${CACHE_DIR}/artifacts/ori-small/config.json" \
      && -f "${CACHE_DIR}/artifacts/ori-small/ori_tokenizer.model" \
      && -f "${CACHE_DIR}/artifacts/ori-small/model.weights.h5" ]]; then
  restore_cached_models
  exit 0
fi

# Model training remains opt-in. Never retrain just because the source is
# temporarily unavailable; fail clearly so an incomplete runtime is not
# mistaken for a ready model service.
if [[ "${ORI_TRAIN_MODELS:-false}" != "true" ]]; then
  echo "ERROR: No cached models or authenticated model source configured."
  echo "Refusing to deploy an incomplete inference service. Restore the artifacts or opt in to a deliberate training build."
  exit 1
fi

echo "ORI_TRAIN_MODELS=true: training Ori models as an explicit opt-in..."
python tensorflow_core/train_core.py \
  --output "${CORE_MODEL}" \
  --epochs "${ORI_CORE_EPOCHS:-30}" \
  --batch-size 8 \
  --validation-split 0.2

python tensorflow_core/train_ori.py \
  --data tensorflow_core/data/ori_training_expanded.jsonl \
  --output "${LM_DIR}" \
  --epochs "${ORI_SMALL_EPOCHS:-20}" \
  --batch-size 8 \
  --validation-split 0.2

python tensorflow_core/train_specialists.py \
  --tokenizer "${LM_DIR}/ori_tokenizer.model" \
  --output-root artifacts \
  --code-epochs "${ORI_CODER_EPOCHS:-10}" \
  --cad-epochs "${ORI_3D_EPOCHS:-10}"

echo "Ori model training completed."
