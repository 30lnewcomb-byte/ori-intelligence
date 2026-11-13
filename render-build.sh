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

base_models_cached() {
  [[ -f "${CACHE_DIR}/models/ori_core.keras" \
      && -f "${CACHE_DIR}/artifacts/ori-small/config.json" \
      && -f "${CACHE_DIR}/artifacts/ori-small/ori_tokenizer.model" \
      && -f "${CACHE_DIR}/artifacts/ori-small/model.weights.h5" ]]
}

restore_cached_models() {
  echo "Restoring Ori model artifacts from Render's persistent build cache..."
  mkdir -p models artifacts
  cp "${CACHE_DIR}/models/ori_core.keras" "${CORE_MODEL}"
  rm -rf "${LM_DIR}"
  cp -a "${CACHE_DIR}/artifacts/ori-small" "${LM_DIR}"

  for model_name in ori-coder ori-3d; do
    rm -rf "artifacts/${model_name}"
    if [[ -d "${CACHE_DIR}/artifacts/${model_name}" ]]; then
      cp -a "${CACHE_DIR}/artifacts/${model_name}" "artifacts/${model_name}"
    fi
  done
  echo "Cached Ori Core and Ori Small restored."
}

cache_base_models() {
  echo "Saving base model artifacts into Render's build cache..."
  mkdir -p "${CACHE_DIR}/models" "${CACHE_DIR}/artifacts"
  cp "${CORE_MODEL}" "${CACHE_DIR}/models/ori_core.keras"
  rm -rf "${CACHE_DIR}/artifacts/ori-small"
  cp -a "${LM_DIR}" "${CACHE_DIR}/artifacts/ori-small"
}

cache_specialist_models() {
  echo "Saving specialist worker artifacts into Render's build cache..."
  mkdir -p "${CACHE_DIR}/artifacts"
  for model_name in ori-coder ori-3d; do
    if [[ ! -f "artifacts/${model_name}/config.json" \
          || ! -f "artifacts/${model_name}/ori_tokenizer.model" \
          || ! -f "artifacts/${model_name}/model.weights.h5" ]]; then
      echo "ERROR: Specialist training did not produce a complete ${model_name} artifact set."
      exit 1
    fi
    rm -rf "${CACHE_DIR}/artifacts/${model_name}"
    cp -a "artifacts/${model_name}" "${CACHE_DIR}/artifacts/${model_name}"
  done
}

if base_models_cached; then
  restore_cached_models
elif [[ "${ORI_TRAIN_MODELS:-false}" != "true" ]]; then
  echo "ERROR: No cached Core/Ori Small models are available."
  echo "Restore the artifacts or deliberately set ORI_TRAIN_MODELS=true in a training build."
  exit 1
fi

# A cache hit must not silently skip an explicitly requested training run.
# ORI_TRAIN_MODELS rebuilds Core and Ori Small; it is intentionally expensive.
if [[ "${ORI_TRAIN_MODELS:-false}" == "true" ]]; then
  echo "ORI_TRAIN_MODELS=true: training Ori Core and Ori Small as explicitly requested..."
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

  cache_base_models
fi

# Specialist training can be requested independently while reusing the
# existing Core/Ori Small cache. No specialist training happens by default.
if [[ "${ORI_TRAIN_MODELS:-false}" == "true" \
      || "${ORI_TRAIN_SPECIALISTS:-false}" == "true" ]]; then
  echo "Training Ori's coding and 3D workers as explicitly requested..."
  python tensorflow_core/train_specialists.py \
    --tokenizer "${LM_DIR}/ori_tokenizer.model" \
    --output-root artifacts \
    --code-epochs "${ORI_CODER_EPOCHS:-10}" \
    --cad-epochs "${ORI_3D_EPOCHS:-10}"

  cache_specialist_models
else
  echo "Specialist worker training is disabled; restoring only any previously cached worker artifacts."
fi

echo "Ori Intelligence model preparation completed."
