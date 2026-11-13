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

# A full rebuild is intentionally blocked unless a second, explicit safety
# switch is also supplied. Incremental fine-tuning is the normal improvement
# path and never calls the from-scratch trainers.
if [[ "${ORI_TRAIN_MODELS:-false}" == "true" \
      && "${ORI_ALLOW_FULL_REBUILD:-false}" != "true" ]]; then
  echo "ERROR: Full model rebuilding is blocked."
  echo "Use ORI_FINE_TUNE_REVISION to continue from the existing checkpoints."
  echo "A full rebuild requires both ORI_TRAIN_MODELS=true and ORI_ALLOW_FULL_REBUILD=true."
  exit 1
fi

if [[ -n "${ORI_FINE_TUNE_REVISION:-}" \
      && "${ORI_TRAIN_MODELS:-false}" == "true" ]]; then
  echo "ERROR: Incremental fine-tuning and full rebuilding cannot run together."
  exit 1
fi

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
  echo "Restore the artifacts or deliberately authorize a full rebuild."
  exit 1
fi

# This old path remains available only behind the two-switch guard above.
# It is not the recommended way to improve an existing model.
if [[ "${ORI_TRAIN_MODELS:-false}" == "true" ]]; then
  echo "Explicitly authorized full rebuild: training Core and Ori Small from scratch..."
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

# Fine-tune an exact revision once. The marker lives beside the cached model
# artifacts, so leaving the environment variable set will not repeat training
# on every subsequent code deployment.
if [[ -n "${ORI_FINE_TUNE_REVISION:-}" ]]; then
  if [[ ! "${ORI_FINE_TUNE_REVISION}" =~ ^[A-Za-z0-9._-]{1,64}$ ]]; then
    echo "ERROR: ORI_FINE_TUNE_REVISION must be 1-64 letters, numbers, dots, underscores, or hyphens."
    exit 1
  fi

  FINE_TUNE_MARKER="${CACHE_DIR}/fine-tune/${ORI_FINE_TUNE_REVISION}.done"
  if [[ -f "${FINE_TUNE_MARKER}" ]]; then
    echo "Fine-tune revision ${ORI_FINE_TUNE_REVISION} was already processed; skipping training."
  else
    echo "Preparing one-shot incremental fine-tuning from the cached checkpoints..."
    CANDIDATE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ori-finetune.XXXXXX")"
    trap 'rm -rf "${CANDIDATE_ROOT}"' EXIT

    python tensorflow_core/fine_tune_existing.py \
      --core-input "${CORE_MODEL}" \
      --core-output "${CANDIDATE_ROOT}/models/ori_core.keras" \
      --small-input-dir "${LM_DIR}" \
      --small-output-dir "${CANDIDATE_ROOT}/artifacts/ori-small" \
      --core-data tensorflow_core/data/ori_core_incremental.jsonl \
      --small-data tensorflow_core/data/ori_small_incremental.jsonl \
      --small-replay-data tensorflow_core/data/ori_training_expanded.jsonl \
      --small-eval-data tensorflow_core/data/ori_eval_expanded.jsonl \
      --core-epochs "${ORI_CORE_FINE_TUNE_EPOCHS:-1}" \
      --small-epochs "${ORI_SMALL_FINE_TUNE_EPOCHS:-1}" \
      --batch-size "${ORI_FINE_TUNE_BATCH_SIZE:-2}" \
      --summary "${CANDIDATE_ROOT}/fine-tune-summary.json"

    if [[ ! -f "${CANDIDATE_ROOT}/models/ori_core.keras" \
          || ! -f "${CANDIDATE_ROOT}/artifacts/ori-small/config.json" \
          || ! -f "${CANDIDATE_ROOT}/artifacts/ori-small/ori_tokenizer.model" \
          || ! -f "${CANDIDATE_ROOT}/artifacts/ori-small/model.weights.h5" ]]; then
      echo "ERROR: Incremental fine-tuning did not produce a complete candidate artifact set."
      exit 1
    fi

    echo "Incremental evaluation passed; installing the candidate-or-original outputs..."
    cp "${CANDIDATE_ROOT}/models/ori_core.keras" "${CORE_MODEL}"
    rm -rf "${LM_DIR}"
    cp -a "${CANDIDATE_ROOT}/artifacts/ori-small" "${LM_DIR}"
    cache_base_models

    mkdir -p "$(dirname "${FINE_TUNE_MARKER}")"
    printf '%s\n' "${ORI_FINE_TUNE_REVISION}" > "${FINE_TUNE_MARKER}.tmp"
    mv "${FINE_TUNE_MARKER}.tmp" "${FINE_TUNE_MARKER}"
    echo "Fine-tune revision ${ORI_FINE_TUNE_REVISION} completed and recorded."
  fi
fi

# Specialist weights do not exist yet. Their first training is a separate,
# explicitly authorized bootstrap and is never part of normal deployments.
if [[ "${ORI_TRAIN_SPECIALISTS:-false}" == "true" ]]; then
  if [[ "${ORI_ALLOW_SPECIALIST_BOOTSTRAP:-false}" != "true" ]]; then
    echo "ERROR: Specialist bootstrap is blocked."
    echo "It requires both ORI_TRAIN_SPECIALISTS=true and ORI_ALLOW_SPECIALIST_BOOTSTRAP=true."
    exit 1
  fi

  echo "Explicitly authorized specialist bootstrap..."
  python tensorflow_core/train_specialists.py \
    --tokenizer "${LM_DIR}/ori_tokenizer.model" \
    --output-root artifacts \
    --code-epochs "${ORI_CODER_EPOCHS:-10}" \
    --cad-epochs "${ORI_3D_EPOCHS:-10}"

  cache_specialist_models
else
  echo "Specialist bootstrap is disabled; restoring only previously cached workers."
fi

echo "Ori Intelligence model preparation completed."
