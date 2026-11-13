#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Installing Ori Intelligence runtime dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Model training is deliberately opt-in. A web-service deployment should
# start the API predictably rather than run hours of CPU-heavy training in
# Render's build phase. Deploy pre-trained artifacts with the service, or
# explicitly set ORI_TRAIN_MODELS=true for a dedicated training build.
if [[ "${ORI_TRAIN_MODELS:-false}" != "true" ]]; then
  echo "ORI_TRAIN_MODELS is not true; skipping model training."
  echo "The API will start, but /v1/status will report model readiness accurately."
  exit 0
fi

mkdir -p models artifacts

echo "ORI_TRAIN_MODELS=true: training Ori models as an explicit opt-in..."
python tensorflow_core/train_core.py \
  --output models/ori_core.keras \
  --epochs "${ORI_CORE_EPOCHS:-30}" \
  --batch-size 8 \
  --validation-split 0.2

python tensorflow_core/train_ori.py \
  --data tensorflow_core/data/ori_training_expanded.jsonl \
  --output artifacts/ori-small \
  --epochs "${ORI_SMALL_EPOCHS:-20}" \
  --batch-size 8 \
  --validation-split 0.2

python tensorflow_core/train_specialists.py \
  --tokenizer artifacts/ori-small/ori_tokenizer.model \
  --output-root artifacts \
  --code-epochs "${ORI_CODER_EPOCHS:-10}" \
  --cad-epochs "${ORI_3D_EPOCHS:-10}"

echo "Ori model training completed."
