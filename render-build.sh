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

if [[ -n "${ORI_MODEL_SOURCE_URL:-}" && -n "${ORI_MODEL_TRANSFER_TOKEN:-}" ]]; then
  echo "Downloading existing Ori model artifacts from the original runtime..."
  export ORI_MODEL_CACHE_DIR="${CACHE_DIR}"
  python - <<'PY'
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

url = os.environ["ORI_MODEL_SOURCE_URL"]
token = os.environ["ORI_MODEL_TRANSFER_TOKEN"]
cache_dir = Path(os.environ["ORI_MODEL_CACHE_DIR"])
archive_path = None
stage_dir = None
max_bytes = 512 * 1024 * 1024

try:
    with tempfile.NamedTemporaryFile(prefix="ori-model-download-", suffix=".zip", delete=False) as temporary:
        archive_path = Path(temporary.name)

    request = Request(url, headers={"X-Ori-Model-Transfer": token})
    total = 0
    with urlopen(request, timeout=300) as response, archive_path.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise RuntimeError("Ori model archive exceeds the 512 MiB build limit.")
            output.write(chunk)

    if total == 0:
        raise RuntimeError("Original runtime returned an empty model archive.")

    stage_dir = Path(tempfile.mkdtemp(prefix="ori-model-stage-"))
    allowed_roots = ("models/", "artifacts/ori-small/", "artifacts/ori-coder/", "artifacts/ori-3d/")
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or not member.filename.startswith(allowed_roots)
                or stat.S_ISLNK(mode)
            ):
                raise RuntimeError("Model archive contains an unsafe path or file type.")
        archive.extractall(stage_dir)

    required = (
        stage_dir / "models" / "ori_core.keras",
        stage_dir / "artifacts" / "ori-small" / "config.json",
        stage_dir / "artifacts" / "ori-small" / "ori_tokenizer.model",
        stage_dir / "artifacts" / "ori-small" / "model.weights.h5",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError("Downloaded archive is missing required Ori Core or Ori Small artifacts.")

    # Populate the build output that becomes the service's runtime filesystem.
    Path("models").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    shutil.copy2(required[0], Path("models") / "ori_core.keras")
    shutil.copytree(stage_dir / "artifacts" / "ori-small", Path("artifacts") / "ori-small", dirs_exist_ok=True)
    for model_name in ("ori-coder", "ori-3d"):
        source = stage_dir / "artifacts" / model_name
        if source.is_dir():
            shutil.copytree(source, Path("artifacts") / model_name, dirs_exist_ok=True)

    # Persist the same artifacts in Render's build cache so future deploys restore
    # them without contacting the original service or retraining.
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    (cache_dir / "models").mkdir(parents=True, exist_ok=True)
    (cache_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(required[0], cache_dir / "models" / "ori_core.keras")
    shutil.copytree(stage_dir / "artifacts" / "ori-small", cache_dir / "artifacts" / "ori-small")
    for model_name in ("ori-coder", "ori-3d"):
        source = stage_dir / "artifacts" / model_name
        if source.is_dir():
            shutil.copytree(source, cache_dir / "artifacts" / model_name)
    (cache_dir / "source.txt").write_text("Imported from original Ori TensorFlow runtime\n", encoding="utf-8")
    print(f"Imported and cached Ori artifacts ({total} archive bytes).")
finally:
    if stage_dir is not None:
        shutil.rmtree(stage_dir, ignore_errors=True)
    if archive_path is not None:
        archive_path.unlink(missing_ok=True)
PY
  exit 0
fi

# Model training remains opt-in. Never retrain just because the source is
# temporarily unavailable; fail clearly so an incomplete runtime is not
# mistaken for a ready model service.
if [[ "${ORI_TRAIN_MODELS:-false}" != "true" ]]; then
  echo "No cached models or authenticated model source configured."
  echo "Skipping training; /v1/status will report model readiness as false."
  exit 0
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
