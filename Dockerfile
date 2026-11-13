FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    ORI_LM_DIR=/app/services/intelligence/artifacts/ori-small

WORKDIR /app

COPY services/intelligence/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY services/intelligence ./services/intelligence

WORKDIR /app/services/intelligence

# Train the native learned models during the image build so deployed inference
# always has real TensorFlow weights and never silently falls back to scripts.
RUN python tensorflow_core/train_core.py \
    --output models/ori_core.keras \
    --epochs 30 \
    --batch-size 8 \
    --validation-split 0.2

RUN python tensorflow_core/train_ori.py \
    --data tensorflow_core/data/ori_training_expanded.jsonl \
    --output artifacts/ori-small \
    --epochs 8 \
    --english-epochs 16 \
    --english-max-chars 1000000 \
    --batch-size 16 \
    --validation-split 0

# Train task-specialized workers from the shared English/Ori tokenizer.
RUN python tensorflow_core/train_specialists.py \
    --tokenizer artifacts/ori-small/ori_tokenizer.model \
    --output-root artifacts \
    --code-epochs 10 \
    --cad-epochs 10
EXPOSE 8000
CMD ["sh", "start.sh"]
