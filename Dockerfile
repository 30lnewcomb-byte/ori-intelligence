FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    ORI_LM_DIR=/app/artifacts/ori-small

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Keep normal image builds fast and predictable. Model training is an
# explicit, separately managed operation rather than a side effect of deploy.
ARG ORI_TRAIN_MODELS=false
RUN if [ "$ORI_TRAIN_MODELS" = "true" ]; then \
      python tensorflow_core/train_core.py --output models/ori_core.keras --epochs 30 --batch-size 8 --validation-split 0.2 && \
      python tensorflow_core/train_ori.py --data tensorflow_core/data/ori_training_expanded.jsonl --output artifacts/ori-small --epochs 20 --batch-size 8 --validation-split 0.2 && \
      python tensorflow_core/train_specialists.py --tokenizer artifacts/ori-small/ori_tokenizer.model --output-root artifacts --code-epochs 10 --cad-epochs 10; \
    else \
      echo "Skipping model training (ORI_TRAIN_MODELS=false)."; \
    fi

EXPOSE 8000
CMD ["sh", "start.sh"]
