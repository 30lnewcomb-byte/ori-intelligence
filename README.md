# Ori Intelligence Service

This service is the server-side intelligence boundary for Ori Platform.

## Architecture

- **Ori TensorFlow language model** is the native learned language component.
- **TensorFlow Core** provides the small intent classifier used for platform signals.
- **Ori Core** owns identity, memory, tools, permissions, orchestration, and state outside the model weights.
- **The web app** talks to this service server-to-server. Runtime credentials are never exposed to the browser.

The production intelligence path no longer depends on a hosted external language-model provider.

## Native Ori model

`tensorflow_core/ori_model.py` contains a compact decoder-only Transformer implemented directly with TensorFlow/Keras. It supports:

- causal self-attention
- trainable token and position embeddings
- configurable model size and context length
- real learned generation from saved weights
- a stable tokenizer interface that can later be backed by a subword tokenizer
- JSONL training and held-out evaluation

The language model is trained in two stages: it first learns general English from a bounded public-domain Project Gutenberg corpus, then fine-tunes on `tensorflow_core/data/ori_training_expanded.jsonl` for Ori-specific behavior.

From the repository root:

```bash
python tensorflow_core/train_ori.py --data tensorflow_core/data/ori_training_expanded.jsonl --output artifacts/ori-small
```

Model training is intentionally separate from normal web-service startup. **The recommended improvement path is incremental fine-tuning from the saved Core and Ori Small checkpoints**, not rebuilding either model. The incremental trainer loads the existing weights, keeps the current vocabulary/tokenizer unchanged, uses a small batch and one epoch by default, replays a limited sample of earlier training examples, and evaluates the candidate against held-out data. Each candidate is kept only when its evaluation improves without a regression; otherwise the original checkpoint is copied forward unchanged.

The incremental path is opt-in through `ORI_FINE_TUNE_REVISION`. A unique revision is processed once and recorded in Render's build cache, so leaving the variable set does not retrain on every code deployment. This training runs in the Render build pipeline, not inside the live API process. It therefore does not consume the running service's free instance hours, but it does consume build-pipeline minutes. Normal deployments do no training.

The full from-scratch path is deliberately blocked unless both `ORI_TRAIN_MODELS=true` and `ORI_ALLOW_FULL_REBUILD=true` are set. Specialist workers are also separate: their first bootstrap requires both `ORI_TRAIN_SPECIALISTS=true` and `ORI_ALLOW_SPECIALIST_BOOTSTRAP=true`. Neither path is enabled by default.

The model uses a SentencePiece BPE tokenizer shared by both stages, and replay examples help reduce catastrophic forgetting during fine-tuning. This remains a compact bootstrap model, not a general-purpose assistant.

## Deployment and model artifacts

The standalone Render service builds from this repository root with Python 3.12.9 and starts:

```bash
uvicorn runtime:app --host 0.0.0.0 --port $PORT
```

The existing trained Core and Ori Small artifacts were imported once from the legacy runtime and saved in Render's persistent **build cache**. Each build restores those files into the service workspace; the service does not retrain them. This is a build-cache strategy, not a mounted persistent disk. If the cache is cleared, the build intentionally fails until the artifacts are restored or a deliberate training build is run.

The one-time bootstrap variables `ORI_MODEL_SOURCE_URL` and `ORI_MODEL_TRANSFER_TOKEN` are not needed after the first successful cache import and should remain unset. `ORI_TRAIN_MODELS` defaults to `false`; training is never an automatic recovery path.

Current artifact baseline: Ori Core and Ori Small load successfully in the standalone service. The specialist worker code and opt-in trainer are present, but neither `ori-coder` nor `ori-3d` is currently loaded; the legacy runtime also reports `workers=none`. Both workers remain unavailable until their trained artifacts are deliberately produced and deployed. No worker training runs automatically.

## API

Authenticated endpoints:

- `GET /v1/status` — runtime and model readiness
- `GET /v1/models` — registered TensorFlow models
- `POST /v1/predict` — intent classification
- `POST /v1/chat` — learned language generation

The runtime requires `Authorization: Bearer <ORI_INTELLIGENCE_API_KEY>` for these endpoints.

## Model boundary

```text
User message
    ↓
Ori Platform / server route
    ↓
Authenticated TensorFlow runtime
    ├── ori-small       ← learned language generation
    └── ori-core        ← intent classification
    ↓
Ori Core
    ├── identity
    ├── memory
    ├── tools
    ├── permissions
    └── orchestration
    ↓
User-facing response
```

The model does not own Ori's identity, memory, permissions, tools, or persistent state. This lets us improve or replace the weights without redefining Ori.

## Configuration

Runtime:

- `ORI_INTELLIGENCE_API_KEY` — shared secret used for server-to-server authentication.
- `ORI_LM_DIR` — path to the `ori-small` artifact directory; defaults to `artifacts/ori-small`.
- `ORI_MODEL_DIR` — path to the intent-classifier model directory; defaults to `models`.
- `LOG_LEVEL` — runtime log level; defaults to `INFO`.
- `ORI_FINE_TUNE_REVISION` — optional unique revision ID that runs one incremental Core/Ori Small fine-tune from the existing checkpoints. Each revision runs once; the completion marker is stored in the Render build cache.
- `ORI_CORE_FINE_TUNE_EPOCHS` and `ORI_SMALL_FINE_TUNE_EPOCHS` — incremental epochs, limited to 1-2; default `1`.
- `ORI_FINE_TUNE_BATCH_SIZE` — incremental batch size, limited to 1-4; default `2`.
- `ORI_TRAIN_MODELS` — legacy full-rebuild switch; blocked unless `ORI_ALLOW_FULL_REBUILD=true` is also set.
- `ORI_TRAIN_SPECIALISTS` — first-time worker bootstrap switch; blocked unless `ORI_ALLOW_SPECIALIST_BOOTSTRAP=true` is also set.

A normal deploy restores cached models without training. Incremental candidates are evaluated before promotion, and both original cached checkpoints remain untouched until candidate outputs are complete. Fine-tuning consumes Render build-pipeline minutes, not the live service's free instance hours.

Web application:

- `ORI_INTELLIGENCE_URL` — server-side URL of the TensorFlow runtime.
- `ORI_INTELLIGENCE_API_KEY` — same shared secret as the runtime.

Never commit these credentials or expose them to client-side code.

## Current state

The runtime is ready to load native TensorFlow artifacts when they are present. A healthy API process does not imply that its language model has been trained or loaded; check `/v1/status` before routing production chat traffic.

A live language-model deployment is only considered ready when the runtime URL and API key are configured and `/v1/status` confirms that `language_model_loaded` is true. Core and specialist readiness are reported independently.

## Rules

Do not add a hosted external language-model dependency to the production intelligence path.

Do not give the language model direct device authority.

Do not claim live intelligence until the configured runtime has been health-checked and its model readiness verified.

## Ori specialist workers

Ori stays as one user-facing intelligence. Ori Small handles the conversation
and task recognition, then can silently dispatch a matching internal worker.

```
User
  ↓
Ori Small
  ├── normal request → Ori Small response
  ├── coding request → ori-coder → result
  └── 3D/CAD request → ori-3d → validated geometry → result
```

The workers are implementation details. They cannot select other workers,
grant permissions, or execute platform tools directly. Tool permissions and
execution remain in the server-side platform.

The coding worker is a task-focused TensorFlow Transformer trained on coding
examples. The 3D worker is a task-focused TensorFlow Transformer trained to
produce constrained parametric CAD plans. A server-side CAD compiler converts
valid plans into OpenSCAD source, keeping the learned geometry proposal
inspectable and the final representation deterministic.

Worker artifacts live under:

- `artifacts/ori-coder`
- `artifacts/ori-3d`

The user-facing chat remains simply Ori. No worker selection UI is required.
