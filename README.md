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

From `services/intelligence/tensorflow_core`:

```bash
python train_ori.py --data data/ori_training_expanded.jsonl --output artifacts/ori-small
```

For the deployed intelligence image, the Docker build trains this bootstrap model and packages its weights with the runtime.

The English foundation is intentionally bounded so the free Render build remains practical. The model uses a SentencePiece BPE tokenizer shared by both stages, and a small English replay set during fine-tuning helps reduce catastrophic forgetting. This is still a compact bootstrap model, not a general-purpose assistant.

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

Web application:

- `ORI_INTELLIGENCE_URL` — server-side URL of the TensorFlow runtime.
- `ORI_INTELLIGENCE_API_KEY` — same shared secret as the runtime.

Never commit these credentials or expose them to client-side code.

## Current state

The web chat route is wired to the native TensorFlow runtime, and the runtime can load and generate from the trained `ori-small` artifact.

A live deployment is only considered connected when the runtime URL and API key are configured and `/v1/status` confirms that `ori-small` is loaded.

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
