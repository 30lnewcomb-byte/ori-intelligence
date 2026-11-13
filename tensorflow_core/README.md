# Ori TensorFlow Core

This is Ori's native learned-intelligence boundary.

## Runtime architecture

```
Ori Platform (Vercel)
        |
        | authenticated HTTPS
        v
Ori TensorFlow Runtime
        |
        +--> /v1/status
        +--> /v1/models
        +--> /v1/predict
        +--> /v1/chat
        |
        v
TensorFlow / Keras models
```

The Vercel application is the control plane. The TensorFlow runtime owns model loading and inference. Model files and service credentials stay server-side.

## Current model

`model.py` contains Ori's first real trainable TensorFlow/Keras classifier. It is a learned model, not a scripted decision tree. It currently provides intent signals for the orchestrator.

A classifier is deliberately not presented as a chat model. The runtime reserves `/v1/chat` for a future trained TensorFlow generative model.

## Model contract

Every deployed model should have:

- a stable model id
- an explicit version
- a framework/task declaration
- a server-side load path
- a stable inference contract

That lets Ori add larger TensorFlow models later without changing the Vercel frontend.

## Security

The runtime API requires a server-side bearer credential. Keep the runtime private where the hosting platform supports private networking. Never put the runtime credential in client-side code, localStorage, query parameters, or browser-visible responses.

Hugging Face is not part of this production intelligence path. Ori calls its own TensorFlow runtime.
