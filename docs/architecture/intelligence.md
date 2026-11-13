# Ori Intelligence Architecture

## Core decision

Ori's **TensorFlow model is the primary AI model**.

TensorFlow is the machine-learning framework used to build, train, evaluate, and run that model. The model itself is Ori's learned intelligence.

Alongside it, Ori will have a smaller **Mentor** language model. The Mentor is not the replacement for Ori's core model. It is a supporting intelligence layer that is especially good at language-model tasks and can help the core model interpret, structure, evaluate, and communicate complex work.

## Mental model

Think of the relationship like a parent helping a toddler:

- The TensorFlow model is the developing core intelligence.
- The Mentor has stronger language/task-structuring abilities and helps the core intelligence where it is currently weak.
- The orchestrator coordinates both.
- Evaluation determines whether changes actually improve the core model.

The analogy describes assistance, not unrestricted control. The Mentor must not silently overwrite the core model or become the authority that decides its own correctness.

## Responsibilities

### TensorFlow core

The core model is responsible for Ori's primary learned behavior. Its architecture, training data, objectives, and capabilities will evolve over time.

### Mentor

The Mentor can assist with:

- language understanding and reformulation
- task decomposition
- structured planning assistance
- explaining tool results to the core model
- identifying ambiguity or uncertainty
- generating candidate responses or plans for evaluation
- evaluating outputs against defined criteria
- helping construct training/evaluation examples

### Orchestrator

The orchestrator decides when and how the two models interact. It also connects intelligence to memory, tools, sandbox execution, permissions, and user-facing responses.

### Evaluation

Candidate changes must be evaluated independently. Mentor feedback can be one signal, but it is not sufficient by itself to promote a model.

A future promotion path should look like:

```text
Training data
    ↓
Candidate TensorFlow model
    ↓
Automated benchmarks
    ↓
Mentor-assisted evaluation
    ↓
Regression / safety checks
    ↓
Sandbox testing
    ↓
Human approval when required
    ↓
Promotion
```

## Replaceability

The Mentor should be accessed through a stable internal interface so its underlying model can be replaced later without redesigning Ori's core architecture.

The same principle applies to the TensorFlow model: the orchestration layer should not assume one permanent model architecture.

## Long-term learning

Ori may eventually support continuous improvement, but production self-modification must be gated by evaluation. A model should never promote itself merely because it believes it improved.

The goal is **self-improvement with evidence**, not uncontrolled self-modification.
