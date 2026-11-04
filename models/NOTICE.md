# Model weights notice

The files under `models/` are research artifacts exported for inference. They are
**not** covered by the repository's source license in `LICENSE`, and no permission
to redistribute them is granted by that license.

## What is shipped

- A validated base ensemble (`argus.ahem-e2e-inference-weights/v1` manifest).
- A shadow challenger, recorded for comparison only.
- Sanitized checkpoints: no optimizer state, no learning-rate schedule, no epoch
  counter, no RNG state, and no dataset identifiers.

## What they are

Research outputs of the training pipeline. They are not a validated detector and
must not be described as one. Forecast quality on a deployment depends on traffic,
topology, and time of day in ways that a public release cannot guarantee.

## Terms

Evaluation and research use is intended. Redistribution, commercial deployment, or
publishing derived weights requires the author's written permission until a
separate model license is published. If you need different terms for a pilot, ask
before deploying.

## Provenance

Every artifact carries a manifest with its schema version, parameter count, tensor
names, and per-tensor digests. The application verifies those digests before
loading and refuses an artifact that does not match. The source checkpoint and the
training configuration are not published.
