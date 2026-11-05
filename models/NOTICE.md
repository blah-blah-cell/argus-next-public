# Model weights notice

The files under `models/` are research artifacts exported for inference. They are
**not** covered by the repository's source license in `LICENSE`, and no permission
to redistribute them is granted by that license.

## What is shipped

- The frozen AHEM v6.1 forecast model, available as PyTorch and ONNX artifacts.
- An experimental end-to-end shadow challenger, disabled by default and not
  eligible to make security decisions.
- The end-to-end export omits optimizer and RNG state. The artifacts still
  contain limited provenance and configuration metadata.

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

The application checks whole-artifact hashes and validates the expected model
schema and tensor state when loading. The manifests do not provide a per-tensor
digest inventory. Original training checkpoints and raw captures are not included;
the inference artifacts and manifests retain selected configuration and provenance.
