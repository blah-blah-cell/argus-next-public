# Research overview

Adaptive Hierarchical Event Memory (AHEM) is the research question inside Argus
Next: can a small model use a flow's own earlier, causally observed transitions to
forecast what that flow does next?

"Causally observed" is the important part. At every prediction point the model may
only use packets that have already arrived. It gets no future packets, no labels,
and no knowledge of how the capture ends. The targets are the next packet's size
and the flow's next arrival delay.

## Why forecast traffic at all

Most small organizations cannot afford full packet retention, and much of what
crosses a modern network is encrypted. Flow metadata — size, timing, direction,
protocol shape — survives both constraints. If a model can say "this flow's next
few packets look unlike its own history", that is something an analyst can look
at. It is a reason to look, not a verdict.

## The frozen experiment

In a frozen experiment on a selected subset of CSE-CIC-IDS2018, AHEM v6.1 had
lower weighted forecasting loss than a similarly sized flow-lane LSTM.

Four limits matter when reading that sentence:

- The selected sample covered **returning flows with eligible memory**, not every
  packet and not every kind of network.
- The compared models **did not receive identical recent global inputs**, so the
  comparison isolates the effect of memory only partly.
- Weighted forecasting loss is a **forecast** metric. It is not detection
  accuracy, precision, recall, or a false-positive rate.
- It is **one dataset and one frozen configuration**. Nothing here says the result
  transfers to another traffic mix, another topology, or another year.

## What this does not show

- That Argus detects intrusions, or detects them better than a simpler baseline.
- How often it would raise a false alert on a real network.
- That memory helps on flows without useful history — new or one-shot flows are
  exactly where the mechanism has the least to work with.
- That a forecast error translates into anything an operator would act on.

Those are the questions a detection claim would need to answer, and they have not
been answered. That is why the app describes its output as a forecast and keeps
the newer model experimental and disabled by default.

## Later work

A later, larger experimental model has been evaluated in exploratory local
comparisons. Its source provenance, fallback behavior and inference cost limit
what can currently be concluded from it. It is not promoted, cannot change a
security decision, and is shipped visibly labelled as experimental.

Negative and inconclusive outcomes are part of the record. A model that failed to
reproduce a gain is documented as a failure rather than quietly dropped, because
a research claim that only ever reports wins is not evidence.

## The artifacts

The shipped models are inference exports with manifests and whole-artifact hashes.
The experimental export excludes optimizer and RNG state; artifacts retain some
configuration and provenance metadata. The app checks the hashes and model
contracts before loading. Terms are in [models/NOTICE.md](../models/NOTICE.md), and the runtime
boundaries are described in [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

## What is public, and what is not

Published: the source, the inference-only weights and manifests, the roadmap, the
architecture decisions, and the limitations above.

Not published: the raw training captures, copies of the datasets, the full
training and evaluation harness, internal working notes, and the manuscript while
it is under review. The reasons are ordinary ones — capture data can identify
people, redistribution terms need checking, and a paper should describe its
methods and limits together rather than in fragments.

If you are reviewing the method and need something that is not here, open an issue
and ask. I would rather explain a gap than have it look like an omission.
