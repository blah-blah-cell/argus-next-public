# 0001 — Record architecture decisions

- Status: Accepted
- Date: 2026-11-04

## Context

The project has several decisions that look arbitrary from the outside: why a
scheduled task exists, why capture is off by default, why the setup program is
hand-built, why the model runtime is not bundled. Without a written record, every
reviewer re-litigates them, and a future contributor may "simplify" a safety
property without knowing it was deliberate.

## Decision

Keep short architecture decision records in `docs/adr/`, one file per decision,
numbered sequentially, with context, decision, and consequences. A record that
changes is superseded by a new record rather than edited.

## Consequences

- Reviewers can check intent instead of inferring it.
- Safety-relevant choices (capture defaults, elevation, packaging) are explicit.
- Small ongoing cost: each architectural change needs a short write-up.
