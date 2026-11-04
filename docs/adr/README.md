# Architecture decision records

Short, dated notes on decisions that are expensive to reverse. Each record states
the context, the decision, and the consequences. Records are immutable: when a
decision changes, add a new record that supersedes the old one.

| ADR | Title | Status |
| --- | --- | --- |
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-per-user-installer-with-verified-payload.md) | Per-user installer with a verified payload | Accepted |
| [0003](0003-metadata-first-capture-and-retention.md) | Metadata-first capture and bounded retention | Accepted |
| [0004](0004-pinned-toolchain-and-ci-gates.md) | Pinned toolchain and CI gates | Accepted |

## Adding a record

Copy the structure of an existing record, number it sequentially, and open a pull
request. Keep it under a page. If a decision is still open, record the options and
the reason it is unresolved rather than inventing certainty.
