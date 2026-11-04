# 0003 — Metadata-first capture and bounded retention

- Status: Accepted
- Date: 2026-11-04

## Context

Argus can record raw packets. At 10 Mbps that is roughly 4.5 GB per hour, it
contains personal data by default, and it is the single largest privacy and disk
risk in the product. The intended users are small organizations without a legal or
security team to manage that risk.

## Decision

- The application never starts capturing on its own. Capture is an explicit action
  with a visible state.
- Metadata (flows, counters, timing, protocol shape) is the default product;
  payload retention is an explicit, bounded opt-in.
- Retention defaults are short, visible in the interface, and enforced by the
  application rather than by documentation alone.
- Live capture requires Pktmon and an elevation path that is separate from the UI
  process; when it is unavailable the app reports the truthful state instead of
  degrading silently.

## Consequences

- Lower privacy exposure and predictable disk use for the default configuration.
- Some investigations need payloads; those users must make a deliberate choice and
  accept the storage and privacy cost.
- Detection and forecasting must remain useful on metadata features, which
  constrains model design.
- Retention enforcement needs tests, not just defaults, or this decision erodes.
