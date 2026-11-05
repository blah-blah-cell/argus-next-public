# 0003 — Metadata-first capture and bounded retention

- Status: Proposed direction; not fully implemented
- Date: 2026-11-04

## Context

Argus can record raw packets. At 10 Mbps that is roughly 4.5 GB per hour, it
contains personal data by default, and it is the single largest privacy and disk
risk in the product. The intended users are small organizations without a legal or
security team to manage that risk.

## Current implementation and intended direction

The current live sensor captures packet bytes into temporary local PCAP
segments, with a configured 9,000-byte snapshot length. These bytes can include
payloads. Starting monitoring is the consent action; there is not currently a
separate payload-retention opt-in or metadata-only capture mode. The derived
learning journal uses numeric features without payload bytes. The product should
not be represented as metadata-only capture.

The following are design goals, not completed guarantees:

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

- The current capture format still exposes packet payloads locally; metadata-only
  capture and a strict storage budget require further work.
- Some investigations need payloads; those users must make a deliberate choice and
  accept the storage and privacy cost.
- Detection and forecasting must remain useful on metadata features, which
  constrains model design.
- Retention enforcement needs tests, not just defaults, or this decision erodes.
