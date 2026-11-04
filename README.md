# Argus Next

Argus Next is an early-stage Windows network-monitoring research prototype. It is intended to help operators inspect local network activity and investigate alerts. It is not a managed security service or a replacement for an established security program.

## What is being explored

The desktop application brings together packet capture, flow views, replay and temporal models. One research component, Adaptive Hierarchical Event Memory (AHEM), forecasts packet characteristics using recent traffic and previously observed flow transitions. A forecast is not an intrusion verdict.

The system is under active development. Its effectiveness at detecting attacks in an organization's network, its false-alert rate and its suitability for production remain to be evaluated in an authorized pilot.

## Privacy and pilots

Network metadata can be sensitive even when packet payloads are not retained. Before any pilot, the operator and organization should agree in writing on capture scope, access permissions, retention, exports and incident handling. Do not capture traffic without authorization.

## Availability

This repository is a public project overview, not an installer or a production release. Installation instructions, source-code release terms and support arrangements are being reviewed.

## Research

See [Research overview](research/README.md). Research benchmarks evaluate forecasting under defined conditions; they do not measure the app's ability to protect an organization.
