# Research overview

Adaptive Hierarchical Event Memory (AHEM) tests whether a small model can use a
flow's history to predict the size and arrival delay of its next packet. Each
prediction uses only packets that have already arrived. The model gets no
labels, future packets, or information about where the capture ends.

## Why forecast traffic at all

Most small organizations cannot afford full packet retention, and much of what
crosses a modern network is encrypted. Flow metadata — size, timing, direction,
protocol shape — survives both constraints. If a model can flag unusual packet
size or timing shifts within a flow, that is a triage cue for an analyst, not
a security verdict.

## 1. Original frozen experiment (CSE-CIC-IDS2018 subset)

The original primary evaluation used the Friday, 23 February 2018 capture from
CSE-CIC-IDS2018. It scored 512 selected next-same-flow prediction points with
eligible older memory across three model seeds, using weighted Smooth L1 loss:

| Model | Trainable parameters | Mean weighted loss across seeds |
| --- | ---: | ---: |
| AHEM v6.1 | 40,851 | 0.248937 |
| Flow-lane LSTM | 41,142 | 0.435422 |

The recorded equal-seed mean relative loss reduction was **42.81%** (and
**40.57%** against a secondary post-first-access dual-lane LSTM control). That
experiment had strict limits: it covered returning flows with eligible memory
rather than cold-start or one-shot packets, used one test day from one dataset
family, and measured forecast loss rather than attack detection.

## 2. Unified matched zero-shot baselines (Nov 2026 — honest negative result)

To test generalization without hidden fallback masking (`0%` fallback), we ran a
unified single-threaded benchmark comparing three zero-parameter online causal
predictors (`split_cell_v1`, `earned_v1`, `ewma`) against frozen 3-seed
`AHEM-v6.1` (`40,851` params) and frozen 3-seed `dual_lane_lstm` (`40,855` params)
across two local development journals (`T08`/`T09`) and three external PCAP
benchmarks (`bigFlows`, `UNSW-SOSR`, `GothamDataset2025`). Native Rust latency
is measured via in-process batched timing (`N = 20,000` packets/batch, 50
batches, `std::hint::black_box`, reporting **batch-mean p50 / p99** rather than
individual per-packet tail spikes) across both L2-resident (`4,096` lanes) and
L3-spill (`65,536` lanes) working sets
(`research/evidence/ahem-unified-baselines-20261126.json`):

| Arm | Params | Single-thread CPU latency (`4,096` / `65,536` lanes batch-mean p50, or e2e p50) | T08/T09 `all` Loss | `bigFlows` `uniform` Loss | `UNSW-SOSR` `uniform` Loss | `Gotham` Macro Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `split_cell_v1` (Rust) | 0 | **`25.6 ns` / `55.1 ns`** batch-mean p50 (`31.7 / 89.9 ns` batch-mean p99; `19.96 us/pkt` subproc) | **0.01059** | 0.07885 | 0.09437 | 0.05652 |
| `earned_v1` (Rust) | 0 | **`27.9 ns` / `73.1 ns`** batch-mean p50 (`42.1 / 110.7 ns` batch-mean p99; `22.35 us/pkt` subproc) | 0.01292 | **0.07668** | **0.08916** | — |
| `ewma` (Rust) | 0 | **`15.9 ns` / `25.7 ns`** batch-mean p50 (`19.4 / 35.5 ns` batch-mean p99; `22.35 us/pkt` subproc) | 0.02268 | 0.07873 | 0.09446 | **0.04961** |
| `AHEM-v6.1` (`forecast_only` / 3-seed) | 40,851 | **`3.225 ms`** / `15.049 ms` e2e (`0.492 ms` `_read`; `15.2 us` native ctypes) | 0.13348 / 0.16377 | 0.13339 / 0.13965 | 0.18160 / 0.19275 | 0.07184 |
| `dual_lane_lstm` (3-seed mean) | 40,855 | `1.465 ms` / `4.580 ms` e2e | 0.47500 | 0.30552 | 0.44012 | 0.27697 |

**Key takeaway:** Zero-parameter online causal predictors (`split_cell_v1`,
`earned_v1`, `ewma`) decisively beat both frozen neural models on **every**
tested dataset and stratum (including `first_four` cold starts and `long_gap`
interleaving). The latency gap against batch=1 PyTorch (`3.225 ms`) splits into
roughly `210x` Python/PyTorch dispatch overhead (`3.225 ms` vs `15.2 us` native
ctypes AHEM) and `275x`–`590x` algorithmic complexity (`15.2 us` vs `25.6–55.1 ns`
native `split_cell_v1`). `AHEM-v6.1` outperforms `dual_lane_lstm` across every
capture, showing better shift resilience than an LSTM without clearing the
zero-parameter baseline floor.

## 3. CTU IoT-23 IDS detection bridge & CIC-IDS2017 boundary

We tested whether frozen `AHEM-v6.1` forecast residuals or hidden states add
packet-time attack-detection value on official CTU IoT-23 captures (`8-1`:
`2,684` events; `21-1`: `27,352` events; benign calibration `4-1`: `6,503`
events; tracked in `research/evidence/iot23-ids-detection-bridge-20261126.json`):

- **Direction A (`8-1` fit + `4-1` cal $\to$ `21-1` test):** A 14-feature
  causal logistic probe (`base_logreg`) achieved **`0.999219` PR-AUC /
  `0.973240` ROC-AUC / `0.000000` FPR@95%TPR**, beating `base_plus_ahem_v61_residual`
  (`0.998246` PR-AUC / `0.940444` ROC-AUC / `0.710425` FPR@95%TPR),
  `base_plus_lstm_residual` (`0.981668` PR-AUC), `base_hgb` (`0.978695`), and
  `base_plus_ewma_residual` (`0.919915`). In the 120-dim state ablation,
  `ahem_v61_state` (`0.976940` PR-AUC, `0.712314` ROC-AUC) trailed
  `memory_disabled_state` (`0.992305` PR-AUC, `0.867390` ROC-AUC).
- **Direction B (`21-1` fit + `4-1` cal $\to$ `8-1` test):** All 8 arms saturated
  at `1.0000` PR-AUC / `1.0000` ROC-AUC / `0.0000` FPR@95%TPR.
- **Operating-point calibration:** Benign `4-1` thresholding (`<= 1.0 FA/h`)
  failed closed at `0.0 FA/h` and `0.0` recall in both directions due to
  probability saturation on unseen honeypot traffic.
- **CIC-IDS2017 status:** Full 48 GB AWS CIC-IDS2017 and CSE-CIC-IDS2018 raw
  PCAP archives are not staged locally, so CIC-IDS2017 detection bridge
  evaluation has not been run and no CIC-IDS2017 detection results are claimed.

## 4. Compact router right-sizing & candidate retirements

- **Compact router (`compact-router-v1`, `4,591 B`):** Replaced the quarantined
  421M-parameter fp16 `Laya v2` transformer (`842.6 MB` weights, `61.38 ms`
  median GPU latency, `831.39 MiB` peak VRAM, never evaluated on the locked test
  split) with `research/evidence/compact-router-v1.json` (`argus.compact-router/v1`):
  a **`4,591 B` (`4.483 KB`)** pure-JSON 29-node decision tree + temperature-`1.04`
  logistic abstain head executed in pure Python (`python/src/argus_research/compact_router.py`,
  zero PyTorch/GPU, SHA-256 pinned, fail-closed fallback to `lstm_cse`). The
  decision head runs in **`4.3 us` CPU median** (`8.2 us` p95, `0 MB` VRAM) on
  pre-extracted 60-dim features (`35.4 us` validated `route_features()`), while
  end-to-end pure-Python `route_window()` including 128-packet extraction runs in
  **`0.97 ms` median** (`1.46 ms` p95; ~63x faster than GPU `Laya v2`). On the
  1,000-row source-disjoint locked test ($G=3$ days: `2026-09-17`/`19`/`20`,
  small-sample indicative; tracked in
  `research/evidence/compact-router-locked-test-20261126.json`), it achieved
  `0.6700` selected accuracy (`0.7002` when non-abstaining), `+0.002023` utility
  gain over best-fixed `lstm_cse` (95% CI `[+0.000363, +0.002976]`), and abstain
  AUROC `0.7389` (abstain gating stays diagnostic because enabling
  `fallback_on_abstain=True` reduces locked-test utility gain to `+0.001561`).
- **Retired & non-confirmatory records:** The incomplete 921,229-parameter
  `step-0003808.pt` checkpoint (retired; up to `95.9%` hidden fallback masking) is
  formally excluded and removed from public `models/`. An earlier post-hoc
  control study (`ahem-cost-context-20261004`) was marked non-confirmatory due
  to `select_blend` validation leakage and remediated with disjoint split guards.

## The artifacts

The public repository ships the frozen `AHEM-v6.1` research exports and
manifests (`models/`, hash-verified before loading; validated in fail-closed
standby by `NativeOnnxWorker` while native Rust `split_cell_v1` runs live per
packet without requiring `onnxruntime.dll` or Python) and tracked JSON
evaluation artifacts under `research/evidence/`. The CI/GitHub Release package
(`scripts/package-windows.ps1`) bundles these frozen weights and Python sources,
whereas `scripts/build-windows-release.ps1` provides an optional weightless
portable build (native executables, scripts, and license notices only, with
`-IncludeModels` opt-in). Terms are in [models/NOTICE.md](../models/NOTICE.md),
and runtime boundaries are in [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

## What is public, and what is not

Published: the source, frozen `AHEM-v6.1` inference weights and manifests,
tracked JSON evidence artifacts (`research/evidence/`), clean-room spatial UI
assets (`Apache-2.0` / `CC0-1.0`), the roadmap, architecture decisions, and the
honest positive and negative findings above.

Not published: raw training captures, copies of third-party datasets, the full
private training harness, or private working logs.

For questions about the method or results, open an issue.
