# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-12

First public release.

### Added
- Physics-based vehicle model (`canguard/vehicle.py`) and 500 kbit/s bus
  simulator with per-ECU clock jitter (`canguard/simulator.py`).
- Binary signal encoding with rolling counters and checksums
  (`canguard/signals.py`).
- Five attack families under two threat models - injection and masquerade
  (`canguard/attacks.py`).
- 67-feature windowed extractor spanning timing, integrity, physics and
  cross-ECU agreement (`canguard/features.py`).
- Four detectors: TensorFlow autoencoder, Isolation Forest, Random Forest and
  a TensorFlow MLP, plus a six-class attack-type classifier
  (`canguard/models.py`).
- Online detector with alarm debouncing (`scripts/5_live_detect.py`).
- Evaluation reporting overall metrics, per-attack recall, per-threat-model
  recall and five charts (`scripts/4_evaluate.py`).
- Deterministic end-to-end run via `run_all.py`, including TensorFlow op
  determinism, so published metrics reproduce exactly.
- Test suite covering signal round-trips, bus timing, feature sanity and
  determinism.

[1.0.0]: https://github.com/yourname/can-guard/releases/tag/v1.0.0
