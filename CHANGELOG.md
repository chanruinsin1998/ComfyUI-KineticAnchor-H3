# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-10-15

Initial public release.

### Added

- Community nodes: `KineticAnchorGreenSubjectMask`, `KineticAnchorROIPlanner`, `KineticAnchorLatentActivityCanvas`, `KineticAnchorTemporalPaste`.
- SAM3 tracking nodes: `KineticH3SubjectTrack`, `KineticH3LipTrack`. Nodes are skipped without error when SAM3 nodes are unavailable.
- Built-in detection modes on the activity canvas: green screen heuristic, border-based saliency, and manual bounds.
- Context-driven paste-back with color matching, cosine-window feathering, and motion-gated blending.
- `target_megapixels` match-source semantics: values at or below zero keep the crop size as the model size.
- Example workflows for character replacement and lip re-sync.
- Offline regression tests (`test_nodes.py`).
