# ComfyUI-KineticAnchor-H3

**English** | [简体中文](README.zh-CN.md)

Region-of-interest (ROI) canvas reduction and paste-back nodes for MiniMax H3 video generation in ComfyUI, plus single-node SAM3 subject and lip mask tracking for masked regeneration workflows.

## Overview

MiniMax H3 video models run on a rectangular latent canvas. When the subject covers only a small part of the frame, generating the full canvas spends compute on static background. KineticAnchor-H3 reduces the canvas to a stride-aligned region around the active subject, generates video on the reduced canvas, and pastes the result back into the original frame with color matching and feathered blending.

The paste-back path is context-driven: nodes exchange crop geometry through a JSON context string, so no manual coordinate wiring is required.

A second group of nodes compresses the SAM3 detection → tracking → mask conversion chain into single nodes. They emit masks following the H3 `noise_mask` convention (`1` = regenerate, `0` = keep original pixels), which ComfyUI core implements natively for H3 (`MiniMaxH3.scale_latent_inpaint`).

## Requirements

- A recent ComfyUI build with MiniMax H3 support.
- SAM3 nodes (`comfy_extras.nodes_sam3`) for the two tracking nodes. If SAM3 is unavailable, the tracking nodes are not registered and the remaining four nodes load normally.
- No Python dependencies beyond the ComfyUI environment.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-username>/ComfyUI-KineticAnchor-H3.git
```

Restart ComfyUI. Alternatively, download the repository as a ZIP archive, extract it into `ComfyUI/custom_nodes`, and restart ComfyUI.

## Nodes

All classes are registered under the `KineticAnchor` menu categories.

| Class | Category | Function |
|---|---|---|
| `KineticAnchorGreenSubjectMask` | `KineticAnchor/Mask` | Full-canvas green screen subject mask. |
| `KineticAnchorROIPlanner` | `KineticAnchor/ROI` | ROI planning with green-auto, external-mask, and manual modes; stride-aligned outputs. |
| `KineticAnchorLatentActivityCanvas` | `KineticAnchor/Canvas` | Parameterized activity canvas. Built-in green screen / saliency detection, up to four subject masks with independent expansion margins, support and candidate masks, manual and fallback bounds. Exposes 12 discrete outputs, including the local conditioning frames and the paste context. |
| `KineticAnchorTemporalPaste` | `KineticAnchor/Temporal` | Scales the generated ROI video back to the source ROI size and pastes it into the anchor frames with color matching, cosine-window feathering, and motion-gated blending. |
| `KineticH3SubjectTrack` | `KineticAnchor-H3` | SAM3 full-frame subject tracking for character replacement. Outputs the tracking mask, the isolated subject on black background (usable as `ref_video_1`), and a mask preview video. |
| `KineticH3LipTrack` | `KineticAnchor-H3` | SAM3 mouth-region tracking for lip re-sync. Outputs the lip-region mask and a mask preview video. |

## Usage

### ROI canvas workflow (image to video)

```text
LoadImage -> Latent Activity Canvas
  detection_mode = "Auto Green Screen"   (or "Auto Salient", or external masks)
  local_first_frame / local_last_frame / model_width / model_height -> H3 FL2VA conditioning
  H3 AV decode -> Temporal Paste.generated_roi
  LoadImage -> Temporal Paste.anchor_image
  Canvas.kinetic_context -> Temporal Paste.kinetic_context
  Temporal Paste -> SaveVideo
```

Generated video that differs in size from the source ROI is resized automatically; the local mask is resized with it. Setting `target_megapixels` to `0` keeps the crop size as the model size (match-source semantics).

### Masked regeneration (character replacement, lip re-sync)

```text
VHS_LoadVideo -> KineticH3SubjectTrack (or KineticH3LipTrack)
  sam3_model / clip <- CheckpointLoaderSimple (SAM3 checkpoint)
  mask -> SetLatentNoiseMask on the video latent stream
  isolated_video -> MiniMaxH3ReferenceToVideo.ref_video_1   (character replacement)
```

Mask semantics follow the H3 noise mask convention: `1` marks pixels to regenerate, `0` marks pixels to keep. Retained regions are injected at the H3 conditioning timestep by ComfyUI core; video and audio streams carry independent denoise masks.

Parameters shared by both tracking nodes:

| Parameter | Description |
|---|---|
| `grow` | Outward (positive) or inward (negative) mask expansion, in pixels. |
| `feather` | Edge feathering width in pixels; softens the transition between regenerated and retained regions. |
| `invert` | Inverts the mask; generate outside the tracked region instead of inside. |
| `detect_interval` | Re-detection interval in frames. Lower values track fast motion more accurately. |
| `max_objects` | Maximum number of tracked targets. `0` uses the internal limit. |
| `extra_mask` | Optional mask unioned with the tracking result (subject tracking only). |

## Examples

| Workflow | Description |
|---|---|
| `examples/KineticAnchor_视频角色替换_遮罩引导简化版.json` | Character replacement using `KineticH3SubjectTrack`. |
| `examples/KineticAnchor_重新对口型.json` | Lip re-sync using `KineticH3LipTrack` with new audio. |
| `examples/KineticAnchor_视频角色替换_SAM3版.json` | Reference workflow assembled from raw SAM3 nodes; equivalent to what `KineticH3SubjectTrack` performs internally. |

The example workflows additionally require [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) and [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes).

## Testing

```bash
python test_nodes.py
```

Offline regression test covering detection, ROI alignment, context round-trip, and paste-back geometry on synthetic inputs. No models are loaded. Run from a ComfyUI installation so that `comfy_extras` is importable.

## Limitations

- H3 uses a rectangular latent canvas. Disjoint subject regions resolve to their union rectangle; masks protect the paste-back but do not remove attention inside the rectangle. Actual speedup depends on how far `model_width × model_height` falls below the full-frame baseline.
- Offline tests do not measure sampling speed. Compare full-canvas and ROI-canvas runs on the same GPU, model, seed, frame count, and step count to evaluate a real speedup.

## Studio Pro

A commercial edition (Studio Pro) is distributed separately. It adds a single-bus pipeline node pair, a browser-based activity region editor, and scene presets. This repository contains no commercial code.

## License

MIT. See [LICENSE](LICENSE).
