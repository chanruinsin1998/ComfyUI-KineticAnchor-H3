"""Offline regression tests for the open-source nodes.

Loads no models. Run from the repository root inside a ComfyUI installation:

    python test_nodes.py
"""

import importlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))        # custom_nodes
sys.path.insert(0, str(ROOT.parent.parent))  # ComfyUI root (comfy_extras)

PACKAGE = ROOT.name
comm = importlib.import_module(f"{PACKAGE}.editions.community.nodes_community")
masks = importlib.import_module(f"{PACKAGE}.nodes_h3_masks")
pkg = importlib.import_module(PACKAGE)


def make_green_image():
    img = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
    img[..., 0] = 0.05
    img[..., 1] = 0.7
    img[..., 2] = 0.05
    img[0, 60:200, 90:170, :] = torch.tensor([0.75, 0.3, 0.2])
    return img


def main():
    img = make_green_image()

    subject = comm.KineticAnchorGreenSubjectMask().execute(img, 0.35, 0.08, 2)[0]
    assert subject.dim() == 3 and subject.shape[-2:] == (256, 256)
    assert 0.0 < subject.sum().item() < subject.numel()

    canvas = comm.KineticAnchorLatentActivityCanvas()
    out = canvas.execute(img, "Auto Green Screen", 0.35, 0.08, 2, 0.5, 0.2, 32,
                         0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12,
                         32, 0.12, 0.12, False, False,
                         0.05, 0.03, 0.95, 0.97, 0.05, 0.03, 0.95, 0.97)
    assert len(out) == 12
    _, first, _, _, _, _, _, _, preview, context_text, mw, mh = out
    assert first.shape == (1, mh, mw, 3) and preview.shape == (1, 256, 256, 3)
    ctx = json.loads(context_text)
    assert ctx["version"] == 6 and ctx["regions"][0]["source"] == "auto_green"

    paste = comm.KineticAnchorTemporalPaste()
    full = paste.execute(first, img, context_text, 8)[0]
    assert full.shape == img.shape

    planner_out = comm.KineticAnchorROIPlanner().execute(
        img, "green_auto", 0.05, 0.03, 0.95, 0.97, 0.04, 0.04, 0.35, 0.08, 2, 32)
    assert planner_out[4] > 0 and planner_out[3]

    assert isinstance(masks.NODE_CLASS_MAPPINGS, dict)
    assert len(pkg.NODE_CLASS_MAPPINGS) == 4 + len(masks.NODE_CLASS_MAPPINGS)
    assert set(pkg.NODE_CLASS_MAPPINGS) >= {
        "KineticAnchorGreenSubjectMask",
        "KineticAnchorROIPlanner",
        "KineticAnchorLatentActivityCanvas",
        "KineticAnchorTemporalPaste",
    }

    print(json.dumps({"PASS": True, "nodes": sorted(pkg.NODE_CLASS_MAPPINGS),
                      "context_box": list(ctx["box"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
