from __future__ import annotations
import torch
import torch.nn.functional as F

def _normalize_mask(mask, height, width, device):
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise ValueError("mask input must be a MASK tensor")
    m = mask.float().to(device=device)
    if m.ndim == 2: m = m[None]
    if m.ndim != 3: raise ValueError("mask must have shape [B,H,W] or [H,W]")
    m = m.amax(0, keepdim=True)[:, None]
    if tuple(m.shape[-2:]) != (height, width):
        m = F.interpolate(m, size=(height, width), mode="bilinear", align_corners=False)
    return m[:, 0].clamp(0, 1)

def _largest_component(mask):
    try:
        import cv2
    except Exception:
        return mask
    binary = (mask[0].detach().cpu().numpy() > 0.5).astype("uint8")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1: return mask
    largest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    return torch.from_numpy((labels == largest).astype("float32")).to(device=mask.device)[None]

def _subject_region(mask, height, width, ratio_x, ratio_y):
    m = _largest_component(mask)
    ys, xs = torch.where(m[0] > 0.5)
    if len(xs) == 0: return torch.zeros((1, height, width), device=mask.device), None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    dx = round((x2 - x1) * float(ratio_x))
    dy = round((y2 - y1) * float(ratio_y))
    box = [max(0, x1-dx), max(0, y1-dy), min(width, x2+dx), min(height, y2+dy)]
    out = torch.zeros((1, height, width), device=mask.device)
    out[:, box[1]:box[3], box[0]:box[2]] = 1.0
    return out, box

def _support_region(mask, height, width, margin_px):
    m = mask
    k = max(0, int(margin_px))
    if k > 0:
        kernel = min(2*k+1, min(height, width) if min(height,width)%2==1 else min(height,width)-1)
        pad = kernel//2
        m = F.max_pool2d(m[:, None], kernel, stride=1, padding=pad)[:, 0]
    return m.clamp(0, 1)
