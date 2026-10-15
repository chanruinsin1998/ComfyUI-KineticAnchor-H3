from __future__ import annotations
import json
import torch
import torch.nn.functional as F

def _cosine_alpha(h, w, feather, device):
    alpha = torch.ones((h,w), dtype=torch.float32, device=device)
    feather = max(0, min(int(feather), h//2, w//2))
    if feather == 0: return alpha
    ramp = 0.5 * (1.0 - torch.cos(torch.linspace(0.0, 3.14159265, feather, device=device)))
    alpha[:feather,:] *= ramp[:,None]
    alpha[-feather:,:] *= ramp.flip(0)[:,None]
    alpha[:,:feather] *= ramp[None,:]
    alpha[:,-feather:] *= ramp.flip(0)[None,:]
    return alpha


def _match_edge_color(crop, anchor_roi, band, strength=0.35):
    band = max(1, min(int(band), crop.shape[1] // 2, crop.shape[2] // 2))
    crop_edges = torch.cat((crop[:, :band].reshape(crop.shape[0], -1, crop.shape[-1]), crop[:, -band:].reshape(crop.shape[0], -1, crop.shape[-1]), crop[:, :, :band].reshape(crop.shape[0], -1, crop.shape[-1]), crop[:, :, -band:].reshape(crop.shape[0], -1, crop.shape[-1])), dim=1)
    anchor_edges = torch.cat((anchor_roi[:, :band].reshape(anchor_roi.shape[0], -1, anchor_roi.shape[-1]), anchor_roi[:, -band:].reshape(anchor_roi.shape[0], -1, anchor_roi.shape[-1]), anchor_roi[:, :, :band].reshape(anchor_roi.shape[0], -1, anchor_roi.shape[-1]), anchor_roi[:, :, -band:].reshape(anchor_roi.shape[0], -1, anchor_roi.shape[-1])), dim=1)
    crop_mean = crop_edges.mean(dim=1, keepdim=True).unsqueeze(1)
    anchor_mean = anchor_edges.mean(dim=1, keepdim=True).unsqueeze(1)
    crop_std = crop_edges.std(dim=1, keepdim=True).unsqueeze(1)
    anchor_std = anchor_edges.std(dim=1, keepdim=True).unsqueeze(1)
    if float(crop_std.max()) < 1e-4 or float(anchor_std.max()) < 1e-4: return crop
    adjusted = (crop - crop_mean) * (anchor_std / crop_std.clamp_min(1e-4)) + anchor_mean
    return torch.lerp(crop, adjusted.clamp(0, 1), float(strength)).clamp(0, 1)

def _coerce_mask(mask, batch, height, width, canvas_hw, x1, y1, x2, y2, device, dtype):
    m = mask.to(device=device, dtype=dtype) if isinstance(mask, torch.Tensor) else torch.as_tensor(mask, device=device, dtype=dtype)
    if m.ndim == 4:
        if m.shape[-1] == 1: m = m[..., 0]
        elif m.shape[1] == 1: m = m[:, 0]
        else: raise ValueError("MASK must be [B,H,W] or [H,W]")
    if m.ndim == 2: m = m[None]
    if m.ndim != 3: raise ValueError("MASK must be [B,H,W] or [H,W]")
    if tuple(m.shape[-2:]) == tuple(canvas_hw):
        m = m[:, y1:y2, x1:x2]
    if tuple(m.shape[-2:]) != (height, width):
        m = F.interpolate(m[:, None], size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    if m.shape[0] == 1 and batch > 1: m = m.expand(batch, -1, -1)
    elif m.shape[0] != batch:
        if m.shape[0] > batch: m = m[:batch]
        else: m = m[:1].expand(batch, -1, -1)
    return m.clamp(0, 1)

def _motion_alpha(crop, gate, blur):
    vmin = None; vmax = None
    for i in range(0, crop.shape[0], 32):
        c = crop[i:i+32]
        vmin = c.amin(0) if vmin is None else torch.minimum(vmin, c.amin(0))
        vmax = c.amax(0) if vmax is None else torch.maximum(vmax, c.amax(0))
    r = (vmax - vmin).mean(-1)
    m = ((r - gate) / gate).clamp(0, 1)[None, None]
    k = max(3, 2*int(blur)+1)
    m = F.max_pool2d(m, k, stride=1, padding=k//2)
    m = F.avg_pool2d(F.avg_pool2d(m, k, stride=1, padding=k//2), k, stride=1, padding=k//2)
    return m[0, 0]

def _unpack_pipe(kinetic_pipe):
    if isinstance(kinetic_pipe, dict):
        box = kinetic_pipe.get("crop_box") or (kinetic_pipe.get("context") or {}).get("box")
        mask = (kinetic_pipe.get("masks") or {}).get("roi_mask")
        return box, mask
    if isinstance(kinetic_pipe, str) and kinetic_pipe.strip():
        ctx = json.loads(kinetic_pipe)
        return ctx.get("box"), None
    raise ValueError("Temporal Paste requires a kinetic_pipe connection or legacy context JSON")

def paste_video(crop, anchor, kinetic_pipe, feather, mask=None, color_match=True, color_match_strength=0.35, motion_gate=0.08):
    if crop.ndim != 4 or anchor.ndim != 4: raise ValueError("Temporal Paste expects IMAGE tensors")
    box, pipe_mask = _unpack_pipe(kinetic_pipe)
    if not box or len(box) != 4:
        raise ValueError("KineticAnchor pipe has no valid crop box")
    x1,y1,x2,y2 = [int(v) for v in box]
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError("KineticAnchor context box is invalid")
    b,h,w,c = crop.shape
    target_w, target_h = x2 - x1, y2 - y1
    if (w, h) != (target_w, target_h):
        crop = F.interpolate(crop.movedim(-1, 1), size=(target_h, target_w), mode="bilinear", align_corners=False).movedim(1, -1).contiguous()
        h, w = target_h, target_w
    if anchor.shape[1] < y2 or anchor.shape[2] < x2: raise ValueError("Anchor image is smaller than ROI context")
    if anchor.shape[0] not in (1, b):
        raise ValueError("Anchor batch must be 1 or match generated video batch")
    base_anchor = anchor[:1] if anchor.shape[0] == 1 else anchor
    if anchor.shape[0] == 1:
        base = base_anchor.to(device=crop.device, dtype=crop.dtype).repeat(b, 1, 1, 1)
    else:
        base = base_anchor.to(device=crop.device, dtype=crop.dtype).clone()
    alpha = _cosine_alpha(h,w,feather,crop.device).to(crop.dtype)[None,:,:,None].expand(b,-1,-1,-1)
    if mask is None: mask = pipe_mask
    if mask is not None:
        m = _coerce_mask(mask, b, h, w, (base.shape[1], base.shape[2]), x1, y1, x2, y2, crop.device, crop.dtype)
        alpha = alpha * m[:b, :, :, None]
    roi = base[:,y1:y2,x1:x2,:]
    if color_match:
        crop = _match_edge_color(crop, roi, max(2, int(feather)), color_match_strength)
    if motion_gate > 0:
        alpha = alpha * _motion_alpha(crop, float(motion_gate), feather)[None,:,:,None]
    base[:,y1:y2,x1:x2,:] = crop*alpha + roi*(1.0-alpha)
    return base.contiguous()
