from __future__ import annotations
import json
import torch
import torch.nn.functional as F
try:
    import cv2
except Exception:
    cv2 = None

def _align_ceil(value, stride):
    return max(stride, -(-int(value) // int(stride)) * int(stride))

def _align_box(x1, y1, x2, y2, width, height, stride):
    stride = max(1, int(stride))
    x1 = max(0, min(int(x1), width - stride))
    y1 = max(0, min(int(y1), height - stride))
    w = min(_align_ceil(x2 - x1, stride), width - x1)
    h = min(_align_ceil(y2 - y1, stride), height - y1)
    return x1, y1, x1 + w, y1 + h

def _green_foreground(image, green_threshold, green_dominance, cleanup):
    h, w = image.shape[1:3]
    fg = torch.zeros((h, w), dtype=torch.float32, device=image.device)
    for i in range(0, image.shape[0], 8):
        rgb = image[i:i+8].float().clamp(0, 1)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        green_score = g - (r + b) * 0.5
        is_green = (g >= float(green_threshold)) & (green_score >= float(green_dominance))
        fg = torch.maximum(fg, (~is_green).float().amax(0))
    fg = fg[None]
    if int(cleanup) > 0:
        k = int(cleanup) * 2 + 1
        fg = F.max_pool2d(fg[:, None], k, stride=1, padding=int(cleanup))[:, 0]
        fg = 1.0 - F.max_pool2d((1.0 - fg)[:, None], k, stride=1, padding=int(cleanup))[:, 0]
    fg = fg.clamp(0, 1)
    if cv2 is not None:
        binary = (fg[0].detach().cpu().numpy() > 0.5).astype("uint8")
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if count > 1:
            largest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
            fg = torch.from_numpy((labels == largest).astype("float32")).to(device=image.device)[None]
    return fg

def _salient_foreground(image, sensitivity, cleanup):
    h, w = image.shape[1:3]
    band = max(2, min(h, w) // 16)
    frame = torch.zeros((h, w), device=image.device, dtype=torch.float32)
    frame[:band, :] = 1.0; frame[-band:, :] = 1.0; frame[:, :band] = 1.0; frame[:, -band:] = 1.0
    border_n = frame.sum().clamp_min(1.0)
    rgb0 = image[0:1].float().clamp(0, 1)
    bg0 = (rgb0 * frame[None, :, :, None]).sum((1, 2)) / border_n
    peak = float((rgb0 - bg0[:, None, None, :]).pow(2).sum(-1).sqrt().amax().clamp_min(1e-4))
    thresh = max(0.05, 0.75 - 0.7 * float(sensitivity)) * peak
    fg = torch.zeros((h, w), dtype=torch.float32, device=image.device)
    for i in range(0, image.shape[0], 8):
        rgb = image[i:i+8].float().clamp(0, 1)
        bg = (rgb * frame[None, :, :, None]).sum((1, 2)) / border_n
        score = (rgb - bg[:, None, None, :]).pow(2).sum(-1).sqrt()
        fg = torch.maximum(fg, (score > thresh).float().amax(0))
    fg = fg[None]
    if int(cleanup) > 0:
        k = int(cleanup) * 2 + 1
        fg = F.max_pool2d(fg[:, None], k, stride=1, padding=int(cleanup))[:, 0]
        fg = 1.0 - F.max_pool2d((1.0 - fg)[:, None], k, stride=1, padding=int(cleanup))[:, 0]
    fg = fg.clamp(0, 1)
    if cv2 is not None:
        binary = (fg[0].detach().cpu().numpy() > 0.5).astype("uint8")
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if count > 1:
            largest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
            fg = torch.from_numpy((labels == largest).astype("float32")).to(device=image.device)[None]
    return fg

def _prepare_subject_mask(subject_mask, height, width, device, cleanup):
    if subject_mask is None:
        return None
    if not isinstance(subject_mask, torch.Tensor):
        raise ValueError("subject_mask must be a MASK tensor")
    m = subject_mask.float().to(device=device)
    if m.ndim == 2:
        m = m[None]
    if m.ndim != 3:
        raise ValueError("subject_mask must have shape [B,H,W] or [H,W]")
    m = m.amax(0, keepdim=True)[:, None]
    if tuple(m.shape[-2:]) != (height, width):
        m = F.interpolate(m, size=(height, width), mode="bilinear", align_corners=False)
    m = m[:, 0].clamp(0, 1)
    if int(cleanup) > 0:
        k = int(cleanup) * 2 + 1
        m = F.max_pool2d(m[:, None], k, stride=1, padding=int(cleanup))[:, 0]
        m = 1.0 - F.max_pool2d((1.0 - m)[:, None], k, stride=1, padding=int(cleanup))[:, 0]
    if cv2 is not None:
        binary = (m[0].detach().cpu().numpy() > 0.5).astype("uint8")
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if count > 1:
            largest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
            m = torch.from_numpy((labels == largest).astype("float32")).to(device=device)[None]
    return m.clamp(0, 1)

def _manual_box(width, height, left, top, right, bottom, stride):
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("ROI bounds are invalid")
    return _align_box(round(width*left), round(height*top), round(width*right), round(height*bottom), width, height, stride)

def plan_roi(image, mode, left, top, right, bottom, margin_x, margin_y, green_threshold, green_dominance, cleanup, stride, subject_mask=None):
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError("KineticAnchor ROI Planner expects IMAGE shaped [B,H,W,3]")
    _, height, width, _ = image.shape
    if mode == "subject_mask":
        fg = _prepare_subject_mask(subject_mask, height, width, image.device, cleanup)
        if fg is None:
            raise ValueError("subject_mask mode requires a connected MASK input")
        ys, xs = torch.where(fg[0] > 0.5)
        if len(xs) == 0:
            box = _manual_box(width, height, left, top, right, bottom, stride)
            detected = False
        else:
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            box = _align_box(x1 - round(width*margin_x), y1 - round(height*margin_y), x2 + round(width*margin_x), y2 + round(height*margin_y), width, height, stride)
            detected = True
    elif mode == "green_auto":
        fg = _green_foreground(image, green_threshold, green_dominance, cleanup)
        ys, xs = torch.where(fg[0] > 0.5)
        if len(xs) == 0:
            box = _manual_box(width, height, left, top, right, bottom, stride)
            detected = False
        else:
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            box = _align_box(x1 - round(width*margin_x), y1 - round(height*margin_y), x2 + round(width*margin_x), y2 + round(height*margin_y), width, height, stride)
            detected = True
    else:
        box = _manual_box(width, height, left, top, right, bottom, stride)
        fg = torch.ones((1, height, width), dtype=torch.float32, device=image.device)
        detected = False
    x1, y1, x2, y2 = box
    roi_mask = fg[:, y1:y2, x1:x2].contiguous()
    preview = image[:1].clone()
    edge = max(2, min(8, width // 128))
    red = torch.tensor([1.0, 0.1, 0.05], device=image.device, dtype=preview.dtype)
    preview[:, y1:y1+edge, x1:x2, :] = red
    preview[:, y2-edge:y2, x1:x2, :] = red
    preview[:, y1:y2, x1:x1+edge, :] = red
    preview[:, y1:y2, x2-edge:x2, :] = red
    context = {"version": 3, "mode": mode, "subject_mask_used": mode == "subject_mask", "source_size": [int(width), int(height)], "box": [x1,y1,x2,y2], "crop_size": [x2-x1,y2-y1], "stride": int(stride), "area_ratio": ((x2-x1)*(y2-y1))/(width*height), "green_detected": detected}
    crop = image[:, y1:y2, x1:x2, :].contiguous()
    return crop, roi_mask, preview, json.dumps(context, ensure_ascii=False, separators=(",", ":")), int(x2-x1), int(y2-y1)
