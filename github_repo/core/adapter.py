from __future__ import annotations
import torch
import torch.nn.functional as F
from .roi import _align_ceil

def _fit_size(width, height, target_mp, stride):
    stride = max(8, int(stride))
    if float(target_mp) <= 0:
        return _align_ceil(int(width), stride), _align_ceil(int(height), stride)
    target_area = max(float(target_mp), 0.05) * 1000000.0
    scale = min(1.0, (target_area / max(1.0, width * height)) ** 0.5)
    out_w = max(stride, int(width * scale) // stride * stride)
    out_h = max(stride, int(height * scale) // stride * stride)
    return out_w, out_h

def _resize_image(image, width, height):
    if image.shape[1] == height and image.shape[2] == width:
        return image.contiguous()
    x = image.movedim(-1, 1)
    x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    return x.movedim(1, -1).contiguous()
