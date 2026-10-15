"""H3 mask tracking nodes: one-stop subject/lip region masks for masked generation.

Compresses the CLIPTextEncode -> SAM3_Detect -> SAM3_VideoTrack -> SAM3_TrackToMask
-> GrowMask graph into single nodes. Mask semantics follow H3 noise_mask:
1 = regenerate, 0 = keep original pixels.
"""

import torch
import torch.nn.functional as F

from comfy_extras.nodes_mask import GrowMask

try:
    from comfy_extras.nodes_sam3 import SAM3_Detect, SAM3_VideoTrack, SAM3_TrackToMask
except ImportError:
    SAM3_Detect = SAM3_VideoTrack = SAM3_TrackToMask = None


def _feather_masks(masks, feather):
    sigma = feather / 3.0
    radius = max(1, int(round(sigma * 3.0)))
    xs = torch.arange(-radius, radius + 1, device=masks.device, dtype=torch.float32)
    k = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
    k = (k / k.sum()).view(1, 1, 1, -1)
    m = masks[:, None].to(torch.float32)
    m = F.conv2d(F.pad(m, (radius, radius, 0, 0), mode="replicate"), k)
    m = F.conv2d(F.pad(m, (0, 0, radius, radius), mode="replicate"), k.view(1, 1, -1, 1))
    return m[:, 0].to(masks.dtype)


def _track_masks(images, sam3_model, clip, prompt, threshold, detect_interval, max_objects, grow, feather, invert, extra_mask=None):
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    initial = SAM3_Detect.execute(model=sam3_model, image=images[0:1], conditioning=cond,
                                  threshold=threshold).result[0]
    track = SAM3_VideoTrack.execute(images=images, model=sam3_model, initial_mask=initial,
                                    conditioning=cond, detection_threshold=threshold,
                                    max_objects=max_objects, detect_interval=detect_interval).result[0]
    masks = SAM3_TrackToMask.execute(track_data=track, object_indices="").result[0]  # (N,H,W)
    if grow:
        masks = GrowMask.execute(mask=masks, expand=grow, tapered_corners=True).result[0]
    if feather > 0:
        # 软边让遮罩边界在latent混合时渐变，消除生成区与保留区的锐度断层
        masks = _feather_masks(masks, feather)
    if invert:
        masks = 1.0 - masks
    if extra_mask is not None:
        e = extra_mask.reshape(-1, *extra_mask.shape[-2:]).to(masks)
        if e.shape[0] not in (1, masks.shape[0]):
            e = e.amax(0, keepdim=True)
        if e.shape[0] == 1 and masks.shape[0] != 1:
            e = e.expand_as(masks)
        masks = torch.maximum(masks, e)
    return masks


class KineticH3SubjectTrack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "视频全部帧（VHS_LoadVideo / LoadVideo 的 IMAGE 输出）"}),
                "sam3_model": ("MODEL", {"tooltip": "CheckpointLoaderSimple 加载 sam3.1_multiplex_fp16.safetensors"}),
                "clip": ("CLIP", {"tooltip": "同一个 SAM3 checkpoint 的 CLIP 输出"}),
                "prompt": ("STRING", {"default": "a person", "tooltip": "追踪目标（开放词汇），如 a person / the man in white"}),
                "threshold": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grow": ("INT", {"default": 12, "min": -256, "max": 256, "tooltip": "遮罩外扩/内缩像素。正=外扩（推荐，边缘留融合余量），负=内缩"}),
                "feather": ("INT", {"default": 8, "min": 0, "max": 64, "tooltip": "边缘羽化宽度（像素）。生成区与保留区锐度不一致时调大，边界渐变融合"}),
                "invert": ("BOOLEAN", {"default": False, "label_on": "只对遮罩外生成", "label_off": "只对遮罩内生成"}),
                "detect_interval": ("INT", {"default": 4, "min": 1, "max": 12, "tooltip": "每N帧重新检测一次；1最准最慢，快速运动场景调小"}),
                "max_objects": ("INT", {"default": 1, "min": 0, "max": 16, "tooltip": "最多追踪几个目标；0=内部上限64"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0, "tooltip": "mask_video预览的帧率，与源视频一致即可"}),
            },
            "optional": {
                "extra_mask": ("MASK", {"tooltip": "额外前景/背景遮罩，与追踪结果取并集（逐帧或单张均可）"}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "VIDEO")
    RETURN_NAMES = ("mask", "isolated_video", "mask_video")
    FUNCTION = "track"
    CATEGORY = "KineticAnchor-H3"
    DESCRIPTION = "自动追踪视频每一帧的人物遮罩（SAM3），带外扩/内缩、内/外生成选择和前景/背景补充遮罩。isolated_video=主体隔离到黑底，可直接作 ref_video_1；mask_video=逐帧遮罩预览，直接接 SaveVideo 核对追踪。"

    def track(self, images, sam3_model, clip, prompt, threshold, grow, feather, invert,
              detect_interval, max_objects, fps, extra_mask=None):
        masks = _track_masks(images, sam3_model, clip, prompt, threshold, detect_interval,
                             max_objects, grow, feather, invert, extra_mask)
        isolated = images * masks[..., None].to(images.dtype)
        mask_rgb = masks.clamp(0.0, 1.0)[..., None].repeat(1, 1, 1, 3).to(images.dtype)
        mask_video = {"images": mask_rgb, "fps": fps, "audio": None}
        return (masks, isolated, mask_video)


class KineticH3LipTrack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "视频全部帧"}),
                "sam3_model": ("MODEL", {"tooltip": "CheckpointLoaderSimple 加载 sam3.1_multiplex_fp16.safetensors"}),
                "clip": ("CLIP",),
                "prompt": ("STRING", {"default": "mouth", "tooltip": "口型重绘区域；可试 lips / mouth and chin"}),
                "threshold": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grow": ("INT", {"default": 8, "min": -64, "max": 64, "tooltip": "小区域建议外扩8~16像素保证融合"}),
                "feather": ("INT", {"default": 6, "min": 0, "max": 64, "tooltip": "边缘羽化宽度（像素）"}),
                "invert": ("BOOLEAN", {"default": False, "label_on": "只对遮罩外生成", "label_off": "只对遮罩内生成"}),
                "detect_interval": ("INT", {"default": 1, "min": 1, "max": 12, "tooltip": "口部运动快，建议1~2帧一检"}),
                "max_objects": ("INT", {"default": 1, "min": 0, "max": 16}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0, "tooltip": "mask_video预览帧率"}),
            },
        }

    RETURN_TYPES = ("MASK", "VIDEO")
    RETURN_NAMES = ("mask", "mask_video")
    FUNCTION = "track"
    CATEGORY = "KineticAnchor-H3"
    DESCRIPTION = "自动追踪嘴部（+下巴）区域，输出口型重绘遮罩。接 SetLatentNoiseMask 到视频流即可只重绘口型区；mask_video 接 SaveVideo 核对追踪。"

    def track(self, images, sam3_model, clip, prompt, threshold, grow, feather, invert,
              detect_interval, max_objects, fps):
        masks = _track_masks(images, sam3_model, clip, prompt, threshold, detect_interval,
                             max_objects, grow, feather, invert)
        mask_rgb = masks.clamp(0.0, 1.0)[..., None].repeat(1, 1, 1, 3)
        return (masks, {"images": mask_rgb, "fps": fps, "audio": None})


NODE_CLASS_MAPPINGS = {
    "KineticH3SubjectTrack": KineticH3SubjectTrack,
    "KineticH3LipTrack": KineticH3LipTrack,
} if SAM3_Detect is not None else {}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KineticH3SubjectTrack": "H3 人物全帧追踪遮罩（角色替换）",
    "KineticH3LipTrack": "H3 口型区域追踪遮罩（对口型）",
} if SAM3_Detect is not None else {}
