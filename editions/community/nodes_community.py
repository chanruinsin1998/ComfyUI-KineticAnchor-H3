from __future__ import annotations
from ...core.roi import plan_roi, _green_foreground
from ...core.paste import paste_video
from ...core.activity_canvas import build_activity_canvas

class KineticAnchorGreenSubjectMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "green_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}), "green_dominance": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}), "cleanup": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1})}}
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("full_subject_mask",)
    FUNCTION = "execute"
    CATEGORY = "KineticAnchor/Mask"
    def execute(self, image, green_threshold, green_dominance, cleanup):
        return (_green_foreground(image, green_threshold, green_dominance, cleanup),)

class KineticAnchorROIPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "mode": (["green_auto", "subject_mask", "manual"],), "left": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}), "top": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 1.0, "step": 0.01}), "right": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}), "bottom": ("FLOAT", {"default": 0.97, "min": 0.0, "max": 1.0, "step": 0.01}), "margin_x": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.5, "step": 0.01}), "margin_y": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.5, "step": 0.01}), "green_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}), "green_dominance": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}), "cleanup": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1}), "stride": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8})}, "optional": {"subject_mask": ("MASK",)}}
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("roi_image", "roi_mask", "roi_preview", "kinetic_context", "roi_width", "roi_height")
    FUNCTION = "execute"
    CATEGORY = "KineticAnchor/ROI"
    def execute(self, image, mode, left, top, right, bottom, margin_x, margin_y, green_threshold, green_dominance, cleanup, stride, subject_mask=None):
        return plan_roi(image, mode, left, top, right, bottom, margin_x, margin_y, green_threshold, green_dominance, cleanup, stride, subject_mask)

class KineticAnchorLatentActivityCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "detection_mode": (["Auto Green Screen", "Auto Salient", "Manual Full Box"],), "green_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}), "green_dominance": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}), "cleanup": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1}), "salient_sensitivity": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05}), "target_megapixels": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "局部画布兆像素预算（只缩不放）；0=Match Source：不缩放，直接用裁剪盒原始分辨率（按 stride 向上取整），无插值软化"}), "stride": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}), "subject_1_margin_x": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}), "subject_1_margin_y": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}), "subject_2_margin_x": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}), "subject_2_margin_y": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}), "subject_3_margin_x": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}), "subject_3_margin_y": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}), "subject_4_margin_x": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}), "subject_4_margin_y": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}), "activity_margin_px": ("INT", {"default": 32, "min": 0, "max": 512, "step": 4}), "candidate_ratio_x": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}), "candidate_ratio_y": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}), "use_candidate": ("BOOLEAN", {"default": False}), "use_manual_bounds": ("BOOLEAN", {"default": False}), "manual_left": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}), "manual_top": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 1.0, "step": 0.01}), "manual_right": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}), "manual_bottom": ("FLOAT", {"default": 0.97, "min": 0.0, "max": 1.0, "step": 0.01}), "fallback_left": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}), "fallback_top": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 1.0, "step": 0.01}), "fallback_right": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}), "fallback_bottom": ("FLOAT", {"default": 0.97, "min": 0.0, "max": 1.0, "step": 0.01})}, "optional": {"subject_mask_1": ("MASK",), "subject_mask_2": ("MASK",), "subject_mask_3": ("MASK",), "subject_mask_4": ("MASK",), "activity_mask_1": ("MASK",), "activity_mask_2": ("MASK",), "candidate_mask": ("MASK",)}}
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "MASK", "MASK", "MASK", "MASK", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("local_image", "local_first_frame", "local_last_frame", "subject_mask", "activity_mask", "candidate_mask", "protected_background", "roi_mask", "activity_canvas", "kinetic_context", "model_width", "model_height")
    FUNCTION = "execute"
    CATEGORY = "KineticAnchor/Canvas"
    def execute(self, image, detection_mode, green_threshold, green_dominance, cleanup, salient_sensitivity, target_megapixels, stride, subject_1_margin_x, subject_1_margin_y, subject_2_margin_x, subject_2_margin_y, subject_3_margin_x, subject_3_margin_y, subject_4_margin_x, subject_4_margin_y, activity_margin_px, candidate_ratio_x, candidate_ratio_y, use_candidate, use_manual_bounds, manual_left, manual_top, manual_right, manual_bottom, fallback_left, fallback_top, fallback_right, fallback_bottom, subject_mask_1=None, subject_mask_2=None, subject_mask_3=None, subject_mask_4=None, activity_mask_1=None, activity_mask_2=None, candidate_mask=None):
        local, first, last, pipe = build_activity_canvas(image, target_megapixels, stride, [subject_mask_1, subject_mask_2, subject_mask_3, subject_mask_4], [activity_mask_1, activity_mask_2], candidate_mask, use_candidate, [(subject_1_margin_x, subject_1_margin_y), (subject_2_margin_x, subject_2_margin_y), (subject_3_margin_x, subject_3_margin_y), (subject_4_margin_x, subject_4_margin_y)], activity_margin_px, candidate_ratio_x, candidate_ratio_y, manual_left, manual_top, manual_right, manual_bottom, use_manual_bounds, (fallback_left, fallback_top, fallback_right, fallback_bottom), detection_mode, green_threshold, green_dominance, cleanup, salient_sensitivity)
        masks = pipe["masks"]
        mw, mh = pipe["model_size"]
        return local, first, last, masks["subject"], masks["activity"], masks["candidate"], masks["protected_background"], masks["roi_mask"], pipe["preview"], pipe["context_json"], int(mw), int(mh)


class KineticAnchorTemporalPaste:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"generated_roi": ("IMAGE",), "anchor_image": ("IMAGE",), "kinetic_context": ("STRING",), "feather_pixels": ("INT", {"default": 16, "min": 0, "max": 128, "step": 4})}, "optional": {"roi_mask": ("MASK",), "color_match": ("BOOLEAN", {"default": True}), "color_match_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}), "motion_gate": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.3, "step": 0.01, "tooltip": "运动门控：生成ROI中帧间运动低于该阈值的像素回贴anchor原像素（抗静态区漂移），0=关闭"})}}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("full_video",)
    FUNCTION = "execute"
    CATEGORY = "KineticAnchor/Temporal"
    def execute(self, generated_roi, anchor_image, kinetic_context, feather_pixels, roi_mask=None, color_match=True, color_match_strength=0.35, motion_gate=0.08):
        return (paste_video(generated_roi, anchor_image, kinetic_context, feather_pixels, roi_mask, color_match, color_match_strength, motion_gate),)

NODE_CLASS_MAPPINGS = {"KineticAnchorGreenSubjectMask": KineticAnchorGreenSubjectMask, "KineticAnchorROIPlanner": KineticAnchorROIPlanner, "KineticAnchorLatentActivityCanvas": KineticAnchorLatentActivityCanvas, "KineticAnchorTemporalPaste": KineticAnchorTemporalPaste}
NODE_DISPLAY_NAME_MAPPINGS = {"KineticAnchorGreenSubjectMask": "KineticAnchor Green Subject Mask", "KineticAnchorROIPlanner": "KineticAnchor ROI Planner (Green Auto)", "KineticAnchorLatentActivityCanvas": "KineticAnchor Latent Activity Canvas", "KineticAnchorTemporalPaste": "KineticAnchor Temporal Paste"}
