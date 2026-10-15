from __future__ import annotations
import json
import torch
from .multi_roi import _normalize_mask, _subject_region, _support_region
from .adapter import _fit_size, _resize_image
from .roi import _align_box, _green_foreground, _salient_foreground

def build_activity_canvas(image, target_mp, stride, subject_masks, activity_masks, candidate_mask, use_candidate, subject_ratios, activity_margin_px, candidate_ratio_x, candidate_ratio_y, manual_left, manual_top, manual_right, manual_bottom, use_manual_bounds, fallback_box, detection_mode="Manual Full Box", green_threshold=0.35, green_dominance=0.08, cleanup=2, salient_sensitivity=0.5, jaw_margin_px=0):
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] != 3: raise ValueError("Activity Canvas expects IMAGE [B,H,W,3]")
    _, height, width, _ = image.shape
    device = image.device
    subjects = torch.zeros((1,height,width), device=device)
    activities = torch.zeros((1,height,width), device=device)
    candidates = torch.zeros((1,height,width), device=device)
    regions = []
    if detection_mode == "Auto Green Screen":
        m = _green_foreground(image, green_threshold, green_dominance, cleanup)
        if float(m.max()) <= 0: m = _salient_foreground(image, salient_sensitivity, cleanup); detected_by = "auto_salient"
        else: detected_by = "auto_green"
    elif detection_mode == "Auto Salient":
        m = _salient_foreground(image, salient_sensitivity, cleanup); detected_by = "auto_salient"
    else:
        m = None
    if m is not None and float(m.max()) > 0:
        region, box = _subject_region(m, height, width, subject_ratios[0][0], subject_ratios[0][1])
        if box is not None: subjects = torch.maximum(subjects, region); regions.append({"kind":"subject","index":0,"source":detected_by,"box":box,"ratio":[float(subject_ratios[0][0]),float(subject_ratios[0][1])]})
    for idx, mask in enumerate(subject_masks):
        if mask is None: continue
        mm = _normalize_mask(mask,height,width,device)
        if mm is None: continue
        region, box = _subject_region(mm,height,width,subject_ratios[idx][0],subject_ratios[idx][1])
        if box is not None: subjects = torch.maximum(subjects,region); regions.append({"kind":"subject","index":idx+1,"box":box,"ratio":[float(subject_ratios[idx][0]),float(subject_ratios[idx][1])]})
    for idx, mask in enumerate(activity_masks):
        if mask is None: continue
        mm = _normalize_mask(mask,height,width,device)
        if mm is not None: activities = torch.maximum(activities,_support_region(mm,height,width,activity_margin_px)); regions.append({"kind":"activity","index":idx+1,"margin_px":int(activity_margin_px)})
    candidate_used=False
    if use_candidate and candidate_mask is not None:
        mm=_normalize_mask(candidate_mask,height,width,device)
        if mm is not None:
            region,box=_subject_region(mm,height,width,candidate_ratio_x,candidate_ratio_y)
            if box is not None: candidates=torch.maximum(candidates,region); candidate_used=True; regions.append({"kind":"candidate","box":box,"ratio":[float(candidate_ratio_x),float(candidate_ratio_y)]})
    union=torch.maximum(subjects,torch.maximum(activities,candidates))
    use_manual = use_manual_bounds or detection_mode == "Manual Full Box"
    if use_manual:
        box=_align_box(round(width*manual_left),round(height*manual_top),round(width*manual_right),round(height*manual_bottom),width,height,stride); detected=True; manual=True
    elif float(union.max())>0:
        ys,xs=torch.where(union[0]>0.5); box=_align_box(int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1+max(0,int(jaw_margin_px)),width,height,stride); detected=True; manual=False
    else:
        x1,y1,x2,y2=fallback_box; box=_align_box(round(width*x1),round(height*y1),round(width*x2),round(height*y2),width,height,stride); union[:,box[1]:box[3],box[0]:box[2]]=1.0; detected=False; manual=False
    x1,y1,x2,y2=box
    if use_manual and float(union.max())<=0: union[:,y1:y2,x1:x2]=1.0
    roi_mask=union[:,y1:y2,x1:x2].contiguous()
    source=image[:,y1:y2,x1:x2,:].contiguous()
    model_w,model_h=_fit_size(x2-x1,y2-y1,target_mp,stride)
    local=_resize_image(source,model_w,model_h)
    local_mask=_resize_image(roi_mask[:,:,:,None],model_w,model_h)[:,:,:,0]
    protected=(1.0-union).clamp(0,1)
    canvas=image[:1].clone()*0.55
    tint_sub=torch.tensor([0.95,0.12,0.10],device=device,dtype=canvas.dtype)
    tint_act=torch.tensor([0.95,0.80,0.10],device=device,dtype=canvas.dtype)
    tint_cand=torch.tensor([0.15,0.45,1.00],device=device,dtype=canvas.dtype)
    overlay=subjects[:,:,:,None]*tint_sub+activities[:,:,:,None]*tint_act+candidates[:,:,:,None]*tint_cand
    weight=(subjects+activities+candidates).clamp(0,1)[:,:,:,None]*0.45
    canvas=canvas*(1.0-weight)+overlay*weight
    green=torch.tensor([0.10,0.95,0.35],device=device,dtype=canvas.dtype); edge=max(2,min(8,width//128))
    canvas[:,y1:y1+edge,x1:x2,:]=green; canvas[:,y2-edge:y2,x1:x2,:]=green; canvas[:,y1:y2,x1:x1+edge,:]=green; canvas[:,y1:y2,x2-edge:x2,:]=green
    context={"version":6,"mode":"latent_activity_canvas","detection_mode":detection_mode,"source_size":[int(width),int(height)],"box":[x1,y1,x2,y2],"crop_size":[x2-x1,y2-y1],"model_size":[model_w,model_h],"target_megapixels":float(target_mp),"stride":int(stride),"area_ratio":((x2-x1)*(y2-y1))/(width*height),"detected":detected,"manual_bounds":manual,"candidate_used":candidate_used,"subject_count":len([r for r in regions if r["kind"]=="subject"]),"activity_count":len([r for r in regions if r["kind"]=="activity"]),"regions":regions}
    text=json.dumps(context,ensure_ascii=False,separators=(",",":"))
    pipe={"version":6,"kind":"kinetic_pipe","detection_mode":detection_mode,"source_size":[int(width),int(height)],"crop_box":[x1,y1,x2,y2],"crop_size":[x2-x1,y2-y1],"model_size":[int(model_w),int(model_h)],"target_megapixels":float(target_mp),"stride":int(stride),"area_ratio":context["area_ratio"],"detected":detected,"manual_bounds":manual,"candidate_used":candidate_used,"subject_count":context["subject_count"],"activity_count":context["activity_count"],"regions":regions,"masks":{"subject":subjects,"activity":activities,"candidate":candidates,"protected_background":protected,"roi_mask":roi_mask,"local_mask":local_mask},"preview":canvas,"context":context,"context_json":text}
    return local, local[:1].clone(), local[-1:].clone(), pipe
