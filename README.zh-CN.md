# ComfyUI-KineticAnchor-H3

[English](README.md) | **简体中文**

面向 ComfyUI 中 MiniMax H3 视频生成的 ROI 画布缩减与回贴节点，以及用于遮罩重生成工作流的单节点 SAM3 主体/口型遮罩追踪。

## 概述

MiniMax H3 视频模型在矩形 latent 画布上运行。当主体仅占画面较小区域时，按完整画布生成会将算力消耗在静止背景上。KineticAnchor-H3 将画布缩减为主体周围 stride 对齐的区域，在缩减后的画布上生成视频，再通过色彩匹配与羽化融合将结果回贴到原始画面。

回贴链路由上下文驱动：节点间通过 JSON 上下文字符串传递裁剪几何信息，无需手动连线坐标。

另一组节点将 SAM3 检测 → 追踪 → 遮罩转换的链路压缩为单节点，输出遮罩遵循 H3 noise_mask 约定（`1` = 重生成，`0` = 保留原帧）。该约定由 ComfyUI 核心对 H3 原生实现（`MiniMaxH3.scale_latent_inpaint`）。

## 环境要求

- 较新版本的 ComfyUI，需支持 MiniMax H3。
- 两个遮罩追踪节点需要 SAM3 节点（`comfy_extras.nodes_sam3`）。SAM3 不可用时追踪节点不注册，其余四个节点正常加载。
- 除 ComfyUI 自身环境外无额外 Python 依赖。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-username>/ComfyUI-KineticAnchor-H3.git
```

重启 ComfyUI。或将仓库下载为 ZIP 压缩包，解压至 `ComfyUI/custom_nodes` 后重启 ComfyUI。

## 节点

全部节点注册在 `KineticAnchor` 菜单分类下。

| 类名 | 分类 | 功能 |
|---|---|---|
| `KineticAnchorGreenSubjectMask` | `KineticAnchor/Mask` | 全画布绿幕主体遮罩。 |
| `KineticAnchorROIPlanner` | `KineticAnchor/ROI` | ROI 规划，支持绿幕自动、外接遮罩与手动三种模式，输出按 stride 对齐。 |
| `KineticAnchorLatentActivityCanvas` | `KineticAnchor/Canvas` | 参数化活动画布。内置绿幕/显著性检测，至多四路主体遮罩（各自独立扩张比例）、支撑区域与候选遮罩、手动与兜底边界。提供 12 个分立输出，含局部条件帧与回贴上下文。 |
| `KineticAnchorTemporalPaste` | `KineticAnchor/Temporal` | 将生成的 ROI 视频缩放回源 ROI 尺寸并回贴至锚定帧，含色彩匹配、余弦窗羽化与运动门控融合。 |
| `KineticH3SubjectTrack` | `KineticAnchor-H3` | SAM3 全帧主体追踪，用于视频角色替换。输出追踪遮罩、黑底主体隔离视频（可作 `ref_video_1`）与遮罩预览视频。 |
| `KineticH3LipTrack` | `KineticAnchor-H3` | SAM3 口型区域追踪，用于重新对口型。输出口型区域遮罩与遮罩预览视频。 |

## 用法

### ROI 画布工作流（图片生成视频）

```text
LoadImage -> Latent Activity Canvas
  detection_mode = "Auto Green Screen"   （或 "Auto Salient"，或外接遮罩）
  local_first_frame / local_last_frame / model_width / model_height -> H3 FL2VA 条件
  H3 AV 解码 -> Temporal Paste.generated_roi
  LoadImage -> Temporal Paste.anchor_image
  Canvas.kinetic_context -> Temporal Paste.kinetic_context
  Temporal Paste -> SaveVideo
```

生成视频与源 ROI 尺寸不一致时自动缩放，局部遮罩随之缩放。`target_megapixels` 设为 `0` 时保持裁剪尺寸作为模型尺寸（match-source 语义）。

### 遮罩重生成（视频角色替换、重新对口型）

```text
VHS_LoadVideo -> KineticH3SubjectTrack（或 KineticH3LipTrack）
  sam3_model / clip <- CheckpointLoaderSimple（SAM3 权重）
  mask -> SetLatentNoiseMask 接至视频 latent 流
  isolated_video -> MiniMaxH3ReferenceToVideo.ref_video_1   （角色替换）
```

遮罩语义遵循 H3 noise mask 约定：`1` 标记重生成像素，`0` 标记保留像素。保留区域由 ComfyUI 核心在 H3 条件时间步注入；视频流与音频流各自携带独立的 denoise mask。

两个追踪节点共有的参数：

| 参数 | 说明 |
|---|---|
| `grow` | 遮罩外扩（正值）或内缩（负值），单位像素。 |
| `feather` | 边缘羽化宽度（像素），软化重生成区与保留区之间的过渡。 |
| `invert` | 反选遮罩，改为对追踪区域之外生成。 |
| `detect_interval` | 重新检测的帧间隔；值越小对快速运动的追踪越准确。 |
| `max_objects` | 最多追踪目标数；`0` 使用内部上限。 |
| `extra_mask` | 可选补充遮罩，与追踪结果取并集（仅主体追踪节点）。 |

## 示例

| 工作流 | 说明 |
|---|---|
| `examples/KineticAnchor_视频角色替换_遮罩引导简化版.json` | 使用 `KineticH3SubjectTrack` 的视频角色替换。 |
| `examples/KineticAnchor_重新对口型.json` | 使用 `KineticH3LipTrack` 与新音频的重新对口型。 |
| `examples/KineticAnchor_视频角色替换_SAM3版.json` | 由原始 SAM3 节点搭建的参考工作流，等价于 `KineticH3SubjectTrack` 的内部流程。 |

示例工作流另需 [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) 与 [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)。

## 测试

```bash
python test_nodes.py
```

离线回归测试，覆盖检测、ROI 对齐、上下文往返与回贴几何，使用合成输入，不加载模型。需在 ComfyUI 安装环境内运行，以保证 `comfy_extras` 可导入。

## 已知限制

- H3 使用矩形 latent 画布。分散的主体区域合并为其联合矩形；遮罩保护回贴区域，但不能消除矩形内部的注意力计算。实际提速幅度取决于 `model_width × model_height` 相对完整画布基线的下降程度。
- 离线测试不测量采样速度。评估实际提速应在相同 GPU、模型、seed、帧数与步数下对比完整画布与 ROI 画布的完整生成。

## Studio Pro

商业专业版（Studio Pro）另行发行，提供单总线 Pipeline 节点对、浏览器端活动区编辑器与场景预设。本仓库不包含任何商业版代码。

## 许可

MIT，详见 [LICENSE](LICENSE)。
