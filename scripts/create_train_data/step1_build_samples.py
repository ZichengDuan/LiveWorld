#!/usr/bin/env python3
"""
LiveWorld 数据处理主流程

整体流程:
1. 收集源视频 -> 随机采样clip (指定帧数/fps/分辨率)
2. Qwen VL: 识别视频中的前景物体 (如 person, car)
3. SAM3: 根据文字描述分割前景mask
4. Stream3R: 估计深度、相机位姿、内参
5. Sample Building: 构建训练样本 (点云投影、参考帧选择等)

配置在 configs/data_preparation.yaml 中修改
"""
from __future__ import annotations

import os
os.environ.setdefault("SAM3_TQDM_DISABLE", "1")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from omegaconf import OmegaConf
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG = OmegaConf.load(str(_PROJECT_ROOT / "configs" / "data_preparation.yaml"))

import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from tqdm import tqdm

import cv2
import numpy as np
import torch
import gc
import traceback
import datetime

for _logger_name in ["sam3", "sam3_video_predictor", "sam3.model_builder", "sam3.video_predictor"]:
    logging.getLogger(_logger_name).setLevel(logging.WARNING)
from scripts.create_train_data._sample_builder import save_video, save_training_sample, save_projection_rgb_video
from scripts.create_train_data._utils import get_rank_info, shard_items
from scripts.create_train_data._sample_builder import (
    build_training_sample,
    scale_intrinsics,
    extract_foreground,
    augment_foreground,
)
from scripts.create_train_data._estimators import Stream3REstimator
from scripts.create_train_data._sample_builder import build_scene_point_cloud
from scripts.create_train_data._projection import render_projection
from scripts.create_train_data._utils import VideoGeometry
from scripts.create_train_data._entity_detector import DEFAULT_PROMPT, Qwen3VLEntityExtractor
from scripts.create_train_data._sam3_segmenter import Sam3VideoSegmenter
from scripts.create_train_data._video_io import get_video_frame_count, get_video_fps, list_video_files, load_video_frames
from scripts.create_train_data._utils import get_sample_naming


# ============================================================================
# 常量
# ============================================================================

CLIP_FILENAME = "clip.mp4"


# ============================================================================
# 工具函数
# ============================================================================

def _cleanup_memory() -> None:
    """清理GPU和CPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ============================================================================
# Clip 提取函数 - 从原始视频中采样固定长度的片段
# ============================================================================

def _get_video_info(video_path: Path) -> Optional[Tuple[float, int, int, int]]:
    """获取视频基本信息: (fps, 帧数, 宽, 高)"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        return None
    return fps, frame_count, width, height


def _stable_rng(seed: int, video_path: Path) -> random.Random:
    """基于seed和视频路径生成确定性随机数生成器，确保同一视频每次采样结果一致"""
    key = f"{seed}:{video_path.as_posix()}".encode("utf-8")
    digest = hashlib.md5(key).digest()
    seed_int = int.from_bytes(digest[:8], "big")
    return random.Random(seed_int)


def _sample_frame_indices(
    frame_count: int,
    src_fps: float,
    target_fps: int,
    num_frames: int,
    rng: random.Random,
) -> Optional[list[int]]:
    """
    计算要提取的帧索引列表
    从源视频中按target_fps采样num_frames帧，随机选择起始时间点
    如果视频太短或fps不足则返回None
    """
    if src_fps + 1e-6 < target_fps:
        return None
    clip_seconds = num_frames / float(target_fps)
    duration = frame_count / float(src_fps)
    if duration + 1e-6 < clip_seconds:
        return None
    max_start = max(0.0, duration - clip_seconds)
    start_time = rng.uniform(0.0, max_start) if max_start > 0 else 0.0
    indices = []
    prev_idx = 0
    for i in range(num_frames):
        t = start_time + (i / float(target_fps))
        idx = int(round(t * src_fps))
        if idx < 0:
            idx = 0
        elif idx >= frame_count:
            idx = frame_count - 1
        if idx < prev_idx:
            idx = prev_idx
        indices.append(idx)
        prev_idx = idx
    return indices


def _resize_letterbox(frame: np.ndarray, target_wh: Tuple[int, int]) -> np.ndarray:
    """将帧resize到目标尺寸，保持宽高比，用黑边填充(letterbox)"""
    target_w, target_h = target_wh
    h, w = frame.shape[:2]
    if h == target_h and w == target_w:
        return frame
    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def extract_clip(
    video_path: Path,
    output_path: Path,
    num_frames: int,
    target_fps: int,
    target_wh: Tuple[int, int],
    seed: int,
    extract_fps: int | None = None,
) -> Tuple[bool, str]:
    """
    从源视频中提取clip: 随机选起点 -> 按extract_fps采样 -> letterbox resize -> 以target_fps保存mp4
    extract_fps 控制从源视频抽帧的间隔（影响运动速度），target_fps 控制输出视频帧率。
    """
    if extract_fps is None:
        extract_fps = target_fps

    info = _get_video_info(video_path)
    if info is None:
        return False, "invalid video metadata"
    src_fps, frame_count, _, _ = info

    # 检查视频是否满足要求（源视频fps必须 >= 抽帧fps）
    if src_fps + 1e-6 < extract_fps:
        return False, f"video fps too low (src_fps={src_fps:.1f} < extract={extract_fps})"

    clip_seconds = num_frames / float(extract_fps)
    duration = frame_count / float(src_fps)
    if duration + 1e-6 < clip_seconds:
        return False, f"video too short ({duration:.1f}s < {clip_seconds:.1f}s needed)"

    # 使用确定性随机数计算起始时间
    rng = _stable_rng(seed, video_path)
    max_start = max(0.0, duration - clip_seconds)
    start_time = rng.uniform(0.0, max_start) if max_start > 0 else 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 临时文件路径（FFmpeg 需要先写入临时文件）
    temp_path = output_path.with_suffix(".tmp.mp4")

    target_w, target_h = target_wh

    # 构建 FFmpeg 命令
    # -ss 在 -i 前面可以利用关键帧快速 seek
    # scale + pad 实现 letterbox 效果
    # -frames:v 精确指定输出帧数
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.6f}",
        "-i", str(video_path),
        "-vf", (
            f"fps={extract_fps},"
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black"
        ),
        "-frames:v", str(num_frames),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", str(target_fps),  # 输出帧率
        "-an",  # 无音频
        "-hide_banner",
        "-loglevel", "error",
        str(temp_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            if temp_path.exists():
                temp_path.unlink()
            return False, f"ffmpeg error: {result.stderr[:200]}"

        # 验证输出帧数
        out_info = _get_video_info(temp_path)
        if out_info is None:
            if temp_path.exists():
                temp_path.unlink()
            return False, "failed to read output video"

        _, out_frames, out_w, out_h = out_info

        # 精确匹配帧数
        if out_frames != num_frames:
            if temp_path.exists():
                temp_path.unlink()
            return False, f"frame count mismatch: got {out_frames}, expected {num_frames}"

        if out_w != target_w or out_h != target_h:
            if temp_path.exists():
                temp_path.unlink()
            return False, f"size mismatch: got {out_w}x{out_h}, expected {target_w}x{target_h}"

        # 重命名临时文件为最终文件
        temp_path.rename(output_path)
        return True, "ok"

    except subprocess.TimeoutExpired:
        if temp_path.exists():
            temp_path.unlink()
        return False, "ffmpeg timeout"
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        return False, f"ffmpeg exception: {str(e)[:100]}"


# ============================================================================
# 辅助函数
# ============================================================================

def _get_video_hw(video_path: str | Path) -> tuple[int, int] | None:
    """获取视频的高宽 (H, W)"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return (h, w) if h > 0 and w > 0 else None


def _load_cached_prompts(path: Path) -> list[str]:
    """从缓存的JSON文件中加载Qwen提取的prompts"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        prompts = payload.get("prompts") or payload.get("entities") or []
        return [str(p).strip() for p in prompts if str(p).strip()]
    except Exception:
        return []


def resize_masks(masks: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """将mask序列resize到目标尺寸"""
    H, W = target_hw
    resized = []
    for mask in masks:
        if mask.shape != (H, W):
            mask_u8 = (mask.astype(np.uint8) * 255)
            mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = mask_u8 > 0
        resized.append(mask)
    return np.stack(resized, axis=0)


def _compute_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Compute bbox (x0, y0, x1, y1) from a boolean mask."""
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return x0, y0, x1, y1


def _render_instance_on_canvas(
    frame: np.ndarray,
    mask: np.ndarray,
    canvas_hw: tuple[int, int],
) -> np.ndarray | None:
    """Crop instance by mask, scale to fit canvas, and center on black background."""
    bbox = _compute_mask_bbox(mask)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return None

    crop = frame[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    if crop_mask.size == 0:
        return None

    crop[~crop_mask] = 0

    canvas_h, canvas_w = canvas_hw
    box_h, box_w = crop.shape[:2]
    if box_h <= 0 or box_w <= 0:
        return None

    scale = min(canvas_w / float(box_w), canvas_h / float(box_h))
    new_w = max(1, int(round(box_w * scale)))
    new_h = max(1, int(round(box_h * scale)))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(
        crop_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
    ) > 0
    resized[~resized_mask] = 0

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=resized.dtype)
    x_off = (canvas_w - new_w) // 2
    y_off = (canvas_h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def check_dynamic_mask_quality(dynamic_masks: np.ndarray, threshold: float = 0.95) -> tuple[bool, float]:
    """检查mask质量: 如果某帧mask覆盖超过95%则认为无效(可能是全屏遮挡)"""
    if dynamic_masks is None or dynamic_masks.size == 0:
        return True, 0.0
    max_white = max(np.mean(m.astype(np.float32)) for m in dynamic_masks)
    return max_white < threshold, max_white


def is_mask_all_black(mask_path: Path) -> bool:
    """检查mask mp4文件是否全黑(没有任何分割结果)"""
    if not mask_path.exists():
        return False
    try:
        # 支持传入 .npy 或 .mp4 路径，统一转成 .mp4
        mp4_path = mask_path.with_suffix(".mp4") if mask_path.suffix == ".npy" else mask_path
        masks = load_mask_from_mp4(mp4_path)
        if masks is None or masks.size == 0:
            return True
        return not masks.any()
    except Exception:
        return False


def save_per_category_masks(
    folder_path: Path,
    category_masks: dict[str, np.ndarray],
    fps: float,
) -> None:
    """保存每个类别的mask为独立的mp4文件（不再保存npy）"""

    for prompt, masks in category_masks.items():
        safe_name = prompt.replace(" ", "_").replace("/", "_")
        mask_mp4 = folder_path / f"mask_{safe_name}.mp4"

        masks_rgb = np.repeat(masks.astype(np.uint8)[..., None] * 255, 3, axis=-1)
        save_video(masks_rgb, mask_mp4, fps=fps)


def load_mask_from_mp4(mask_mp4: Path) -> np.ndarray | None:
    """从mp4文件加载mask序列，返回 (T, H, W) 的 bool 数组"""
    if not mask_mp4.exists():
        return None
    try:
        cap = cv2.VideoCapture(str(mask_mp4))
        if not cap.isOpened():
            return None
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # mask mp4 是灰度的（或RGB全相同），取第一个通道 > 127 作为 bool
            if frame.ndim == 3:
                frame = frame[..., 0]
            frames.append(frame > 127)
        cap.release()
        if not frames:
            return None
        return np.stack(frames, axis=0)
    except Exception:
        return None


def load_and_merge_category_masks(folder_path: Path, prompts: list[str], exclude_prompts: list[str] = None) -> np.ndarray | None:
    """
    加载并合并多个类别的mask（从mp4文件读取）。

    Args:
        folder_path: 数据文件夹路径
        prompts: 要加载的类别列表
        exclude_prompts: 要排除的类别（如 sky）

    Returns:
        合并后的mask或None
    """
    exclude_prompts = exclude_prompts or []
    merged = None

    for prompt in prompts:
        if prompt.lower() in [p.lower() for p in exclude_prompts]:
            continue
        safe_name = prompt.replace(" ", "_").replace("/", "_")
        mask_mp4 = folder_path / f"mask_{safe_name}.mp4"
        masks = load_mask_from_mp4(mask_mp4)
        if masks is None or masks.size == 0:
            continue
        if merged is None:
            merged = masks.astype(bool)
        else:
            merged = merged | masks.astype(bool)

    return merged


def _has_foreground_mask_files(folder_path: Path) -> bool:
    """Check if any non-sky per-category mask mp4 exists (used to infer fg availability)."""
    for mask_path in folder_path.glob("mask_*.mp4"):
        if mask_path.name == "mask_sky.mp4":
            continue
        return True
    return False


def _normalize_output_size(
    output_size: object,
    geometry: VideoGeometry,
    cfg,
) -> tuple[int, int]:
    """Normalize output size (H, W) with safe fallbacks."""
    if isinstance(output_size, (list, tuple)) and len(output_size) == 2:
        try:
            out_h, out_w = int(output_size[0]), int(output_size[1])
            if out_h > 0 and out_w > 0:
                return out_h, out_w
        except Exception:
            pass
    if geometry.original_size is not None:
        return int(geometry.original_size[0]), int(geometry.original_size[1])
    if cfg is not None:
        return int(cfg.clip_target_height), int(cfg.clip_target_width)
    H, W = geometry.frames.shape[1:3]
    return H, W


def _get_projection_rgb_slice(channels: list[str]) -> Optional[slice]:
    """Find RGB slice in projection channel layout."""
    cursor = 0
    rgb_slice = None
    for name in channels:
        if name == "rgb":
            rgb_slice = slice(cursor, cursor + 3)
            cursor += 3
        else:
            cursor += 1
    return rgb_slice


def _resolve_projection_aug(sample_meta: dict, sample_config) -> Optional[dict]:
    """Resolve projection augmentation params for rerendering."""
    def _defaults() -> dict:
        return {
            "apply_prob": getattr(sample_config, "projection_aug_apply_prob", 1.0),
            "pixel_drop_min": getattr(sample_config, "projection_aug_pixel_drop_min", 0.0),
            "pixel_drop_max": getattr(sample_config, "projection_aug_pixel_drop_max", 0.0),
            "block_drop_min": getattr(sample_config, "projection_aug_block_drop_min", 0.0),
            "block_drop_max": getattr(sample_config, "projection_aug_block_drop_max", 0.0),
            "block_size_min": getattr(sample_config, "projection_aug_block_size_min", 16),
            "block_size_max": getattr(sample_config, "projection_aug_block_size_max", 64),
            "erode_iters_min": getattr(sample_config, "projection_aug_erode_iters_min", 0),
            "erode_iters_max": getattr(sample_config, "projection_aug_erode_iters_max", 0),
            "blur_sigma_min": getattr(sample_config, "projection_aug_blur_sigma_min", 0.0),
            "blur_sigma_max": getattr(sample_config, "projection_aug_blur_sigma_max", 0.0),
        }

    if "projection_aug" in sample_meta:
        meta = sample_meta.get("projection_aug")
        if not isinstance(meta, dict):
            return None
        params = _defaults()
        for key in params:
            if key in meta:
                params[key] = meta[key]
        params["seed"] = meta.get("seed")
        return params

    if not getattr(sample_config, "projection_aug_enable", False):
        return None
    params = _defaults()
    params["seed"] = None
    return params


def _rerender_projection_videos(
    folder_path: Path,
    sample_meta: dict,
    geometry: VideoGeometry,
    frames: np.ndarray,
    dynamic_masks_proc: Optional[np.ndarray],
    foreground_masks_proc: Optional[np.ndarray],
    cfg,
    sample_config,
    fps: float,
    prefix: str,
) -> str:
    output_size = _normalize_output_size(sample_meta.get("output_size"), geometry, cfg)
    proj_h, proj_w = output_size
    H, W = geometry.frames.shape[1:3]
    if H <= 0 or W <= 0:
        return "invalid_geometry_size"
    scale_x = proj_w / W
    scale_y = proj_h / H

    geometry.depths



    scene_idx = sample_meta.get("scene_idx")
    if scene_idx is None:
        return "missing_scene_idx"
    scene_idx = int(scene_idx)
    if scene_idx < 0 or scene_idx >= geometry.depths.shape[0]:
        return "scene_idx_oob"

    projection_channels = list(sample_meta.get("projection_channels") or cfg.projection_channels)
    rgb_slice = _get_projection_rgb_slice(projection_channels)
    if rgb_slice is None:
        return "no_rgb_channel"

    naming_style = sample_meta.get("naming") or cfg.naming_style
    names = get_sample_naming(naming_style)

    scene_depth = geometry.depths[scene_idx]
    scene_K = geometry.intrinsics[scene_idx]
    scene_rgb = geometry.frames[scene_idx]
    scene_valid_mask = None if geometry.masks is None else geometry.masks[scene_idx]
    scene_dynamic_mask = None if dynamic_masks_proc is None else dynamic_masks_proc[scene_idx]

    if sample_config.upsample_geometry_to_output and output_size is not None:
        scene_depth = cv2.resize(scene_depth, (proj_w, proj_h), interpolation=cv2.INTER_NEAREST)
        scene_K = scale_intrinsics(scene_K, scale_x, scale_y)
        if frames is not None:
            scene_rgb = np.asarray(frames)[scene_idx]
        if scene_rgb.shape[:2] != (proj_h, proj_w):
            scene_rgb = cv2.resize(scene_rgb, (proj_w, proj_h), interpolation=cv2.INTER_LINEAR)
        if scene_valid_mask is not None and scene_valid_mask.shape != (proj_h, proj_w):
            scene_valid_mask = cv2.resize(
                scene_valid_mask.astype(np.uint8),
                (proj_w, proj_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if scene_dynamic_mask is not None and scene_dynamic_mask.shape != (proj_h, proj_w):
            scene_dynamic_mask = cv2.resize(
                scene_dynamic_mask.astype(np.uint8),
                (proj_w, proj_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

    scene_xyz, scene_rgb_pts = build_scene_point_cloud(
        depth=scene_depth,
        K=scene_K,
        c2w=geometry.poses_c2w[scene_idx],
        rgb=scene_rgb,
        valid_mask=scene_valid_mask,
        dynamic_mask=scene_dynamic_mask,
        voxel_size=sample_config.scene_voxel_size,
    )

    if scene_xyz.size == 0:
        return "empty_scene"
    if "rgb" in projection_channels and scene_rgb_pts is None:
        return "missing_scene_rgb"

    # Get indices (support both old P_idx and new P9_idx/P1_idx)
    preceding_idx_9 = sample_meta.get("P9_idx", sample_meta.get("P_idx", []))
    preceding_idx_1 = sample_meta.get("P1_idx", [])
    target_idx = sample_meta.get("T_idx", [])
    if not preceding_idx_9 or not target_idx:
        return "missing_indices"

    # P9 scene projection (9帧)
    # NOTE: Hole filling is disabled to preserve geometry.
    preceding_proj_9 = []
    for idx in preceding_idx_9:
        idx = int(idx)
        if idx < 0 or idx >= geometry.intrinsics.shape[0]:
            return "P9_idx_oob"
        K_scaled = scale_intrinsics(geometry.intrinsics[idx], scale_x, scale_y)
        proj = render_projection(
            scene_xyz,
            K_scaled,
            geometry.poses_c2w[idx],
            (proj_h, proj_w),
            projection_channels,
            colors=scene_rgb_pts,
            fill_holes_kernel=0,
        )
        preceding_proj_9.append(proj)
    preceding_proj_9 = np.stack(preceding_proj_9, axis=0).astype(np.float32)

    # P1 scene projection (1帧)
    if preceding_idx_1:
        preceding_proj_1 = []
        for idx in preceding_idx_1:
            idx = int(idx)
            if idx < 0 or idx >= geometry.intrinsics.shape[0]:
                return "P1_idx_oob"
            K_scaled = scale_intrinsics(geometry.intrinsics[idx], scale_x, scale_y)
            proj = render_projection(
                scene_xyz,
                K_scaled,
                geometry.poses_c2w[idx],
                (proj_h, proj_w),
                projection_channels,
                colors=scene_rgb_pts,
                fill_holes_kernel=0,
            )
            preceding_proj_1.append(proj)
        preceding_proj_1 = np.stack(preceding_proj_1, axis=0).astype(np.float32)
    else:
        # Fallback: use last frame of P9
        preceding_proj_1 = preceding_proj_9[-1:, ...]

    # Target scene projection
    target_proj = []
    for idx in target_idx:
        idx = int(idx)
        if idx < 0 or idx >= geometry.intrinsics.shape[0]:
            return "T_idx_oob"
        K_scaled = scale_intrinsics(geometry.intrinsics[idx], scale_x, scale_y)
        proj = render_projection(
            scene_xyz,
            K_scaled,
            geometry.poses_c2w[idx],
            (proj_h, proj_w),
            projection_channels,
            colors=scene_rgb_pts,
            fill_holes_kernel=0,
        )
        target_proj.append(proj)
    target_proj = np.stack(target_proj, axis=0).astype(np.float32)

    save_projection_rgb_video(
        preceding_proj_9[..., rgb_slice],
        folder_path / f"{prefix}{names.preceding_scene_proj_rgb_9}.mp4",
        fps=fps,
    )
    save_projection_rgb_video(
        preceding_proj_1[..., rgb_slice],
        folder_path / f"{prefix}{names.preceding_scene_proj_rgb_1}.mp4",
        fps=fps,
    )
    save_projection_rgb_video(
        target_proj[..., rgb_slice],
        folder_path / f"{prefix}{names.target_scene_proj_rgb}.mp4",
        fps=fps,
    )

    # Foreground projections (augmented foreground on black)
    preceding_idx_9_int = [int(i) for i in preceding_idx_9]
    target_idx_int = [int(i) for i in target_idx]
    if preceding_idx_1:
        preceding_idx_1_int = [int(i) for i in preceding_idx_1]
    else:
        preceding_idx_1_int = [preceding_idx_9_int[-1]]

    proj_P9_fg = np.zeros((len(preceding_idx_9_int), proj_h, proj_w, 3), dtype=np.uint8)
    proj_P1_fg = np.zeros((len(preceding_idx_1_int), proj_h, proj_w, 3), dtype=np.uint8)
    proj_T_fg = np.zeros((len(target_idx_int), proj_h, proj_w, 3), dtype=np.uint8)

    if frames is not None and foreground_masks_proc is not None and foreground_masks_proc.size > 0:
        frames_np = np.asarray(frames)

        def _resize_rgb_stack(stack: np.ndarray) -> np.ndarray:
            if stack.shape[1:3] == (proj_h, proj_w):
                return stack
            resized = [
                cv2.resize(frame, (proj_w, proj_h), interpolation=cv2.INTER_LINEAR)
                for frame in stack
            ]
            return np.stack(resized, axis=0)

        fg_preceding_9 = foreground_masks_proc[preceding_idx_9_int]
        fg_preceding_1 = foreground_masks_proc[preceding_idx_1_int]
        fg_target = foreground_masks_proc[target_idx_int]
        fg_preceding_9 = resize_masks(fg_preceding_9, (proj_h, proj_w))
        fg_preceding_1 = resize_masks(fg_preceding_1, (proj_h, proj_w))
        fg_target = resize_masks(fg_target, (proj_h, proj_w))

        preceding_rgb_9 = _resize_rgb_stack(frames_np[preceding_idx_9_int])
        preceding_rgb_1 = _resize_rgb_stack(frames_np[preceding_idx_1_int])
        target_rgb = _resize_rgb_stack(frames_np[target_idx_int])

        preceding_fg_9 = extract_foreground(preceding_rgb_9, fg_preceding_9)
        preceding_fg_1 = extract_foreground(preceding_rgb_1, fg_preceding_1)
        target_fg = extract_foreground(target_rgb, fg_target)

        depth_preceding_9 = geometry.depths[preceding_idx_9_int]
        depth_preceding_1 = geometry.depths[preceding_idx_1_int]
        depth_target = geometry.depths[target_idx_int]
        depth_preceding_9 = np.stack(
            [cv2.resize(d, (proj_w, proj_h), interpolation=cv2.INTER_NEAREST) for d in depth_preceding_9]
        )
        depth_preceding_1 = np.stack(
            [cv2.resize(d, (proj_w, proj_h), interpolation=cv2.INTER_NEAREST) for d in depth_preceding_1]
        )
        depth_target = np.stack(
            [cv2.resize(d, (proj_w, proj_h), interpolation=cv2.INTER_NEAREST) for d in depth_target]
        )

        rng = np.random.default_rng(sample_config.random_seed)
        proj_P9_fg = augment_foreground(
            preceding_fg_9,
            fg_preceding_9,
            depth_preceding_9,
            dot_ratio_min=sample_config.fg_dot_ratio_min,
            dot_ratio_max=sample_config.fg_dot_ratio_max,
            rng=rng,
        ).astype(np.uint8)
        proj_P1_fg = augment_foreground(
            preceding_fg_1,
            fg_preceding_1,
            depth_preceding_1,
            dot_ratio_min=sample_config.fg_dot_ratio_min,
            dot_ratio_max=sample_config.fg_dot_ratio_max,
            rng=rng,
        ).astype(np.uint8)
        proj_T_fg = augment_foreground(
            target_fg,
            fg_target,
            depth_target,
            dot_ratio_min=sample_config.fg_dot_ratio_min,
            dot_ratio_max=sample_config.fg_dot_ratio_max,
            rng=rng,
        ).astype(np.uint8)

    save_projection_rgb_video(
        proj_P9_fg,
        folder_path / f"{prefix}{names.preceding_proj_fg_rgb_9}.mp4",
        fps=fps,
    )
    save_projection_rgb_video(
        proj_P1_fg,
        folder_path / f"{prefix}{names.preceding_proj_fg_rgb_1}.mp4",
        fps=fps,
    )
    save_projection_rgb_video(
        proj_T_fg,
        folder_path / f"{prefix}{names.target_proj_fg_rgb}.mp4",
        fps=fps,
    )
    return "ok"


def log_failed_sample(log_path: Path, video_id: str, reason: str, rank: int) -> None:
    """记录处理失败的样本"""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}][R{rank}] {video_id}: {reason}\n")


def cleanup_failed_output(output_dir: Path) -> None:
    """清理失败的输出目录"""
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)


def mark_as_failed(output_dir: Path, reason: str) -> None:
    """创建.skip标记文件，后续运行时跳过此文件夹"""
    output_dir.mkdir(parents=True, exist_ok=True)
    skip_file = output_dir / ".skip"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    skip_file.write_text(f"[{ts}] {reason}\n", encoding="utf-8")


def is_marked_failed(output_dir: Path) -> tuple[bool, str]:
    """检查文件夹是否已被标记为失败"""
    skip_file = output_dir / ".skip"
    if skip_file.exists():
        try:
            reason = skip_file.read_text(encoding="utf-8").strip()
            return True, reason
        except Exception:
            return True, "unknown"
    return False, ""


# ============================================================================
# 背景光流运动检测 (GPU)
# ============================================================================

def _resize_short_side(img: np.ndarray, short_side: int) -> np.ndarray:
    """Resize image so its short side equals short_side."""
    h, w = img.shape[:2]
    if min(h, w) == short_side:
        return img
    scale = short_side / float(min(h, w))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def check_bg_motion(
    clip_path: Path,
    masks: np.ndarray | None,
    cfg,
    device: str,
) -> tuple[bool, str]:
    """
    用 GPU 光流检测背景是否有足够的 camera motion。

    流程:
    1. 均匀采样 N 对相邻帧
    2. 灰度化 + resize 到 short_side
    3. 前景 mask erode 后扩展为 "不可用区域"
    4. 用 cv2.calcOpticalFlowFarneback 计算光流 (CPU, 但在 resize 后的小图上很快)
    5. 只在可见背景像素上统计光流幅度中位数
    6. 跳过背景面积太小的帧对
    7. 如果足够多帧对的背景运动都 < threshold → 判定静止

    Returns: (is_ok, reason_str)
        is_ok=True: 背景有运动，通过
        is_ok=False: 背景静止，应该拒绝
    """
    num_samples = cfg.bg_motion_num_samples
    short_side = cfg.bg_motion_short_side
    erode_px = cfg.bg_motion_mask_erode
    min_bg_ratio = cfg.bg_motion_min_bg_ratio
    threshold = cfg.bg_motion_threshold

    # 读取视频帧
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return True, "cannot_open"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 2:
        cap.release()
        return True, "too_few_frames"

    # 均匀采样帧索引对
    indices = np.linspace(0, frame_count - 1, num=min(num_samples + 1, frame_count), dtype=int)
    pairs = [(indices[i], indices[i + 1]) for i in range(len(indices) - 1) if indices[i] != indices[i + 1]]

    # 腐蚀 kernel
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)) if erode_px > 0 else None

    flow_magnitudes: list[float] = []
    skipped = 0

    for idx_a, idx_b in pairs:
        # 读取两帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx_a))
        ok_a, frame_a = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx_b))
        ok_b, frame_b = cap.read()
        if not ok_a or not ok_b or frame_a is None or frame_b is None:
            skipped += 1
            continue

        # 灰度 + resize
        gray_a = _resize_short_side(cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY), short_side)
        gray_b = _resize_short_side(cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY), short_side)
        h, w = gray_a.shape[:2]

        # 构建背景 mask (True = 可用背景)
        if masks is not None and len(masks) > 0:
            # 取两帧的前景 mask 的并集，然后 erode 扩展不可用区域
            mask_a = masks[min(idx_a, len(masks) - 1)]
            mask_b = masks[min(idx_b, len(masks) - 1)]
            fg_union = (mask_a | mask_b).astype(np.uint8) * 255
            fg_resized = cv2.resize(fg_union, (w, h), interpolation=cv2.INTER_NEAREST)
            if erode_kernel is not None:
                # dilate foreground = erode background
                fg_resized = cv2.dilate(fg_resized, erode_kernel)
            bg_mask = fg_resized < 128  # True = background
        else:
            bg_mask = np.ones((h, w), dtype=bool)

        bg_ratio = bg_mask.sum() / float(h * w)
        if bg_ratio < min_bg_ratio:
            skipped += 1
            continue

        # 计算光流 (Farneback, CPU, 小图很快)
        flow = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        # flow shape: (h, w, 2)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        # 只统计背景像素的光流
        bg_mag = mag[bg_mask]
        if len(bg_mag) == 0:
            skipped += 1
            continue

        median_mag = float(np.median(bg_mag))
        flow_magnitudes.append(median_mag)

    cap.release()

    if not flow_magnitudes:
        return True, "no_valid_pairs"

    overall_median = float(np.median(flow_magnitudes))

    # 静止检测
    if overall_median < threshold:
        return False, f"static_bg(flow={overall_median:.2f}<{threshold}, pairs={len(flow_magnitudes)}, skip={skipped})"

    # 转场检测: 背景光流突变 (max/median 比值过大说明某帧对发生了场景切换)
    scene_cut_ratio = cfg.bg_motion_scene_cut_ratio
    if scene_cut_ratio > 0 and len(flow_magnitudes) >= 3:
        max_flow = max(flow_magnitudes)
        if overall_median > 1e-6 and max_flow / overall_median > scene_cut_ratio:
            return False, f"scene_cut(max/med={max_flow/overall_median:.1f}>{scene_cut_ratio}, max={max_flow:.2f}, med={overall_median:.2f})"

    return True, f"ok(flow={overall_median:.2f}, pairs={len(flow_magnitudes)}, skip={skipped})"


# ============================================================================
# 日志函数
# ============================================================================

def log_stage_header(stage_name: str, rank: int) -> None:
    """打印阶段开始的header"""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n{'=' * 60}", flush=True)
    print(f"[{ts}][R{rank}] STAGE: {stage_name}", flush=True)
    print(f"{'=' * 60}", flush=True)


def log_stage_progress(stage: str, idx: int, total: int, video_id: str, rank: int, status: str = "", skipped: int = 0) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    skip_info = f" (skipped={skipped})" if skipped > 0 else ""
    status_str = f" [{status}]" if status else ""
    print(f"[{ts}][R{rank}][{stage}] {idx}/{total}{skip_info} {video_id}{status_str}", flush=True)


def log_stage_summary(stage: str, rank: int, processed: int, skipped: int, failed: int = 0) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    parts = [f"processed={processed}", f"skipped={skipped}"]
    if failed > 0:
        parts.append(f"failed={failed}")
    print(f"[{ts}][R{rank}][{stage}] DONE: {', '.join(parts)}", flush=True)


def log_message(msg: str, rank: int, stage: str = None, only_rank0: bool = False) -> None:
    if only_rank0 and rank != 0:
        return
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    tag = f"[{ts}][R{rank}]"
    if stage:
        tag += f"[{stage}]"
    print(f"{tag} {msg}", flush=True)


# ============================================================================
# 视频收集
# ============================================================================

def collect_all_source_videos(video_dirs: list[str], rank: int,
                              video_dirs_ratio: list[int] | None = None) -> list[Path]:
    """从配置的视频目录中收集所有源视频路径，按 ratio 比例采样。

    ratio 控制每个目录在最终池中的相对份额。例如 ratio=[1,2] 表示
    第二个目录的视频列表会被重复2次放入池中。下游 shuffle 和消费逻辑不变。
    """
    per_dir_videos: list[list[Path]] = []
    for vdir in video_dirs:
        vdir_path = Path(vdir)
        if not vdir_path.exists():
            log_message(f"Warning: video_dir not found: {vdir}", rank, "Init")
            per_dir_videos.append([])
            continue
        videos = sorted(list_video_files(vdir))
        log_message(f"Found {len(videos)} videos in {vdir}", rank, "Init", only_rank0=True)
        per_dir_videos.append(videos)

    # 没有 ratio 或长度不匹配时退回等权
    if video_dirs_ratio is None or len(video_dirs_ratio) != len(video_dirs):
        all_videos = []
        for videos in per_dir_videos:
            all_videos.extend(videos)
        return sorted(all_videos)

    # 按比例重复每个目录的视频列表
    all_videos = []
    for videos, r in zip(per_dir_videos, video_dirs_ratio):
        for _ in range(r):
            all_videos.extend(videos)
    log_message(
        f"Video pool after ratio weighting: {len(all_videos)} "
        f"(ratio={video_dirs_ratio})", rank, "Init", only_rank0=True,
    )
    return sorted(all_videos)


def prepare_folder_ids(max_videos: int) -> list[str]:
    """生成输出文件夹ID列表 (00000000, 00000001, ...)"""
    return [f"{idx:08d}" for idx in range(max_videos)]


def shuffle_videos(all_videos: list[Path], shuffle_seed: int | None) -> list[Path]:
    """使用固定seed打乱视频顺序"""
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        all_videos = list(all_videos)
        rng.shuffle(all_videos)
    return all_videos


# ============================================================================
# 完成状态检查 - 用于增量处理和断点续传
# ============================================================================

def _check_clip_complete(folder_path: Path, cfg) -> bool:
    """检查clip.mp4是否存在且符合配置要求"""
    clip_path = folder_path / CLIP_FILENAME
    if not clip_path.exists():
        return False
    try:
        info = _get_video_info(clip_path)
        if info is None:
            return False
        _, frame_count, w, h = info
        return (frame_count == cfg.clip_num_frames and
                w == cfg.clip_target_width and
                h == cfg.clip_target_height)
    except Exception:
        return False


def _check_stage3_geometry_complete(video_out: Path) -> bool:
    """检查geometry.npz是否已生成"""
    return (video_out / "geometry.npz").exists()


def _check_folder_complete(folder_path: Path, cfg) -> tuple[bool, str]:
    """检查文件夹是否完整 (clip + geometry + samples都存在)"""
    # Check if marked as failed - this is an error, needs re-sample
    is_failed, fail_reason = is_marked_failed(folder_path)
    if is_failed:
        return False, f"error:{fail_reason}"

    # Check clip
    if not _check_clip_complete(folder_path, cfg):
        return False, "interrupted:clip_incomplete"

    # Check geometry
    if not _check_stage3_geometry_complete(folder_path):
        return False, "interrupted:geometry_incomplete"

    # Check samples
    if not _check_stage3_samples_complete(folder_path, cfg):
        return False, "interrupted:samples_incomplete"

    return True, "ok"


def cleanup_error_folder(folder_path: Path, rank: int) -> None:
    """清理错误文件夹，下次迭代时会重新采样新视频"""
    if folder_path.exists():
        shutil.rmtree(folder_path, ignore_errors=True)
        log_message(f"Cleaned up error folder (will re-sample): {folder_path.name}", rank, "Cleanup")


def _check_stage3_samples_complete(video_out: Path, cfg, projection_only: bool = False) -> bool:
    """检查训练样本是否完整生成"""
    names = get_sample_naming(cfg.naming_style)
    prefixes = ["train_"] if cfg.num_samples == 1 else [f"train_sample{i:03d}_" for i in range(cfg.num_samples)]
    has_rgb_proj = "rgb" in cfg.projection_channels

    for prefix in prefixes:
        if projection_only:
            # projection_only 模式只检查 scene projection 和 fg projection 文件
            if has_rgb_proj:
                required = [
                    video_out / f"{prefix}{names.preceding_scene_proj_rgb_9}.mp4",
                    video_out / f"{prefix}{names.preceding_scene_proj_rgb_1}.mp4",
                    video_out / f"{prefix}{names.target_scene_proj_rgb}.mp4",
                    video_out / f"{prefix}{names.preceding_proj_fg_rgb_9}.mp4",
                    video_out / f"{prefix}{names.preceding_proj_fg_rgb_1}.mp4",
                    video_out / f"{prefix}{names.target_proj_fg_rgb}.mp4",
                ]
            else:
                required = []
        else:
            # 完整模式检查所有文件
            required = [
                video_out / f"{prefix}sample.json",
                # 两种 preceding 模式
                video_out / f"{prefix}{names.preceding_rgb_9}.mp4",
                video_out / f"{prefix}{names.preceding_rgb_1}.mp4",
                video_out / f"{prefix}{names.target_rgb}.mp4",
            ]
            if has_rgb_proj:
                required.extend([
                    video_out / f"{prefix}{names.preceding_scene_proj_rgb_9}.mp4",
                    video_out / f"{prefix}{names.preceding_scene_proj_rgb_1}.mp4",
                    video_out / f"{prefix}{names.target_scene_proj_rgb}.mp4",
                ])
            # Reference scene output is only expected when reference exists AND fg masks were produced
            reference_path = video_out / f"{prefix}{names.reference_rgb}.mp4"
            if reference_path.exists() and _has_foreground_mask_files(video_out):
                required.extend([
                    video_out / f"{prefix}{names.reference_scene_rgb}.mp4",
                ])
        if not all(p.exists() for p in required):
            return False
    return True


def _save_reference_scene(
    ref_frames: np.ndarray,
    fg_masks: np.ndarray,
    out_scene: Path,
    fps: float,
) -> None:
    reference_scene = ref_frames.copy()
    reference_scene[fg_masks] = 0
    save_video(reference_scene, out_scene, fps=fps)


def _collect_instance_only_items(
    output_root: Path,
    folder_ids: list[str],
    cfg,
) -> list[tuple[Path, str, Path]]:
    """Collect existing folders with target videos for instance-only runs."""
    names = get_sample_naming(cfg.naming_style)
    items = []
    for folder_id in folder_ids:
        folder_path = output_root / folder_id
        if not folder_path.is_dir():
            continue
        if is_marked_failed(folder_path)[0]:
            continue
        sample_json_paths = sorted(folder_path.glob("train*sample.json"))
        if not sample_json_paths:
            continue
        target_found = False
        target_path = None
        for sample_json in sample_json_paths:
            prefix = sample_json.name[: -len("sample.json")]
            candidate = folder_path / f"{prefix}{names.target_rgb}.mp4"
            if candidate.exists():
                target_found = True
                target_path = candidate
                break
        if not target_found:
            continue
        clip_path = folder_path / CLIP_FILENAME
        if not clip_path.exists() and target_path is not None:
            clip_path = target_path
        items.append((clip_path, folder_id, folder_path))
    return items


def _run_instance_reference_stage(
    cfg,
    rank: int,
    local_rank: int,
    video_items: list[tuple[Path, str, Path]],
    output_root: Path,
) -> None:
    log_stage_header("Instance Reference", rank)

    instance_segmenter = Sam3VideoSegmenter(
        gpus_to_use=[local_rank],
        propagation_direction=cfg.sam3_propagation_direction,
        score_threshold_detection=cfg.sam3_score_threshold,
        new_det_thresh=cfg.sam3_new_det_thresh,
        checkpoint_path=cfg.sam3_checkpoint_path,
        multi_prompt=cfg.sam3_multi_prompt,
        resize_input_to=cfg.sam3_resize_input_to,
    )

    inst_processed, inst_skipped, inst_failed = 0, 0, 0
    inst_failed_log = output_root / "instance_ref_failed_samples.txt"
    names = get_sample_naming(cfg.naming_style)

    for idx, (_, folder_id, folder_path) in enumerate(video_items, start=1):
        is_failed, _ = is_marked_failed(folder_path)
        if is_failed:
            inst_skipped += 1
            continue

        sample_json_paths = sorted(folder_path.glob("train*sample.json"))
        if not sample_json_paths:
            inst_skipped += 1
            log_stage_progress("InstRef", idx, len(video_items), folder_id, rank, status="skip(no-sample)", skipped=inst_skipped)
            continue

        prompts_path = folder_path / "dynamic_prompts.json"
        prompts = Sam3VideoSegmenter.normalize_prompts(
            _load_cached_prompts(prompts_path) if prompts_path.exists() else []
        )
        prompts = [p for p in prompts if p.strip().lower() != "sky"]
        if not prompts:
            inst_skipped += 1
            log_stage_progress("InstRef", idx, len(video_items), folder_id, rank, status="skip(no-prompts)", skipped=inst_skipped)
            continue

        sample_total = len(sample_json_paths)
        sample_done = 0
        sample_skipped = 0

        for sample_json in sample_json_paths:
            prefix = sample_json.name[: -len("sample.json")]
            target_path = folder_path / f"{prefix}{names.target_rgb}.mp4"
            if not target_path.exists():
                sample_skipped += 1
                continue

            meta_out = folder_path / f"{prefix}reference_instances.json"
            if cfg.skip_existing and meta_out.exists():
                sample_skipped += 1
                continue

            try:
                frames = load_video_frames(target_path)
            except Exception as exc:
                log_failed_sample(inst_failed_log, folder_id, f"{target_path.name}: {exc}", rank)
                inst_failed += 1
                continue

            fps = cfg.fps_override or get_video_fps(target_path)
            frame_count = len(frames)
            if frame_count == 0:
                sample_skipped += 1
                continue

            instance_masks = instance_segmenter.segment_instances(
                video_path=str(target_path),
                prompts=prompts,
                frame_index=cfg.sam3_frame_index,
                start_frame_index=cfg.sam3_start_frame_index,
                max_frame_num_to_track=cfg.sam3_max_frame_num_to_track,
                expected_frames=frame_count,
                frames=frames,
            )

            H, W = frames[0].shape[:2]
            total_area = float(H * W)

            ranked = []
            for obj_id, masks in instance_masks.items():
                if masks.ndim != 3 or masks.shape[0] == 0:
                    continue
                areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
                max_area = float(areas.max())
                if max_area <= 0:
                    continue
                ranked.append((obj_id, max_area / total_area, areas, masks))

            ranked.sort(key=lambda x: x[1], reverse=True)
            selected = ranked[: cfg.instance_ref_max_instances]

            # Remove old instance videos for this prefix if reprocessing
            for old in folder_path.glob(f"{prefix}reference_instance_*.mp4"):
                old.unlink(missing_ok=True)

            instances_meta = []
            for rank_idx, (obj_id, max_ratio, areas, masks) in enumerate(selected):
                valid_indices = np.where(areas > 0)[0].tolist()
                if not valid_indices:
                    continue

                base_seed = cfg.instance_ref_random_seed
                if base_seed is None:
                    base_seed = cfg.shuffle_seed or 0
                rng = _stable_rng(int(base_seed), Path(f"{target_path.as_posix()}#{obj_id}"))
                k = min(cfg.instance_ref_max_frames, len(valid_indices))
                chosen = rng.sample(valid_indices, k)
                chosen = sorted(chosen)

                instance_frames = []
                for frame_idx in chosen:
                    frame = frames[frame_idx]
                    mask = masks[frame_idx]
                    canvas = _render_instance_on_canvas(frame, mask, (H, W))
                    if canvas is not None:
                        instance_frames.append(canvas)

                if not instance_frames:
                    continue

                out_path = folder_path / f"{prefix}reference_instance_{rank_idx:02d}.mp4"
                save_video(np.stack(instance_frames, axis=0), out_path, fps=fps)

                instances_meta.append(
                    {
                        "rank": rank_idx,
                        "obj_id": int(obj_id),
                        "max_area_ratio": float(max_ratio),
                        "frame_indices": chosen,
                    }
                )

            meta_payload = {
                "target_video": target_path.name,
                "prompts": prompts,
                "instances": instances_meta,
            }
            meta_out.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
            sample_done += 1

            if cfg.cleanup_interval and idx % cfg.cleanup_interval == 0:
                _cleanup_memory()

        if sample_done > 0:
            inst_processed += 1
            log_stage_progress(
                "InstRef",
                idx,
                len(video_items),
                folder_id,
                rank,
                status=f"instances({sample_done}/{sample_total})",
                skipped=inst_skipped,
            )
        else:
            inst_skipped += 1
            status = "skip(inst-exist)" if sample_skipped == sample_total else "skip(inst-empty)"
            log_stage_progress(
                "InstRef",
                idx,
                len(video_items),
                folder_id,
                rank,
                status=status,
                skipped=inst_skipped,
            )

    log_stage_summary("InstRef", rank, inst_processed, inst_skipped, inst_failed)
    del instance_segmenter
    _cleanup_memory()

# ============================================================================
# 主处理流程
# ============================================================================

def run_pipeline_iteration(
    cfg,
    rank: int,
    world_size: int,
    local_rank: int,
    device: str,
    output_root: Path,
    all_videos: list[Path],
    my_folder_ids: list[str],
    iteration: int,
) -> int:
    """
    执行一次完整的处理迭代，返回未完成的文件夹数量

    处理流程:
    1. Clip提取: 从源视频池中为每个folder_id提取clip.mp4
    2. Qwen阶段: 识别视频中的前景物体，输出dynamic_prompts.json
    3. SAM3阶段: 根据prompts分割前景mask，输出dynamic_masks.npy
    4. MapAnything+Sample阶段: 估计几何信息，构建训练样本
    """
    if getattr(cfg, "instance_only", False):
        log_stage_header(f"Instance Only Setup (iter={iteration})", rank)
        video_items = _collect_instance_only_items(output_root, my_folder_ids, cfg)
        if not video_items:
            log_message("No eligible folders for instance-only stage.", rank, "InstRef")
            return 0
        _run_instance_reference_stage(
            cfg=cfg,
            rank=rank,
            local_rank=local_rank,
            video_items=video_items,
            output_root=output_root,
        )
        return 0

    # 每个rank使用不同的随机种子打乱视频顺序
    rank_seed = (cfg.shuffle_seed or 42) + rank + (iteration * 1000)
    shuffled_videos = shuffle_videos(all_videos, rank_seed)

    # ============ 第1步: Clip提取 ============
    # 从源视频池中提取固定长度的clip，每个folder_id对应一个clip
    log_stage_header(f"Clip Extraction (iter={iteration})", rank)

    clip_target_wh = (cfg.clip_target_width, cfg.clip_target_height)
    clip_seed = (cfg.shuffle_seed or 42) + iteration

    # 构建待处理列表: (video_path, folder_id, folder_path)
    video_items = []
    setup_extracted, setup_cached, setup_skipped = 0, 0, 0
    video_pool_idx = 0  # 视频池索引

    # 遍历分配给当前rank的所有folder_id
    for folder_idx, folder_id in enumerate(my_folder_ids, start=1):
        folder_path = output_root / folder_id

        # 跳过已完成的文件夹
        is_complete, _ = _check_folder_complete(folder_path, cfg)
        if is_complete:
            clip_path = folder_path / CLIP_FILENAME
            video_items.append((clip_path, folder_id, folder_path))
            setup_cached += 1
            continue

        # 跳过之前已标记失败的文件夹
        is_failed, fail_reason = is_marked_failed(folder_path)
        if is_failed:
            setup_skipped += 1
            continue

        # 检查clip是否已存在且有效
        clip_path = folder_path / CLIP_FILENAME
        if clip_path.exists():
            try:
                info = _get_video_info(clip_path)
                if info is not None:
                    _, frame_count, w, h = info
                    if frame_count == cfg.clip_num_frames and w == clip_target_wh[0] and h == clip_target_wh[1]:
                        video_items.append((clip_path, folder_id, folder_path))
                        setup_cached += 1
                        continue
            except Exception:
                pass
            clip_path.unlink(missing_ok=True)

        # 从视频池中尝试提取clip，失败则尝试下一个视频
        extraction_success = False
        attempts = 0
        max_attempts = len(shuffled_videos)

        while not extraction_success and attempts < max_attempts:
            if video_pool_idx >= len(shuffled_videos):
                break

            source_video = shuffled_videos[video_pool_idx]
            video_pool_idx += 1
            attempts += 1

            chosen_extract_fps = random.choice(cfg.clip_extract_fps)
            success, message = extract_clip(
                video_path=source_video,
                output_path=clip_path,
                num_frames=cfg.clip_num_frames,
                target_fps=cfg.clip_target_fps,
                target_wh=clip_target_wh,
                seed=clip_seed,
                extract_fps=chosen_extract_fps,
            )

            if success:
                video_items.append((clip_path, folder_id, folder_path))
                setup_extracted += 1
                extraction_success = True
                log_stage_progress(
                    "Extract", folder_idx, len(my_folder_ids), folder_id, rank,
                    status=f"ok (attempts={attempts})" if attempts > 1 else "ok",
                    skipped=setup_cached + setup_skipped
                )

        if not extraction_success:
            log_message(f"WARN: {folder_id} - exhausted video pool after {attempts} attempts", rank, "Extract")
            setup_skipped += 1

    log_message(
        f"Clip extraction: extracted={setup_extracted}, cached={setup_cached}, skipped={setup_skipped}, "
        f"videos_used={video_pool_idx}/{len(shuffled_videos)}",
        rank, "Extract"
    )

    if not video_items:
        log_message("No video items to process!", rank, "Init")
        return 0

    # ============ 第2步: Qwen VL 前景物体识别 ============
    # 输入: clip.mp4
    # 输出: dynamic_prompts.json (包含识别出的前景物体描述，如 "person", "car")
    if not cfg.disable_qwen:
        log_stage_header("Qwen (entity extraction)", rank)
        qwen = Qwen3VLEntityExtractor(model_path=cfg.qwen_model_path, device=device)
        qwen_processed, qwen_skipped = 0, 0

        for idx, (video_path, folder_id, folder_path) in enumerate(video_items, start=1):
            is_failed, fail_reason = is_marked_failed(folder_path)
            if is_failed:
                qwen_skipped += 1
                continue

            prompts_path = folder_path / "dynamic_prompts.json"
            if cfg.skip_existing and prompts_path.exists():
                try:
                    json.loads(prompts_path.read_text(encoding="utf-8"))
                    qwen_skipped += 1
                    continue
                except Exception:
                    pass

            try:
                # 使用Qwen VL模型识别视频中的前景物体
                qwen_prompt = cfg.qwen_prompt or DEFAULT_PROMPT
                raw_prompts, raw_text = qwen.extract(str(video_path), prompt=qwen_prompt)
                raw_prompts = raw_prompts[:cfg.max_prompts]
                prompts = Sam3VideoSegmenter.normalize_prompts(raw_prompts)

                payload = {"raw": raw_text, "entities": raw_prompts, "prompts": prompts, "status": "nothing" if not prompts else "ok"}
                prompts_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                qwen_processed += 1

                log_stage_progress("Qwen", idx, len(video_items), folder_id, rank, status=", ".join(prompts) or "none", skipped=qwen_skipped)
            except Exception as e:
                log_stage_progress("Qwen", idx, len(video_items), folder_id, rank, status=f"ERROR: {str(e)[:50]}", skipped=qwen_skipped)
                mark_as_failed(folder_path, f"Qwen error: {e}")
                qwen_skipped += 1

            if cfg.cleanup_interval and idx % cfg.cleanup_interval == 0:
                _cleanup_memory()

        log_stage_summary("Qwen", rank, qwen_processed, qwen_skipped)
        del qwen
        _cleanup_memory()

    # ============ 第3步: SAM3 前景分割 ============
    # 输入: clip.mp4 + dynamic_prompts.json
    # 输出:
    #   - mask_{category}.npy / mask_{category}.mp4: 每个类别的独立mask
    #   - dynamic_masks.npy / dynamic_masks.mp4: 合并后的前景mask (兼容旧流程)
    if not cfg.disable_sam3:
        log_stage_header("SAM3 (segmentation)", rank)
        sam3 = Sam3VideoSegmenter(
            gpus_to_use=[local_rank],
            propagation_direction=cfg.sam3_propagation_direction,
            score_threshold_detection=cfg.sam3_score_threshold,
            new_det_thresh=cfg.sam3_new_det_thresh,
            checkpoint_path=cfg.sam3_checkpoint_path,
            multi_prompt=cfg.sam3_multi_prompt,
            resize_input_to=cfg.sam3_resize_input_to,
        )

        sam3_failed_log = output_root / "sam3_failed_samples.txt"
        sam3_processed, sam3_skipped, sam3_failed = 0, 0, 0

        for idx, (video_path, folder_id, folder_path) in enumerate(video_items, start=1):
            is_failed, fail_reason = is_marked_failed(folder_path)
            if is_failed:
                sam3_skipped += 1
                continue

            # 加载Qwen识别的prompts
            prompts_path = folder_path / "dynamic_prompts.json"
            prompts = Sam3VideoSegmenter.normalize_prompts(_load_cached_prompts(prompts_path) if prompts_path.exists() else [])

            masks_npy = folder_path / "dynamic_masks.npy"
            masks_mp4 = folder_path / "dynamic_masks.mp4"

            # 跳过逻辑:
            # 1. 如果dynamic_masks.mp4存在且全黑 -> 跳过 (没有分割到任何物体)
            # 2. 如果per-category masks mp4已经存在 -> 跳过 (已分割过)
            if cfg.skip_existing:
                # 检查是否有per-category masks (mp4)
                has_category_masks = False
                if prompts:
                    for prompt in prompts:
                        safe_name = prompt.replace(" ", "_").replace("/", "_")
                        if (folder_path / f"mask_{safe_name}.mp4").exists():
                            has_category_masks = True
                            break

                if has_category_masks or masks_mp4.exists():
                    sam3_skipped += 1
                    continue

            try:
                num_frames = get_video_frame_count(video_path)
                video_fps = get_video_fps(video_path)

                # 如果没有识别到前景物体，生成全零mask
                if not prompts:
                    hw = _get_video_hw(video_path)
                    if hw is None:
                        sam3_failed += 1
                        continue
                    dynamic_masks = np.zeros((num_frames, hw[0], hw[1]), dtype=bool)
                    status = "empty"
                else:
                    # 使用SAM3分割每个类别的前景物体
                    category_masks = sam3.segment_per_category(
                        str(video_path), prompts=prompts,
                        frame_index=cfg.sam3_frame_index,
                        start_frame_index=cfg.sam3_start_frame_index,
                        max_frame_num_to_track=cfg.sam3_max_frame_num_to_track,
                        expected_frames=num_frames,
                    )

                    # 保存每个类别的独立mask
                    save_per_category_masks(folder_path, category_masks, video_fps)

                    # 合并所有类别的mask (用于后续流程)
                    if category_masks:
                        dynamic_masks = None
                        for masks in category_masks.values():
                            if dynamic_masks is None:
                                dynamic_masks = masks.astype(bool)
                            else:
                                dynamic_masks = dynamic_masks | masks.astype(bool)
                        status = ", ".join(category_masks.keys())
                    else:
                        hw = _get_video_hw(video_path)
                        if hw is None:
                            sam3_failed += 1
                            continue
                        dynamic_masks = np.zeros((num_frames, hw[0], hw[1]), dtype=bool)
                        status = "no-mask"

                # 保存合并后的mask为mp4（不再保存npy）
                masks_rgb = np.repeat(dynamic_masks.astype(np.uint8)[..., None] * 255, 3, axis=-1)
                save_video(masks_rgb, masks_mp4, fps=video_fps)
                sam3_processed += 1
                log_stage_progress("SAM3", idx, len(video_items), folder_id, rank, status=status, skipped=sam3_skipped)

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                    log_stage_progress("SAM3", idx, len(video_items), folder_id, rank, status="OOM", skipped=sam3_skipped)
                    log_failed_sample(sam3_failed_log, folder_id, f"OOM: {e}", rank)
                    mark_as_failed(folder_path, f"SAM3 OOM: {e}")
                    sam3_failed += 1
                    _cleanup_memory()
                else:
                    log_stage_progress("SAM3", idx, len(video_items), folder_id, rank, status=f"ERROR: {str(e)[:50]}", skipped=sam3_skipped)
                    log_failed_sample(sam3_failed_log, folder_id, f"RuntimeError: {e}", rank)
                    mark_as_failed(folder_path, f"SAM3 error: {e}")
                    sam3_failed += 1

            if cfg.cleanup_interval and idx % cfg.cleanup_interval == 0:
                _cleanup_memory()

        log_stage_summary("SAM3", rank, sam3_processed, sam3_skipped, sam3_failed)
        del sam3
        _cleanup_memory()

    # ============ 第3.5步: 背景运动过滤 (光流 + mask) ============
    if cfg.bg_motion_filter:
        log_stage_header("Background Motion Filter", rank)
        bg_ok, bg_reject, bg_skip = 0, 0, 0
        filtered_items = []
        for idx, (video_path, folder_id, folder_path) in enumerate(video_items, start=1):
            is_failed, _ = is_marked_failed(folder_path)
            if is_failed:
                bg_skip += 1
                continue

            # 天空面积检查
            sky_mask_mp4 = folder_path / "mask_sky.mp4"
            sky_masks = load_mask_from_mp4(sky_mask_mp4)
            if sky_masks is not None and len(sky_masks) > 0:
                sky_ratio = float(sky_masks.mean())
                if sky_ratio > cfg.bg_motion_max_sky_ratio:
                    mark_as_failed(folder_path, f"sky_too_large: {sky_ratio:.1%} > {cfg.bg_motion_max_sky_ratio:.0%}")
                    bg_reject += 1
                    log_stage_progress("BgMotion", idx, len(video_items), folder_id, rank,
                                       status=f"sky={sky_ratio:.1%}, reject")
                    continue

            # 加载前景 mask
            masks_mp4 = folder_path / "dynamic_masks.mp4"
            masks = load_mask_from_mp4(masks_mp4)

            passed, reason = check_bg_motion(video_path, masks, cfg, device)
            if passed:
                filtered_items.append((video_path, folder_id, folder_path))
                bg_ok += 1
                log_stage_progress("BgMotion", idx, len(video_items), folder_id, rank, status=reason)
            else:
                mark_as_failed(folder_path, f"bg_motion_static: {reason}")
                bg_reject += 1
                log_stage_progress("BgMotion", idx, len(video_items), folder_id, rank, status=reason)

        log_message(f"Background motion filter: passed={bg_ok}, rejected={bg_reject}, skipped={bg_skip}", rank, "BgMotion")
        video_items = filtered_items

    # ============ 第4步: Stream3R几何估计 + 训练样本构建 ============
    # 输入: clip.mp4 + dynamic_masks.npy
    # 输出:
    #   - geometry.npz: 深度图、相机位姿(c2w)、相机内参
    #   - train_*.mp4: 训练样本视频 (preceding/target/reference RGB, scene投影等)
    #   - train_sample.json: 样本元信息
    log_stage_header("Geometry (Stream3R) + Sample Building", rank)

    stream3r_config = OmegaConf.create({
        "model_path": cfg.get("stream3r_model_path", "ckpts/yslan--STream3R"),
        "device": device,
        "preprocess_mode": cfg.get("stream3r_preprocess_mode", "crop"),
        "target_size": cfg.get("stream3r_target_size", 518),
        "window_size": cfg.get("stream3r_window_size", 32),
        "conf_threshold": cfg.get("stream3r_conf_threshold", 0.0),
    })
    estimator = Stream3REstimator(stream3r_config)

    sample_config = CONFIG  # Use the global config directly

    failed_log = output_root / "failed_samples.txt"
    s3_processed, s3_skipped, s3_failed = 0, 0, 0

    _show_pbar = (local_rank == 0)
    pbar = tqdm(total=len(video_items), desc=f"[R{rank}] Stage3", disable=not _show_pbar,
                dynamic_ncols=True, leave=True)

    for idx, (video_path, folder_id, folder_path) in enumerate(video_items, start=1):
        is_failed, fail_reason = is_marked_failed(folder_path)
        if is_failed:
            s3_skipped += 1
            pbar.update(1)
            pbar.set_postfix_str(f"ok={s3_processed} skip={s3_skipped} fail={s3_failed}")
            continue

        geo_complete = _check_stage3_geometry_complete(folder_path) if cfg.skip_existing else False
        sample_complete = _check_stage3_samples_complete(folder_path, cfg, projection_only=cfg.projection_only) if cfg.skip_existing else False

        if getattr(cfg, "reference_only", False):
            names = get_sample_naming(cfg.naming_style)
            sample_json_paths = sorted(folder_path.glob("train*sample.json"))
            if not sample_json_paths:
                s3_skipped += 1
                pbar.update(1); pbar.set_postfix_str(f"ok={s3_processed} skip={s3_skipped} fail={s3_failed}")
                continue

            # Load foreground masks (exclude sky). If missing, nothing to do.
            prompts_path = folder_path / "dynamic_prompts.json"
            prompts = Sam3VideoSegmenter.normalize_prompts(_load_cached_prompts(prompts_path) if prompts_path.exists() else [])
            fg_masks_all = load_and_merge_category_masks(folder_path, prompts, exclude_prompts=["sky"])
            if fg_masks_all is None or fg_masks_all.size == 0:
                s3_skipped += 1
                pbar.update(1); pbar.set_postfix_str(f"ok={s3_processed} skip={s3_skipped} fail={s3_failed}")
                continue

            fps = cfg.fps_override or get_video_fps(video_path)
            updated = 0
            for sample_json in sample_json_paths:
                try:
                    meta = json.loads(sample_json.read_text(encoding="utf-8"))
                except Exception:
                    continue
                prefix = sample_json.name[: -len("sample.json")]

                ref_path = folder_path / f"{prefix}{names.reference_rgb}.mp4"
                out_scene = folder_path / f"{prefix}{names.reference_scene_rgb}.mp4"

                if not ref_path.exists():
                    continue
                if out_scene.exists():
                    continue

                ref_indices = meta.get("R_idx", [])
                if not ref_indices:
                    continue
                if max(ref_indices) >= len(fg_masks_all):
                    continue

                ref_frames = np.asarray(load_video_frames(ref_path))
                if ref_frames.size == 0:
                    continue

                fg_masks = fg_masks_all[ref_indices]
                if fg_masks.shape[1:3] != ref_frames.shape[1:3]:
                    fg_masks = resize_masks(fg_masks, ref_frames.shape[1:3])

                _save_reference_scene(ref_frames, fg_masks, out_scene, fps)
                updated += 1

            if updated > 0:
                s3_processed += 1
            else:
                s3_skipped += 1
            pbar.update(1); pbar.set_postfix_str(f"ok={s3_processed} skip={s3_skipped} fail={s3_failed}")
            continue

        if cfg.projection_only:
            # projection_only 模式: 只要 projection 文件完整就跳过
            if sample_complete:
                s3_skipped += 1
                pbar.update(1); pbar.set_postfix_str(f"ok={s3_processed} skip={s3_skipped} fail={s3_failed}")
                continue
        elif geo_complete and sample_complete:
            s3_skipped += 1
            pbar.update(1); pbar.set_postfix_str(f"ok={s3_processed} skip={s3_skipped} fail={s3_failed}")
            continue

        try:
            folder_path.mkdir(parents=True, exist_ok=True)

            # 加载前景mask
            prompts_path = folder_path / "dynamic_prompts.json"
            prompts = Sam3VideoSegmenter.normalize_prompts(_load_cached_prompts(prompts_path) if prompts_path.exists() else [])

            dynamic_masks = None
            foreground_masks = None  # 前景mask (排除sky)
            if not cfg.disable_sam3 and prompts:
                # 从mp4加载dynamic_masks
                masks_mp4 = folder_path / "dynamic_masks.mp4"
                dynamic_masks = load_mask_from_mp4(masks_mp4)
                # 加载前景mask: 合并除sky以外的所有类别
                foreground_masks = load_and_merge_category_masks(
                    folder_path, prompts, exclude_prompts=["sky"]
                )

            # 检查mask质量
            if dynamic_masks is not None:
                valid, max_white = check_dynamic_mask_quality(dynamic_masks, 0.95)
                if not valid:
                    pbar.update(1); pbar.set_postfix_str(f"ok={s3_processed} skip={s3_skipped} fail={s3_failed}")
                    log_failed_sample(failed_log, folder_id, f"mask white ratio {max_white:.0%}", rank)
                    mark_as_failed(folder_path, f"mask white ratio {max_white:.0%}")
                    s3_failed += 1
                    continue

            fps = cfg.fps_override or get_video_fps(video_path)
            frames = load_video_frames(video_path)

            # 几何估计: 使用Stream3R估计深度和相机位姿
            if not geo_complete:
                geometry = estimator.estimate_video_geometry(frames, frame_indices=list(range(len(frames))))
                np.savez_compressed(
                    folder_path / "geometry.npz",
                    depths=geometry.depths.astype(np.float32),
                    poses_c2w=geometry.poses_c2w.astype(np.float64),
                    intrinsics=geometry.intrinsics.astype(np.float64),
                    fps=np.array([fps]),
                    original_size=np.array(geometry.original_size),
                    processed_size=np.array(geometry.processed_size),
                )
                H, W = geometry.frames.shape[1:3]
                dynamic_masks_proc = resize_masks(dynamic_masks, (H, W)) if dynamic_masks is not None and dynamic_masks.size > 0 else None
                foreground_masks_proc = resize_masks(foreground_masks, (H, W)) if foreground_masks is not None and foreground_masks.size > 0 else None
                geo_status = "geo"
            else:
                # 加载缓存的几何信息
                geo_data = dict(np.load(folder_path / "geometry.npz"))
                proc_h, proc_w = int(geo_data["processed_size"][0]), int(geo_data["processed_size"][1])
                orig_h, orig_w = int(geo_data["original_size"][0]), int(geo_data["original_size"][1])

                # Online upgrade: if depths are at old proc resolution, upsample to original
                if proc_h != orig_h or proc_w != orig_w:
                    depths = geo_data["depths"]
                    if depths.shape[1] != orig_h or depths.shape[2] != orig_w:
                        scale_x, scale_y = orig_w / proc_w, orig_h / proc_h
                        depths_up = np.stack([
                            cv2.resize(d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                            for d in depths
                        ])
                        intrinsics_up = geo_data["intrinsics"].copy()
                        intrinsics_up[:, 0, 0] *= scale_x
                        intrinsics_up[:, 1, 1] *= scale_y
                        intrinsics_up[:, 0, 2] *= scale_x
                        intrinsics_up[:, 1, 2] *= scale_y
                        geo_data["depths"] = depths_up.astype(np.float32)
                        geo_data["intrinsics"] = intrinsics_up
                    geo_data["processed_size"] = np.array([orig_h, orig_w], dtype=np.int64)
                    np.savez_compressed(folder_path / "geometry.npz", **geo_data)

                H, W = int(geo_data["processed_size"][0]), int(geo_data["processed_size"][1])
                frames_resized = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) for f in frames], axis=0)
                geometry = VideoGeometry(
                    frames=frames_resized, depths=geo_data["depths"], intrinsics=geo_data["intrinsics"],
                    poses_c2w=geo_data["poses_c2w"], masks=None, frame_indices=np.arange(len(frames), dtype=np.int32),
                    original_size=tuple(geo_data["original_size"]), processed_size=(H, W),
                )
                dynamic_masks_proc = resize_masks(dynamic_masks, (H, W)) if dynamic_masks is not None and dynamic_masks.size > 0 else None
                foreground_masks_proc = resize_masks(foreground_masks, (H, W)) if foreground_masks is not None and foreground_masks.size > 0 else None
                geo_status = "geo-cached"

            # 构建训练样本
            ref_status = ""
            if cfg.projection_only:
                sample_json_paths = sorted(folder_path.glob("train*sample.json"))
                if not sample_json_paths:
                    sample_status = "projection-only(no-sample)"
                else:
                    total = len(sample_json_paths)
                    ok = 0
                    errors: list[str] = []
                    for sample_json in sample_json_paths:
                        try:
                            meta = json.loads(sample_json.read_text(encoding="utf-8"))
                        except Exception:
                            errors.append(f"{sample_json.name}:json")
                            continue
                        prefix = sample_json.name[: -len("sample.json")]
                        status = _rerender_projection_videos(
                            folder_path=folder_path,
                            sample_meta=meta,
                            geometry=geometry,
                            frames=frames,
                            dynamic_masks_proc=dynamic_masks_proc,
                            foreground_masks_proc=foreground_masks_proc,
                            cfg=cfg,
                            sample_config=sample_config,
                            fps=fps,
                            prefix=prefix,
                        )
                        if status == "ok":
                            ok += 1
                        else:
                            errors.append(f"{prefix}{status}")
                    if errors:
                        pbar.write(f"[R{rank}] proj-only warnings {folder_id}: {', '.join(errors[:3])}")
                    sample_status = f"projection-only({ok}/{total})"
            elif not sample_complete:
                original_frames = frames
                output_size = (cfg.clip_target_height, cfg.clip_target_width)

                rng = np.random.default_rng(cfg.random_seed)
                for sample_idx in range(cfg.num_samples):
                    # 调用build_training_sample构建完整训练样本
                    sample = build_training_sample(
                        geometry=geometry, config=sample_config,
                        dynamic_masks=dynamic_masks_proc,
                        foreground_masks=foreground_masks_proc,
                        rng=rng,
                        projection_fill_kernel=0,
                        original_frames=original_frames,
                        output_size=output_size,
                    )

                    meta = sample.get("meta", {})
                    r_stats = meta.get("R_stats", {})
                    ref_count = len(meta.get("R_idx", []))
                    best_iou = r_stats.get("best_iou", 0)
                    threshold = r_stats.get("threshold", 0)

                    if ref_count > 0:
                        ref_status = f"ref={ref_count}, iou={best_iou:.3f}"
                    else:
                        reason = r_stats.get("no_ref_reason", "unknown")
                        if reason == "iou_below_threshold":
                            ref_status = f"NO_REF(iou={best_iou:.3f}<{threshold})"
                        elif reason == "no_candidates":
                            ref_status = "NO_REF(no_candidates)"
                        else:
                            ref_status = f"NO_REF({reason})"

                    prefix = "train_" if cfg.num_samples == 1 else f"train_sample{sample_idx:03d}_"
                    _fg_overlay_aug = None
                    if getattr(cfg, "fg_overlay_aug_enable", False):
                        _fg_overlay_aug = {
                            k.replace("fg_overlay_aug_", ""): getattr(cfg, k)
                            for k in dir(cfg)
                            if k.startswith("fg_overlay_aug_") and k != "fg_overlay_aug_enable"
                        }
                    save_training_sample(sample, folder_path, projection_channels=cfg.projection_channels,
                                         fps=fps, naming=cfg.naming_style, name_prefix=prefix,
                                         fg_overlay_aug=_fg_overlay_aug)
                    # Save augmented reference frame for visual inspection
                    if "scene_rgb_augmented" in sample:
                        cv2.imwrite(str(folder_path / f"{prefix}scene_rgb_augmented.png"),
                                    cv2.cvtColor(sample["scene_rgb_augmented"], cv2.COLOR_RGB2BGR))
                        cv2.imwrite(str(folder_path / f"{prefix}scene_rgb_original.png"),
                                    cv2.cvtColor(sample["scene_rgb_original"], cv2.COLOR_RGB2BGR))
                sample_status = f"sample({ref_status})"
            else:
                sample_json = folder_path / "train_sample.json"
                if sample_json.exists():
                    try:
                        sample_meta = json.loads(sample_json.read_text(encoding="utf-8"))
                        ref_count = len(sample_meta.get("R_idx", []))
                        r_stats = sample_meta.get("R_stats", {})
                        if ref_count > 0:
                            ref_status = f"ref={ref_count}, iou={r_stats.get('best_iou', 0):.3f}"
                        else:
                            reason = r_stats.get("no_ref_reason", "unknown")
                            best_iou = r_stats.get("best_iou", 0)
                            threshold = r_stats.get("threshold", 0)
                            if reason == "iou_below_threshold":
                                ref_status = f"NO_REF(iou={best_iou:.3f}<{threshold})"
                            else:
                                ref_status = f"NO_REF({reason})"
                    except Exception:
                        ref_status = "ref=?"
                sample_status = f"sample-cached({ref_status})"

            s3_processed += 1
            pbar.update(1); pbar.set_postfix_str(f"ok={s3_processed} skip={s3_skipped} fail={s3_failed}")

        except Exception as e:
            tb = traceback.format_exc()
            pbar.write(f"[R{rank}] Stage3 ERROR {folder_id}: {str(e)[:80]}")
            log_failed_sample(failed_log, folder_id, f"{e}\n{tb}", rank)
            mark_as_failed(folder_path, f"Stage3 error: {e}")
            if not geo_complete:
                skip_file = folder_path / ".skip"
                skip_content = skip_file.read_text(encoding="utf-8") if skip_file.exists() else None
                cleanup_failed_output(folder_path)
                if skip_content:
                    folder_path.mkdir(parents=True, exist_ok=True)
                    skip_file.write_text(skip_content, encoding="utf-8")
            s3_failed += 1
            pbar.update(1); pbar.set_postfix_str(f"ok={s3_processed} skip={s3_skipped} fail={s3_failed}")
            continue

        if cfg.cleanup_interval and idx % cfg.cleanup_interval == 0:
            _cleanup_memory()

    pbar.close()
    log_stage_summary("Stage3", rank, s3_processed, s3_skipped, s3_failed)
    del estimator
    _cleanup_memory()

    # ============ 第5步: Instance Reference (target instances) ============
    if getattr(cfg, "instance_ref_enable", False):
        _run_instance_reference_stage(
            cfg=cfg,
            rank=rank,
            local_rank=local_rank,
            video_items=video_items,
            output_root=output_root,
        )

    # ============ 完成状态检查 ============
    # 检查哪些文件夹完成、哪些失败需要重新采样
    log_stage_header(f"Completeness Check (iter={iteration})", rank)

    error_folders = []      # 有.skip标记，需要删除并重新采样
    interrupted_folders = []  # 无.skip但不完整，可以增量处理
    complete_count = 0

    for folder_id in my_folder_ids:
        folder_path = output_root / folder_id
        is_complete, reason = _check_folder_complete(folder_path, cfg)
        if is_complete:
            complete_count += 1
        elif reason.startswith("error:"):
            error_folders.append((folder_id, reason))
        else:
            interrupted_folders.append((folder_id, reason))

    log_message(
        f"Completeness: complete={complete_count}, errors={len(error_folders)}, interrupted={len(interrupted_folders)}",
        rank, "Check"
    )

    # Clean up error folders (will re-sample new videos next iteration)
    if error_folders:
        log_message(f"Cleaning up {len(error_folders)} error folders for re-sampling:", rank, "Cleanup")
        for folder_id, reason in error_folders:
            folder_path = output_root / folder_id
            cleanup_error_folder(folder_path, rank)
            log_message(f"  {folder_id}: {reason}", rank, "Cleanup")

    # Interrupted folders are kept as-is, will continue incrementally
    if interrupted_folders:
        log_message(f"Keeping {len(interrupted_folders)} interrupted folders for incremental processing:", rank, "Check")
        for folder_id, reason in interrupted_folders:
            log_message(f"  {folder_id}: {reason}", rank, "Check")

    # Return total incomplete count (both error and interrupted need more work)
    return len(error_folders) + len(interrupted_folders)


def main():
    cfg = CONFIG  # Use global config directly

    rank, world_size, local_rank = get_rank_info()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Print config on rank 0
    if rank == 0:
        from omegaconf import OmegaConf as _OC
        print("=" * 60)
        print("LiveWorld Data Preparation Config")
        print("=" * 60)
        print(_OC.to_yaml(cfg))
        print("=" * 60)

    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # ============ Collect source videos ============
    log_stage_header("Video Collection", rank)

    # Collect all source videos
    all_videos = collect_all_source_videos(cfg.video_dirs, rank, cfg.video_dirs_ratio)
    log_message(f"Total source videos: {len(all_videos)}", rank, "Init", only_rank0=True)

    if not all_videos:
        log_message("No source videos found!", rank, "Init")
        return

    # Prepare folder IDs and shard across ranks
    max_videos = cfg.max_videos or len(all_videos)
    folder_ids = prepare_folder_ids(max_videos)
    my_folder_ids = shard_items(folder_ids, rank, world_size)
    log_message(f"This rank processing {len(my_folder_ids)} folders", rank, "Init")

    # ============ Run pipeline with retry loop ============
    max_iterations = 10  # Safety limit
    iteration = 0

    while iteration < max_iterations:
        log_stage_header(f"PIPELINE ITERATION {iteration}", rank)

        incomplete_count = run_pipeline_iteration(
            cfg=cfg,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            output_root=output_root,
            all_videos=all_videos,
            my_folder_ids=my_folder_ids,
            iteration=iteration,
        )

        if incomplete_count == 0:
            log_message(f"All {len(my_folder_ids)} folders complete!", rank, "Done")
            break

        log_message(
            f"Iteration {iteration} done. {incomplete_count} folders incomplete, retrying...",
            rank, "Retry"
        )
        iteration += 1

    if iteration >= max_iterations:
        log_message(f"WARNING: Reached max iterations ({max_iterations}), some folders may be incomplete", rank, "Done")

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n{'=' * 60}", flush=True)
    print(f"[{ts}][R{rank}] ALL STAGES COMPLETE (iterations={iteration + 1})", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
