from __future__ import annotations
"""Sample building pipeline — frame indexing, point clouds, projections, reference selection, writing."""

import imageio.v2 as imageio
from scipy import ndimage
from scripts.create_train_data._augmentation import augment_fg_overlay
from scripts.create_train_data._utils import SampleIndices, VideoGeometry, get_rank_info, shard_items, get_sample_naming, save_point_cloud_ply

# === sample_indices ===

"""
帧索引采样模块

从视频中采样训练样本的帧索引:
- P (Preceding): 前置帧，用于初始化 (支持1帧或9帧两种模式)
- T (Target): 目标帧，模型需要生成的帧
- C (Candidate): 候选帧，用于选择参考帧R

根据 LiveWorld 论文:
- 第一轮 (image-conditioned): 生成81帧，条件是1帧 first frame
- 后续轮 (clip-conditioned): 生成72帧，条件是9帧前一轮生成的帧

数据处理时同时准备1帧和9帧两种 preceding，训练时随机选择。
"""

from typing import Optional

import numpy as np



def sample_frame_indices(
    num_frames: int,
    N_target: int,
    M_pre: int,
    min_gap_for_candidates: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> SampleIndices:
    """
    采样帧索引

    随机选择起始点t0，然后:
    - preceding: [t0-M_pre, t0) 共M_pre帧 (用于9帧模式)
    - target: [t0, t0+N_target) 共N_target帧
    - candidate: 剩余所有帧 (排除P和T)

    注意: 1帧模式的 preceding 是 target 的第一帧 (t0)，
    这样可以模拟 image-conditioned 的情况。

    示例 (81帧视频, M_pre=9, N_target=33):
      若t0=40，则:
      - P9: [31, 32, ..., 39] (9帧)
      - P1: [40] (1帧，即target第一帧)
      - T: [40, 41, ..., 72] (33帧)
      - C: [0, ..., 30] + [73, ..., 80]
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if N_target <= 0 or M_pre <= 0:
        raise ValueError("N_target and M_pre must be positive")
    if num_frames < N_target + M_pre:
        raise ValueError("Not enough frames to sample training sample")

    if rng is None:
        rng = np.random.default_rng()

    t0_min = M_pre
    t0_max = num_frames - N_target
    if t0_min > t0_max:
        raise ValueError("Invalid sampling range for t0")

    t0 = int(rng.integers(t0_min, t0_max + 1))
    target_indices = list(range(t0, t0 + N_target))
    preceding_indices = list(range(t0 - M_pre, t0))

    preceding_set = set(preceding_indices)
    target_set = set(target_indices)
    candidate_indices = [i for i in range(num_frames) if i not in preceding_set and i not in target_set]

    if min_gap_for_candidates > 0:
        protected = sorted(preceding_indices + target_indices)
        def far_enough(idx: int) -> bool:
            return min(abs(idx - t) for t in protected) >= min_gap_for_candidates
        candidate_indices = [i for i in candidate_indices if far_enough(i)]

    return SampleIndices(
        t0=t0,
        preceding_indices=preceding_indices,
        target_indices=target_indices,
        candidate_indices=candidate_indices,
    )


# Backward compatibility alias
sample_episode_indices = sample_frame_indices

# === point_cloud ===

"""
点云构建模块

从深度图反投影构建3D点云，用于场景表示和投影
"""

from typing import Optional

import numpy as np

from scripts.create_train_data._geometry import (
    transform_points,
    unproject_depth_to_points,
    voxel_indices,
)


def build_scene_point_cloud(
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    dynamic_mask: Optional[np.ndarray] = None,
    voxel_size: float = 0.05,
    depth_max: Optional[float] = None,
):
    """
    从单帧深度图构建场景点云

    流程:
    1. 根据valid_mask和dynamic_mask过滤像素 (排除无效区域和动态物体)
    2. 过滤超过depth_max的像素 (排除天空等远处区域)
    3. 使用内参K将深度图反投影为相机坐标系下的3D点
    4. 使用c2w变换到世界坐标系
    5. 使用voxel下采样去除冗余点

    返回: (点云xyz, 点云颜色rgb)
    """
    if valid_mask is not None and valid_mask.shape != depth.shape:
        raise ValueError("valid_mask must match depth shape")
    if dynamic_mask is not None and dynamic_mask.shape != depth.shape:
        raise ValueError("dynamic_mask must match depth shape")

    mask = depth > 0
    if depth_max is not None:
        mask = mask & (depth < depth_max)
    if valid_mask is not None:
        mask = mask & valid_mask
    if dynamic_mask is not None:
        mask = mask & (~dynamic_mask)

    points_cam = unproject_depth_to_points(depth, K, mask=mask)
    if points_cam.size == 0:
        empty_xyz = np.zeros((0, 3), dtype=np.float32)
        if rgb is None:
            return empty_xyz, None
        empty_rgb = np.zeros((0, 3), dtype=np.uint8)
        return empty_xyz, empty_rgb

    points_world = transform_points(points_cam, c2w)
    if voxel_size > 0:
        vox = voxel_indices(points_world, voxel_size)
        _, unique_idx = np.unique(vox, axis=0, return_index=True)
        points_world = points_world[unique_idx]
    else:
        unique_idx = None

    if rgb is None:
        return points_world, None

    ys, xs = np.nonzero(mask)
    colors = rgb[ys, xs]
    if unique_idx is not None:
        colors = colors[unique_idx]
    return points_world, colors

# === reference_frames ===

"""
参考帧选择模块

基于空间IOU选择参考帧:
1. 将每帧深度图反投影为voxel occupancy
2. 计算candidate帧与target帧的voxel IOU
3. 选择IOU超过阈值的帧作为参考帧
"""

from typing import Iterable, Optional

import numpy as np

from scripts.create_train_data._geometry import transform_points, unproject_depth_to_points, voxel_indices


def occupancy_from_frame(
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    voxel_size: float,
    valid_mask: Optional[np.ndarray] = None,
    dynamic_mask: Optional[np.ndarray] = None,
) -> set[tuple[int, int, int]]:
    """从单帧深度图计算voxel occupancy集合"""
    mask = depth > 0
    if valid_mask is not None:
        mask = mask & valid_mask
    if dynamic_mask is not None:
        mask = mask & (~dynamic_mask)

    points_cam = unproject_depth_to_points(depth, K, mask=mask)
    if points_cam.size == 0:
        return set()

    points_world = transform_points(points_cam, c2w)
    vox = voxel_indices(points_world, voxel_size)
    return set(map(tuple, vox.tolist()))


def iou_occupancy(a: set[tuple[int, int, int]], b: set[tuple[int, int, int]]) -> float:
    """计算两个voxel集合的IOU (交集/并集)"""
    if not a and not b:
        return 0.0
    inter = a.intersection(b)
    union = a.union(b)
    if not union:
        return 0.0
    return float(len(inter)) / float(len(union))


class RefSelectionResult:
    """Result of reference frame selection with diagnostic info."""
    def __init__(
        self,
        indices: list[int],
        ious: list[float],
        stats: dict,
    ):
        self.indices = indices
        self.ious = ious
        self.stats = stats

    @property
    def count(self) -> int:
        return len(self.indices)

    def get_status_str(self) -> str:
        """Get a human-readable status string for logging."""
        if self.count > 0:
            return f"ref={self.count}, best_iou={self.stats['best_iou']:.3f}"
        else:
            reason = self.stats.get("no_ref_reason", "unknown")
            best = self.stats.get("best_iou", 0)
            thresh = self.stats.get("threshold", 0)
            if reason == "max_refs_zero":
                return "ref=0 (max_refs=0)"
            elif reason == "no_candidates":
                return "ref=0 (no candidates)"
            elif reason == "iou_below_threshold":
                return f"ref=0 (best_iou={best:.3f}<{thresh:.3f})"
            else:
                return f"ref=0 ({reason})"


def select_reference_frames(
    candidate_indices: Iterable[int],
    target_indices: Iterable[int],
    depths: np.ndarray,
    intrinsics: np.ndarray,
    poses_c2w: np.ndarray,
    voxel_size: float,
    stride: int,
    iou_threshold: float,
    max_refs: int,
    valid_masks: Optional[np.ndarray] = None,
    dynamic_masks: Optional[np.ndarray] = None,
    return_result: bool = False,
):
    """
    Select reference frames based on spatial overlap with target frames.

    Following LiveWorld paper (Algorithm 1): For each target frame, find the candidate
    with highest IOU. A candidate is selected as reference if its max IOU with any
    target frame exceeds the threshold.

    This is different from merging all target frames - we compute per-frame IOU
    which gives much higher overlap scores.

    Args:
        return_result: If True, returns RefSelectionResult with diagnostic info.
                      If False (default), returns (indices, ious) for backward compatibility.
    """
    stats = {
        "threshold": iou_threshold,
        "max_refs": max_refs,
        "stride": stride,
        "voxel_size": voxel_size,
    }

    if max_refs <= 0:
        stats["no_ref_reason"] = "max_refs_zero"
        stats["best_iou"] = 0.0
        if return_result:
            return RefSelectionResult([], [], stats)
        return [], []

    target_list = list(target_indices)
    candidates = list(candidate_indices)
    if stride > 1:
        candidates = candidates[::stride]

    stats["num_targets"] = len(target_list)
    stats["num_candidates"] = len(candidates)

    if not candidates:
        stats["no_ref_reason"] = "no_candidates"
        stats["best_iou"] = 0.0
        if return_result:
            return RefSelectionResult([], [], stats)
        return [], []

    # Pre-compute occupancy for all target frames
    target_occs = []
    for idx in target_list:
        occ = occupancy_from_frame(
            depth=depths[idx],
            K=intrinsics[idx],
            c2w=poses_c2w[idx],
            voxel_size=voxel_size,
            valid_mask=None if valid_masks is None else valid_masks[idx],
            dynamic_mask=None if dynamic_masks is None else dynamic_masks[idx],
        )
        target_occs.append(occ)

    # Pre-compute occupancy for all candidate frames
    candidate_occs = {}
    for idx in candidates:
        occ = occupancy_from_frame(
            depth=depths[idx],
            K=intrinsics[idx],
            c2w=poses_c2w[idx],
            voxel_size=voxel_size,
            valid_mask=None if valid_masks is None else valid_masks[idx],
            dynamic_mask=None if dynamic_masks is None else dynamic_masks[idx],
        )
        candidate_occs[idx] = occ

    # For each candidate, compute max IOU across all target frames
    # (Following LiveWorld paper: select candidate with highest spatial overlap)
    scored = []
    all_ious = []  # For debugging
    for c_idx in candidates:
        c_occ = candidate_occs[c_idx]
        max_iou = 0.0
        for t_occ in target_occs:
            iou = iou_occupancy(c_occ, t_occ)
            if iou > max_iou:
                max_iou = iou
        all_ious.append((c_idx, max_iou, len(c_occ)))
        if max_iou >= iou_threshold:
            scored.append((c_idx, max_iou))

    # Compute statistics
    best_iou = max(x[1] for x in all_ious) if all_ious else 0.0
    avg_iou = sum(x[1] for x in all_ious) / len(all_ious) if all_ious else 0.0
    stats["best_iou"] = best_iou
    stats["avg_iou"] = avg_iou

    if len(scored) == 0:
        stats["no_ref_reason"] = "iou_below_threshold"

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = scored[:max_refs]
    indices = [s[0] for s in selected]
    ious = [s[1] for s in selected]

    if return_result:
        return RefSelectionResult(indices, ious, stats)
    return indices, ious

# === dataset_writer ===

"""
数据集写入模块

将训练样本保存为文件:
- mp4视频: preceding_rgb, target_rgb, reference_rgb, scene_rgb, projection等
- json: 样本元信息 (帧索引、参考帧IOU等)
"""

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image



def save_frames(frames: np.ndarray, out_dir: str | Path, ext: str = "png") -> None:
    """保存帧序列为图片"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        img = frame
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(out_dir / f"{idx:05d}.{ext}")


def save_video(frames: np.ndarray, out_path: str | Path, fps: float = 16.0) -> None:
    """保存帧序列为mp4视频"""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    video = frames
    if video.dtype != np.uint8:
        video = np.clip(video, 0, 255).astype(np.uint8)
    imageio.mimsave(str(out_path), video, fps=fps, macro_block_size=None)


def save_projection_rgb_frames(proj: np.ndarray, out_dir: str | Path, ext: str = "png") -> None:
    if proj.ndim != 4 or proj.shape[-1] != 3:
        raise ValueError("proj must be (N,H,W,3)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(proj):
        img = np.clip(frame, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(out_dir / f"{idx:05d}.{ext}")


def save_projection_rgb_video(proj: np.ndarray, out_path: str | Path, fps: float = 16.0) -> None:
    if proj.ndim != 4 or proj.shape[-1] != 3:
        raise ValueError("proj must be (N,H,W,3)")
    save_video(proj, out_path, fps=fps)


def save_pose_txt(poses_c2w: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    lines = [str(len(poses_c2w))]
    for idx, pose in enumerate(poses_c2w):
        vals = " ".join(f"{v:.6f}" for v in pose.reshape(-1).tolist())
        lines.append(f"{idx} {vals}")
    out_path.write_text("\n".join(lines))


def save_intrinsics_txt(intrinsics: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    lines = [str(len(intrinsics))]
    for idx, K in enumerate(intrinsics):
        vals = " ".join(f"{v:.6f}" for v in K.reshape(-1).tolist())
        lines.append(f"{idx} {vals}")
    out_path.write_text("\n".join(lines))


def save_training_sample(
    sample: dict,
    out_dir: str | Path,
    projection_channels: list[str],
    save_npz: bool = True,
    fps: float = 16.0,
    save_frame_dirs: bool = False,
    naming: str = "figure",
    name_prefix: str = "",
    videos_only: bool = False,
    fg_overlay_aug: dict | None = None,
) -> None:
    """
    Save training sample outputs.

    Simplified output structure:
    - Videos: preceding_rgb_9 (9帧), preceding_rgb_1 (1帧), target_rgb, reference_rgb,
              preceding_scene_proj_rgb, target_scene_proj_rgb
    - sample.json: Contains all meta info (scene_idx, R_idx, R_iou, etc.)

    根据 LiveWorld 论文:
    - P9: 9帧 preceding (clip-conditioned mode, 用于后续轮)
    - P1: 1帧 preceding = target第一帧 (image-conditioned mode, 用于第一轮)

    No longer saves: .npz, .ply, pose/intrinsics txt files (these can be
    reconstructed from geometry.npz + sample.json)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = get_sample_naming(naming)
    prefix = name_prefix

    # Save videos - 两种 preceding 模式
    save_video(sample["P9_rgb"], out_dir / f"{prefix}{names.preceding_rgb_9}.mp4", fps=fps)
    save_video(sample["P1_rgb"], out_dir / f"{prefix}{names.preceding_rgb_1}.mp4", fps=fps)
    save_video(sample["T_rgb"], out_dir / f"{prefix}{names.target_rgb}.mp4", fps=fps)
    if sample["R_rgb"].shape[0] > 0:
        save_video(sample["R_rgb"], out_dir / f"{prefix}{names.reference_rgb}.mp4", fps=fps)
        if "R_scene_rgb" in sample and sample["R_scene_rgb"].shape[0] > 0:
            save_video(sample["R_scene_rgb"], out_dir / f"{prefix}{names.reference_scene_rgb}.mp4", fps=fps)
        if "R_scene_rgb_orig" in sample and sample["R_scene_rgb_orig"].shape[0] > 0:
            save_video(sample["R_scene_rgb_orig"], out_dir / f"{prefix}{names.reference_scene_rgb_orig}.mp4", fps=fps)

    # Save projection RGB videos
    channels = list(projection_channels)
    cursor = 0
    rgb_slice = None
    for name in channels:
        if name == "rgb":
            rgb_slice = slice(cursor, cursor + 3)
            cursor += 3
        else:
            cursor += 1

    if rgb_slice is not None:
        # Background scene projections (prefer explicit bg keys if provided)
        if "proj_P9_bg" in sample:
            preceding_proj_rgb_9 = sample["proj_P9_bg"][..., rgb_slice]
        elif "proj_P9" in sample:
            preceding_proj_rgb_9 = sample["proj_P9"][..., rgb_slice]
        elif "proj_P" in sample:
            # Backward compatibility
            preceding_proj_rgb_9 = sample["proj_P"][..., rgb_slice]
        else:
            preceding_proj_rgb_9 = None

        if preceding_proj_rgb_9 is not None:
            save_projection_rgb_video(
                preceding_proj_rgb_9,
                out_dir / f"{prefix}{names.preceding_scene_proj_rgb_9}.mp4",
                fps=fps,
            )

        if "proj_P1_bg" in sample:
            preceding_proj_rgb_1 = sample["proj_P1_bg"][..., rgb_slice]
        elif "proj_P1" in sample:
            preceding_proj_rgb_1 = sample["proj_P1"][..., rgb_slice]
        else:
            preceding_proj_rgb_1 = None

        if preceding_proj_rgb_1 is not None:
            save_projection_rgb_video(
                preceding_proj_rgb_1,
                out_dir / f"{prefix}{names.preceding_scene_proj_rgb_1}.mp4",
                fps=fps,
            )

        if "proj_T_bg" in sample:
            target_proj_rgb = sample["proj_T_bg"][..., rgb_slice]
        else:
            target_proj_rgb = sample["proj_T"][..., rgb_slice]
        save_projection_rgb_video(
            target_proj_rgb,
            out_dir / f"{prefix}{names.target_scene_proj_rgb}.mp4",
            fps=fps,
        )

        # Non-augmented target scene projection
        if "proj_T_bg_orig" in sample:
            target_proj_rgb_orig = sample["proj_T_bg_orig"][..., rgb_slice]
            save_projection_rgb_video(
                target_proj_rgb_orig,
                out_dir / f"{prefix}{names.target_scene_proj_rgb_orig}.mp4",
                fps=fps,
            )

        # Foreground-only projections (augmented foreground on black)
        if "proj_P9_fg" in sample:
            save_projection_rgb_video(
                sample["proj_P9_fg"],
                out_dir / f"{prefix}{names.preceding_proj_fg_rgb_9}.mp4",
                fps=fps,
            )
        if "proj_P1_fg" in sample:
            save_projection_rgb_video(
                sample["proj_P1_fg"],
                out_dir / f"{prefix}{names.preceding_proj_fg_rgb_1}.mp4",
                fps=fps,
            )
        if "proj_T_fg" in sample:
            save_projection_rgb_video(
                sample["proj_T_fg"],
                out_dir / f"{prefix}{names.target_proj_fg_rgb}.mp4",
                fps=fps,
            )

        # Scene+fg overlay projections (fg non-black pixels overlaid on scene)
        def _overlay_fg_on_scene(scene: np.ndarray, fg: np.ndarray) -> np.ndarray:
            scene_u8 = np.clip(scene, 0, 255).astype(np.uint8)
            fg_u8 = np.clip(fg, 0, 255).astype(np.uint8)
            fg_mask = (fg_u8 > 10).any(axis=-1, keepdims=True)
            return np.where(fg_mask, fg_u8, scene_u8)

        def _maybe_aug_fg(fg: np.ndarray) -> np.ndarray:
            if fg_overlay_aug is None:
                return fg
            fg_u8 = np.clip(fg, 0, 255).astype(np.uint8)
            return augment_fg_overlay(fg_u8, **fg_overlay_aug).astype(fg.dtype)

        if "proj_P9_fg" in sample and preceding_proj_rgb_9 is not None:
            overlay_p9 = _overlay_fg_on_scene(preceding_proj_rgb_9, _maybe_aug_fg(sample["proj_P9_fg"]))
            save_projection_rgb_video(
                overlay_p9,
                out_dir / f"{prefix}{names.preceding_scene_proj_fg_overlay_rgb_9}.mp4",
                fps=fps,
            )
        if "proj_P1_fg" in sample and preceding_proj_rgb_1 is not None:
            overlay_p1 = _overlay_fg_on_scene(preceding_proj_rgb_1, _maybe_aug_fg(sample["proj_P1_fg"]))
            save_projection_rgb_video(
                overlay_p1,
                out_dir / f"{prefix}{names.preceding_scene_proj_fg_overlay_rgb_1}.mp4",
                fps=fps,
            )
        if "proj_T_fg" in sample:
            overlay_t = _overlay_fg_on_scene(target_proj_rgb, _maybe_aug_fg(sample["proj_T_fg"]))
            save_projection_rgb_video(
                overlay_t,
                out_dir / f"{prefix}{names.target_scene_proj_fg_overlay_rgb}.mp4",
                fps=fps,
            )

    # Save sample.json (meta info for reconstruction)
    meta = dict(sample.get("meta", {}))
    meta["naming"] = naming
    meta_path = out_dir / f"{prefix}sample.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# Backward compatibility alias
save_episode_dir = save_training_sample

# === sample_builder ===

"""
训练样本构建模块

核心功能:
1. 从视频几何信息中采样帧索引 (preceding/target/reference)
2. 构建场景点云 (排除动态物体)
3. 将点云投影到各帧生成scene projection
4. 选择参考帧 (基于空间IOU)
"""

from typing import Optional, Iterable

import cv2
import numpy as np


from scripts.create_train_data._projection import render_projection, recolor_projection


def resize_frames(frames: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """将帧序列resize到目标尺寸 (H, W)"""
    target_h, target_w = target_size
    if frames.shape[1] == target_h and frames.shape[2] == target_w:
        return frames
    resized = []
    for frame in frames:
        resized.append(cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR))
    return np.stack(resized, axis=0)


def scale_intrinsics(K: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """缩放相机内参矩阵以适应不同分辨率"""
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy
    return K_scaled


def apply_scene_mask(
    frames: np.ndarray,
    dynamic_masks: Optional[np.ndarray],
    valid_masks: Optional[np.ndarray],
) -> np.ndarray:
    """应用场景mask: 将动态物体区域置为黑色，只保留静态背景"""
    if dynamic_masks is None and valid_masks is None:
        return frames.copy()
    masked = frames.copy()
    num_frames = frames.shape[0]
    for idx in range(num_frames):
        keep = np.ones(frames.shape[1:3], dtype=bool)
        if valid_masks is not None:
            keep &= valid_masks[idx]
        if dynamic_masks is not None:
            keep &= ~dynamic_masks[idx]
        masked[idx][~keep] = 0
    return masked


def extract_foreground(
    frames: np.ndarray,
    foreground_masks: np.ndarray,
) -> np.ndarray:
    """
    提取前景区域RGB，背景置为黑色。

    Args:
        frames: RGB帧序列 (N, H, W, 3)
        foreground_masks: 前景mask (N, H, W), True表示前景

    Returns:
        前景RGB (N, H, W, 3)，背景区域为黑色
    """
    foreground = frames.copy()
    for idx in range(frames.shape[0]):
        foreground[idx][~foreground_masks[idx]] = 0
    return foreground


def augment_foreground(
    foreground: np.ndarray,
    foreground_masks: np.ndarray,
    depths: np.ndarray,
    dot_ratio_min: float = 0.7,
    dot_ratio_max: float = 0.9,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    对前景进行augmentation: 根据连通区域深度决定scatter黑点密度。

    - 检测mask中的连通区域（白色团）
    - 离镜头越近（深度小）黑点越密集，离镜头越远（深度大）黑点越稀疏
    - 深度归一化: 每帧的最大深度作为归一化基准
    - 每帧独立随机采样黑点
    - dot_ratio每帧随机在[dot_ratio_min, dot_ratio_max]之间取值

    Args:
        foreground: 前景RGB (N, H, W, 3)
        foreground_masks: 前景mask (N, H, W)
        depths: 深度图 (N, H, W)
        dot_ratio_min: scatter黑点比例最小值
        dot_ratio_max: scatter黑点比例最大值
        rng: 随机数生成器

    Returns:
        augmented前景RGB
    """
    if rng is None:
        rng = np.random.default_rng()

    N, H, W, _ = foreground.shape
    augmented = foreground.copy()


    # 对每帧独立处理
    for frame_idx in range(N):
        frame_mask = foreground_masks[frame_idx]
        frame_depth = depths[frame_idx]

        if not frame_mask.any():
            continue

        # 检测当前帧的连通区域
        labeled, num_features = ndimage.label(frame_mask)

        if num_features == 0:
            continue

        # 每帧随机选择dot_ratio
        dot_ratio = rng.uniform(dot_ratio_min, dot_ratio_max)

        # 当前帧的最大深度用于归一化
        max_depth = frame_depth.max()
        if max_depth <= 0:
            max_depth = 1.0

        # 对每个连通区域独立采样
        for comp_id in range(1, num_features + 1):
            comp_mask = labeled == comp_id
            comp_pixels = np.where(comp_mask)
            num_pixels = len(comp_pixels[0])

            if num_pixels == 0:
                continue

            # 计算该区域的平均深度
            mean_depth = frame_depth[comp_mask].mean()

            # 根据深度计算密度: 深度越小(近)越密集，深度越大(远)越稀疏
            depth_ratio = mean_depth / max_depth  # 0=近, 1=远
            density = dot_ratio * (1.0 - depth_ratio)  # 近=密集, 远=稀疏

            # 纯随机采样该区域内的像素
            num_dots = int(num_pixels * density)
            if num_dots > 0:
                dot_indices = rng.choice(num_pixels, size=num_dots, replace=False)
                augmented[frame_idx, comp_pixels[0][dot_indices], comp_pixels[1][dot_indices]] = 0

    return augmented


def composite_foreground_on_projection(
    projection: np.ndarray,
    foreground_aug: np.ndarray,
    foreground_masks: np.ndarray,
) -> np.ndarray:
    """
    将augmented前景贴到projection video上。

    Args:
        projection: scene projection RGB (N, H, W, 3)
        foreground_aug: augmented前景RGB (N, H, W, 3)
        foreground_masks: 前景mask (N, H, W)

    Returns:
        合成后的projection RGB
    """
    composited = projection.copy()
    for idx in range(projection.shape[0]):
        mask = foreground_masks[idx]
        composited[idx][mask] = foreground_aug[idx][mask]
    return composited


def _projection_channel_slices(channels: Iterable[str]) -> dict[str, slice]:
    """Return channel slices for projection tensor."""
    slices: dict[str, slice] = {}
    cursor = 0
    for ch in channels:
        if ch == "rgb":
            slices["rgb"] = slice(cursor, cursor + 3)
            cursor += 3
        else:
            slices[ch] = slice(cursor, cursor + 1)
            cursor += 1
    return slices


def _block_dropout_mask(
    height: int,
    width: int,
    ratio: float,
    block: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if ratio <= 0.0 or block <= 1:
        return np.zeros((height, width), dtype=bool)
    grid_h = max(1, int(np.ceil(height / float(block))))
    grid_w = max(1, int(np.ceil(width / float(block))))
    small = rng.random((grid_h, grid_w)) < ratio
    mask = np.repeat(np.repeat(small, block, axis=0), block, axis=1)
    return mask[:height, :width]


def augment_source_rgb(
    img: np.ndarray,
    rng: np.random.Generator,
    config,
) -> np.ndarray:
    """
    Augment the reference frame RGB before using it to build the point cloud.

    Depth estimation and camera params use the ORIGINAL image.
    Only the RGB fed into the point cloud (and thus projected) is augmented.

    Augmentations:
    - Brightness shift (mostly brighter)
    - Gaussian blur
    - Local pixel displacement (elastic-like distortion)
    - Directional smear (motion-blur kernel)

    Args:
        img: (H, W, 3) uint8 RGB image
        rng: numpy random generator
        config with src_aug_* fields

    Returns:
        Augmented (H, W, 3) uint8 image
    """
    out = img.copy()
    h, w = out.shape[:2]

    # 1. Brightness
    bmin = getattr(config, "src_aug_brightness_min", 1.0)
    bmax = getattr(config, "src_aug_brightness_max", 1.0)
    if bmax > bmin:
        factor = float(rng.uniform(bmin, bmax))
        out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # 2. Gaussian blur
    sigma_min = getattr(config, "src_aug_blur_sigma_min", 0.0)
    sigma_max = getattr(config, "src_aug_blur_sigma_max", 0.0)
    if sigma_max > 0:
        sigma = float(rng.uniform(sigma_min, sigma_max))
        if sigma > 0.3:
            out = cv2.GaussianBlur(out, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)

    # 3. Local pixel displacement (elastic-like distortion)
    disp_max = getattr(config, "src_aug_displacement_max", 0)
    if disp_max > 0:
        # Generate smooth displacement field
        dx = rng.uniform(-disp_max, disp_max, size=(h, w)).astype(np.float32)
        dy = rng.uniform(-disp_max, disp_max, size=(h, w)).astype(np.float32)
        # Smooth the displacement field to make it locally coherent
        ksize = max(disp_max * 6 + 1, 7)
        if ksize % 2 == 0:
            ksize += 1
        dx = cv2.GaussianBlur(dx, (ksize, ksize), sigmaX=disp_max * 2)
        dy = cv2.GaussianBlur(dy, (ksize, ksize), sigmaX=disp_max * 2)
        # Build remap coordinates
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = grid_x + dx
        map_y = grid_y + dy
        out = cv2.remap(out, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 4. Directional smear (motion blur)
    smear_prob = getattr(config, "src_aug_smear_prob", 0.0)
    smear_len_min = getattr(config, "src_aug_smear_length_min", 3)
    smear_len_max = getattr(config, "src_aug_smear_length_max", 15)
    if smear_prob > 0 and rng.random() < smear_prob:
        length = int(rng.integers(smear_len_min, smear_len_max + 1))
        if length >= 2:
            angle = float(rng.uniform(0, 180))
            kernel = np.zeros((length, length), dtype=np.float32)
            center = length // 2
            cos_a = np.cos(np.radians(angle))
            sin_a = np.sin(np.radians(angle))
            for i in range(length):
                offset = i - center
                x = int(round(center + offset * cos_a))
                y = int(round(center + offset * sin_a))
                if 0 <= x < length and 0 <= y < length:
                    kernel[y, x] = 1.0
            kernel /= kernel.sum() + 1e-8
            out = cv2.filter2D(out, -1, kernel)

    # 5. Random regional blur (randomly blur a few rectangular patches)
    patch_blur_prob = getattr(config, "src_aug_patch_blur_prob", 0.0)
    if patch_blur_prob > 0 and rng.random() < patch_blur_prob:
        n_patches = int(rng.integers(1, getattr(config, "src_aug_patch_blur_max_patches", 3) + 1))
        min_size = getattr(config, "src_aug_patch_blur_min_size", 0.05)
        max_size = getattr(config, "src_aug_patch_blur_max_size", 0.25)
        blur_sigma_min = getattr(config, "src_aug_patch_blur_sigma_min", 3.0)
        blur_sigma_max = getattr(config, "src_aug_patch_blur_sigma_max", 10.0)
        for _ in range(n_patches):
            pw = int(rng.uniform(min_size, max_size) * w)
            ph = int(rng.uniform(min_size, max_size) * h)
            px = int(rng.integers(0, max(w - pw, 1)))
            py = int(rng.integers(0, max(h - ph, 1)))
            sigma = float(rng.uniform(blur_sigma_min, blur_sigma_max))
            patch = out[py:py+ph, px:px+pw]
            out[py:py+ph, px:px+pw] = cv2.GaussianBlur(patch, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # 6. Random regional fill/inpaint (fill patches with nearby color or constant)
    patch_fill_prob = getattr(config, "src_aug_patch_fill_prob", 0.0)
    if patch_fill_prob > 0 and rng.random() < patch_fill_prob:
        n_patches = int(rng.integers(1, getattr(config, "src_aug_patch_fill_max_patches", 3) + 1))
        min_size = getattr(config, "src_aug_patch_fill_min_size", 0.03)
        max_size = getattr(config, "src_aug_patch_fill_max_size", 0.15)
        for _ in range(n_patches):
            pw = int(rng.uniform(min_size, max_size) * w)
            ph = int(rng.uniform(min_size, max_size) * h)
            px = int(rng.integers(0, max(w - pw, 1)))
            py = int(rng.integers(0, max(h - ph, 1)))
            # Randomly choose fill strategy
            strategy = int(rng.integers(0, 3))
            if strategy == 0:
                # Mean color of the patch region
                fill_color = out[py:py+ph, px:px+pw].mean(axis=(0, 1)).astype(np.uint8)
                out[py:py+ph, px:px+pw] = fill_color
            elif strategy == 1:
                # Heavy blur (simulate inpaint-like smear)
                patch = out[py:py+ph, px:px+pw]
                k = max(pw, ph) | 1  # ensure odd
                out[py:py+ph, px:px+pw] = cv2.GaussianBlur(patch, (k, k), sigmaX=k/2)
            else:
                # Random noise fill
                noise = rng.integers(0, 256, size=(ph, pw, 3), dtype=np.uint8)
                # Blend with heavy blur of original to keep rough color
                patch = out[py:py+ph, px:px+pw]
                blurred = cv2.GaussianBlur(patch, (0, 0), sigmaX=max(pw, ph) / 3)
                out[py:py+ph, px:px+pw] = np.clip(
                    blurred.astype(np.float32) * 0.7 + noise.astype(np.float32) * 0.3,
                    0, 255
                ).astype(np.uint8)

    return out


def build_training_sample(
    geometry: VideoGeometry,
    config,
    dynamic_masks: Optional[np.ndarray] = None,
    foreground_masks: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    projection_fill_kernel: int | None = None,
    original_frames: Optional[np.ndarray] = None,
    output_size: Optional[tuple[int, int]] = None,
):
    """
    从视频几何信息构建训练样本

    流程:
    1. sample_frame_indices: 随机采样t0，确定P(preceding)/T(target)/C(candidate)帧索引
    2. build_scene_point_cloud: 从某个candidate帧构建场景点云(排除动态物体)
    3. render_projection: 将点云投影到P和T帧，生成scene projection
    4. select_reference_frames: 基于空间IOU选择参考帧R

    输出字典包含:
    - P_rgb/T_rgb/R_rgb: 前置帧/目标帧/参考帧的RGB
    - proj_P/proj_T: 点云投影到P/T帧的结果
    - meta: 帧索引、参考帧选择统计等元信息
    """
    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    num_frames = geometry.frames.shape[0]
    indices = sample_frame_indices(
        num_frames=num_frames,
        N_target=config.N_target,
        M_pre=config.M_pre,
        min_gap_for_candidates=config.min_gap_for_candidates,
        rng=rng,
    )

    if not indices.candidate_indices:
        raise RuntimeError("No candidate frames available for scene selection.")

    # 随机选择一个candidate帧来构建场景点云
    scene_idx = int(rng.choice(indices.candidate_indices))

    # # dzc: 总是使用target的第一帧构建点云，符合推理。
    # scene_idx = int(indices.t0)

    H, W = geometry.frames.shape[1:3]  # MapAnything处理后的分辨率
    # NOTE: Hole filling is disabled to preserve geometry.
    fill_kernel = 0

    # 确定投影输出尺寸
    if output_size is not None:
        proj_h, proj_w = output_size
    elif geometry.original_size is not None:
        proj_h, proj_w = geometry.original_size
    else:
        proj_h, proj_w = H, W

    # 计算内参缩放比例 (从处理分辨率到输出分辨率)
    scale_x = proj_w / W
    scale_y = proj_h / H

    # 准备构建点云所需的几何/图像（可选上采样到输出尺寸）
    scene_depth = geometry.depths[scene_idx]
    scene_K = geometry.intrinsics[scene_idx]
    scene_rgb = geometry.frames[scene_idx]
    scene_valid_mask = None if geometry.masks is None else geometry.masks[scene_idx]
    scene_dynamic_mask = None if dynamic_masks is None else dynamic_masks[scene_idx]

    if config.upsample_geometry_to_output and output_size is not None:
        scene_depth = cv2.resize(scene_depth, (proj_w, proj_h), interpolation=cv2.INTER_NEAREST)
        scene_K = scale_intrinsics(scene_K, scale_x, scale_y)
        if original_frames is not None:
            scene_rgb = np.asarray(original_frames)[scene_idx]
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

    # Source-image augmentation: always augment RGB before building point cloud
    # Depth and camera params are from the ORIGINAL image (already computed by Stream3R).
    scene_rgb_original = scene_rgb.copy()  # keep original for building non-augmented projections
    src_aug_applied = False
    scene_rgb_augmented_img = None
    if getattr(config, "src_aug_enable", False):
        scene_rgb = augment_source_rgb(scene_rgb, rng, config)
        src_aug_applied = True
        scene_rgb_augmented_img = scene_rgb.copy()

    # 构建场景点云: 从该帧的深度图反投影3D点，排除动态物体
    scene_xyz, scene_rgb = build_scene_point_cloud(
        depth=scene_depth,
        K=scene_K,
        c2w=geometry.poses_c2w[scene_idx],
        rgb=scene_rgb,
        valid_mask=scene_valid_mask,
        dynamic_mask=scene_dynamic_mask,
        voxel_size=config.scene_voxel_size,
    )

    if scene_xyz.size == 0:
        raise RuntimeError("Scene point cloud is empty; check depth/pose quality.")

    # P1 索引: preceding 的最后一帧 (用于 image-conditioned 模式)
    p1_idx = indices.preceding_indices[-1]

    need_zbuffer = src_aug_applied and "rgb" in config.projection_channels

    # P9: preceding 帧逐帧投影
    preceding_proj_list = []
    for idx in indices.preceding_indices:
        K_proj = scale_intrinsics(geometry.intrinsics[idx], scale_x, scale_y)
        proj = render_projection(
            scene_xyz, K_proj, geometry.poses_c2w[idx],
            (proj_h, proj_w), config.projection_channels,
            colors=scene_rgb, fill_holes_kernel=fill_kernel,
        )
        preceding_proj_list.append(proj)
    preceding_proj_9 = np.stack(preceding_proj_list, axis=0).astype(np.float32)

    # P1: 直接复用P9的最后一帧，不重新渲染
    preceding_proj_1 = preceding_proj_9[-1:].copy()  # [1, H, W, C]

    # Target: 逐帧投影，需要时缓存 z-buffer 用于 recolor
    target_proj_list = []
    target_zbufs = []
    for idx in indices.target_indices:
        K_proj = scale_intrinsics(geometry.intrinsics[idx], scale_x, scale_y)
        result = render_projection(
            scene_xyz, K_proj, geometry.poses_c2w[idx],
            (proj_h, proj_w), config.projection_channels,
            colors=scene_rgb, fill_holes_kernel=fill_kernel,
            return_zbuffer=need_zbuffer,
        )
        if need_zbuffer:
            proj, zbuf = result
            target_zbufs.append(zbuf)
        else:
            proj = result
        target_proj_list.append(proj)
    target_proj = np.stack(target_proj_list, axis=0).astype(np.float32)

    # Keep background-only projections before any foreground compositing
    proj_P9_bg = preceding_proj_9.copy()
    proj_P1_bg = preceding_proj_1.copy()
    proj_T_bg = target_proj.copy()

    # Build non-augmented target projections (original RGB, no augmentation)
    # 复用z-buffer几何信息，只替换颜色，避免重复矩阵乘法和排序
    if src_aug_applied:
        _, scene_colors_orig = build_scene_point_cloud(
            depth=scene_depth, K=scene_K, c2w=geometry.poses_c2w[scene_idx],
            rgb=scene_rgb_original, valid_mask=scene_valid_mask,
            dynamic_mask=scene_dynamic_mask, voxel_size=config.scene_voxel_size,
        )
        if target_zbufs:
            proj_T_bg_orig_list = []
            for zbuf in target_zbufs:
                proj = recolor_projection(zbuf, config.projection_channels, scene_colors_orig)
                proj_T_bg_orig_list.append(proj)
            proj_T_bg_orig = np.stack(proj_T_bg_orig_list, axis=0).astype(np.float32)
        else:
            proj_T_bg_orig = proj_T_bg
    else:
        proj_T_bg_orig = proj_T_bg

    # 选择参考帧: 计算candidate帧与target帧的空间IOU，选择重叠度高的帧作为参考
    # 使用较大的voxel_size来容忍深度噪声
    iou_voxel_size = config.ref_iou_voxel_size
    if iou_voxel_size is None:
        iou_voxel_size = config.scene_voxel_size * 10 if config.scene_voxel_size > 0 else 0.01

    ref_result: RefSelectionResult = select_reference_frames(
        candidate_indices=indices.candidate_indices,
        target_indices=indices.target_indices,
        depths=geometry.depths,
        intrinsics=geometry.intrinsics,
        poses_c2w=geometry.poses_c2w,
        voxel_size=iou_voxel_size,
        stride=config.K_ref_stride,
        iou_threshold=config.eps_iou,
        max_refs=config.max_refs,
        valid_masks=geometry.masks,
        dynamic_masks=dynamic_masks,
        return_result=True,
    )
    reference_indices = ref_result.indices
    reference_ious = ref_result.ious

    # Determine output size and source frames
    # If original_frames provided, use those for RGB output; otherwise use geometry.frames
    if original_frames is not None:
        # Ensure src_frames is a numpy array (load_video_frames returns list)
        src_frames = np.asarray(original_frames)
    else:
        src_frames = geometry.frames

    # Determine target output size
    if output_size is not None:
        out_h, out_w = output_size
    elif geometry.original_size is not None:
        out_h, out_w = geometry.original_size
    else:
        out_h, out_w = H, W  # Fall back to processed size

    # Extract RGB frames from source (original resolution if available)
    # P9: 9帧 preceding (clip-conditioned mode, 用于后续轮)
    preceding_rgb_9 = src_frames[indices.preceding_indices]
    # P1: 1帧 preceding = P9最后一帧 (image-conditioned mode, 用于第一轮)
    # p1_idx already defined above for projection
    preceding_rgb_1 = src_frames[[p1_idx]]

    target_rgb = src_frames[indices.target_indices]
    if reference_indices:
        reference_rgb = src_frames[reference_indices]
    else:
        reference_rgb = np.zeros((0, out_h, out_w, 3), dtype=np.uint8)

    # Resize RGB frames to output size if needed
    preceding_rgb_9 = resize_frames(preceding_rgb_9, (out_h, out_w))
    preceding_rgb_1 = resize_frames(preceding_rgb_1, (out_h, out_w))
    target_rgb = resize_frames(target_rgb, (out_h, out_w))
    if reference_rgb.shape[0] > 0:
        reference_rgb = resize_frames(reference_rgb, (out_h, out_w))

    reference_scene = None
    reference_scene_orig = None

    # Note: Projections are already rendered at output resolution (proj_h, proj_w)
    # using scaled intrinsics, so no resizing needed

    # Apply scene mask (need to resize masks first if sizes differ)
    # For P9 (9帧 preceding)
    if dynamic_masks is not None:
        dm_preceding_9 = dynamic_masks[indices.preceding_indices]
        dm_preceding_1 = dynamic_masks[[p1_idx]]
        dm_target = dynamic_masks[indices.target_indices]
        # Resize masks to output size
        dm_preceding_9 = np.stack([cv2.resize(m.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool) for m in dm_preceding_9])
        dm_preceding_1 = np.stack([cv2.resize(m.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool) for m in dm_preceding_1])
        dm_target = np.stack([cv2.resize(m.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool) for m in dm_target])
    else:
        dm_preceding_9 = None
        dm_preceding_1 = None
        dm_target = None

    if geometry.masks is not None:
        vm_preceding_9 = geometry.masks[indices.preceding_indices]
        vm_preceding_1 = geometry.masks[[p1_idx]]
        vm_target = geometry.masks[indices.target_indices]
        # Resize masks to output size
        vm_preceding_9 = np.stack([cv2.resize(m.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool) for m in vm_preceding_9])
        vm_preceding_1 = np.stack([cv2.resize(m.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool) for m in vm_preceding_1])
        vm_target = np.stack([cv2.resize(m.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool) for m in vm_target])
    else:
        vm_preceding_9 = None
        vm_preceding_1 = None
        vm_target = None

    # Default foreground projections: black (no foreground)
    preceding_fg_9 = np.zeros_like(preceding_rgb_9)
    preceding_fg_1 = np.zeros_like(preceding_rgb_1)
    target_fg = np.zeros_like(target_rgb)
    has_fg = False
    proj_P9_fg = np.zeros((preceding_rgb_9.shape[0], out_h, out_w, 3), dtype=np.uint8)
    proj_P1_fg = np.zeros((preceding_rgb_1.shape[0], out_h, out_w, 3), dtype=np.uint8)
    proj_T_fg = np.zeros((target_rgb.shape[0], out_h, out_w, 3), dtype=np.uint8)

    # 提取前景 (非天空的动态物体) 并合成到scene projection上
    # foreground_masks: 合并除sky以外的所有类别mask
    if foreground_masks is not None:
        # Resize foreground masks to output size
        fg_preceding_9 = foreground_masks[indices.preceding_indices]
        fg_preceding_1 = foreground_masks[[p1_idx]]
        fg_target = foreground_masks[indices.target_indices]
        fg_preceding_9 = np.stack([cv2.resize(m.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool) for m in fg_preceding_9])
        fg_preceding_1 = np.stack([cv2.resize(m.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool) for m in fg_preceding_1])
        fg_target = np.stack([cv2.resize(m.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool) for m in fg_target])
        has_fg = bool(fg_preceding_9.any() or fg_preceding_1.any() or fg_target.any())

        if reference_indices:
            fg_reference = foreground_masks[reference_indices]
            fg_reference = np.stack([cv2.resize(m.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool) for m in fg_reference])
            reference_scene = reference_rgb.copy()
            reference_scene[fg_reference] = 0
            reference_scene_orig = reference_scene.copy()  # pure non-augmented version

            # Augment reference scene frames (same augmentations as source image, per-frame independent)
            ref_scene_aug_prob = getattr(config, "ref_scene_aug_prob", 0.0)
            if ref_scene_aug_prob > 0:
                for fi in range(reference_scene.shape[0]):
                    if rng.random() < ref_scene_aug_prob:
                        reference_scene[fi] = augment_source_rgb(reference_scene[fi], rng, config)

        # 提取前景RGB
        preceding_fg_9 = extract_foreground(preceding_rgb_9, fg_preceding_9)
        preceding_fg_1 = extract_foreground(preceding_rgb_1, fg_preceding_1)
        target_fg = extract_foreground(target_rgb, fg_target)

        # 获取深度数据并resize到输出尺寸 (用于基于深度的scatter密度)
        depth_preceding_9 = geometry.depths[indices.preceding_indices]
        depth_preceding_1 = geometry.depths[[p1_idx]]
        depth_target = geometry.depths[indices.target_indices]
        depth_preceding_9 = np.stack([cv2.resize(d, (out_w, out_h), interpolation=cv2.INTER_NEAREST) for d in depth_preceding_9])
        depth_preceding_1 = np.stack([cv2.resize(d, (out_w, out_h), interpolation=cv2.INTER_NEAREST) for d in depth_preceding_1])
        depth_target = np.stack([cv2.resize(d, (out_w, out_h), interpolation=cv2.INTER_NEAREST) for d in depth_target])

        # Augment前景: 根据深度决定scatter黑点密度（近密集，远稀疏）
        # dot_ratio每帧随机在[min, max]之间取值
        preceding_fg_aug_9 = augment_foreground(
            preceding_fg_9, fg_preceding_9, depth_preceding_9,
            dot_ratio_min=config.fg_dot_ratio_min,
            dot_ratio_max=config.fg_dot_ratio_max,
            rng=rng,
        )
        preceding_fg_aug_1 = augment_foreground(
            preceding_fg_1, fg_preceding_1, depth_preceding_1,
            dot_ratio_min=config.fg_dot_ratio_min,
            dot_ratio_max=config.fg_dot_ratio_max,
            rng=rng,
        )
        target_fg_aug = augment_foreground(
            target_fg, fg_target, depth_target,
            dot_ratio_min=config.fg_dot_ratio_min,
            dot_ratio_max=config.fg_dot_ratio_max,
            rng=rng,
        )

        # Foreground-only projections (augmented foreground on black)
        proj_P9_fg = preceding_fg_aug_9.astype(np.uint8)
        proj_P1_fg = preceding_fg_aug_1.astype(np.uint8)
        proj_T_fg = target_fg_aug.astype(np.uint8)

        # 将augmented前景合成到scene projection上
        # 流程: scene先投影 -> 再把aug_fg贴上去
        # proj_P9/P1/T 已经是scene projection (float32 [0,255])
        if config.add_fg_to_projection:
            proj_P9_rgb = preceding_proj_9[..., :3].astype(np.uint8)  # scene projection RGB
            proj_P1_rgb = preceding_proj_1[..., :3].astype(np.uint8)
            proj_T_rgb = target_proj[..., :3].astype(np.uint8)

            # 在scene projection上贴aug_fg，然后直接覆盖原始projection
            proj_P9_rgb = composite_foreground_on_projection(proj_P9_rgb, preceding_fg_aug_9, fg_preceding_9)
            proj_P1_rgb = composite_foreground_on_projection(proj_P1_rgb, preceding_fg_aug_1, fg_preceding_1)
            proj_T_rgb = composite_foreground_on_projection(proj_T_rgb, target_fg_aug, fg_target)

            # 直接覆盖proj_*，统一输出
            preceding_proj_9 = proj_P9_rgb.astype(np.float32)
            preceding_proj_1 = proj_P1_rgb.astype(np.float32)
            target_proj = proj_T_rgb.astype(np.float32)

    sample = {
        # RGB frames - 两种 preceding 模式
        "P9_rgb": preceding_rgb_9,  # 9帧 preceding (clip-conditioned)
        "P1_rgb": preceding_rgb_1,  # 1帧 preceding (image-conditioned)
        "T_rgb": target_rgb,
        "R_rgb": reference_rgb,
        # Camera parameters
        "P9_poses_c2w": geometry.poses_c2w[indices.preceding_indices],
        "P1_poses_c2w": geometry.poses_c2w[[p1_idx]],
        "T_poses_c2w": geometry.poses_c2w[indices.target_indices],
        "P9_intrinsics": geometry.intrinsics[indices.preceding_indices],
        "P1_intrinsics": geometry.intrinsics[[p1_idx]],
        "T_intrinsics": geometry.intrinsics[indices.target_indices],
        # Scene data
        "scene_xyz": scene_xyz.astype(np.float32),
        "proj_P9": preceding_proj_9,  # 9帧 preceding scene projection (若add_fg_to_projection则含前景)
        "proj_P1": preceding_proj_1,  # 1帧 preceding scene projection
        "proj_T": target_proj,
        "proj_P9_bg": proj_P9_bg,  # background-only projections
        "proj_P1_bg": proj_P1_bg,
        "proj_T_bg": proj_T_bg,
        "proj_T_bg_orig": proj_T_bg_orig,  # non-augmented target scene projection
        "proj_P9_fg": proj_P9_fg,  # foreground-only projections (augmented)
        "proj_P1_fg": proj_P1_fg,
        "proj_T_fg": proj_T_fg,
        # Metadata
        "meta": {
            "t0": indices.t0,
            "P9_idx": indices.preceding_indices,  # 9帧 preceding 索引
            "P1_idx": [p1_idx],  # 1帧 preceding 索引 (= P9最后一帧)
            "T_idx": indices.target_indices,
            "C_idx": indices.candidate_indices,
            "scene_idx": scene_idx,
            "R_idx": reference_indices,
            "R_iou": reference_ious,
            "R_stats": ref_result.stats,  # Contains best_iou, avg_iou, threshold, no_ref_reason (if applicable)
            "projection_channels": list(config.projection_channels),
            "output_size": (out_h, out_w),
            "add_fg_to_projection": config.add_fg_to_projection,
            "has_fg": has_fg,
            "src_aug_applied": src_aug_applied,
        },
    }
    if reference_scene is not None:
        sample["R_scene_rgb"] = reference_scene
    if reference_scene_orig is not None:
        sample["R_scene_rgb_orig"] = reference_scene_orig
    if scene_rgb is not None:
        sample["scene_rgb"] = scene_rgb.astype(np.uint8)
    # Save both original and augmented reference frame for debug inspection
    if src_aug_applied and getattr(config, "src_aug_save_debug", False):
        sample["scene_rgb_original"] = scene_rgb_original.astype(np.uint8)
        sample["scene_rgb_augmented"] = scene_rgb_augmented_img.astype(np.uint8)
    return sample


# Backward compatibility alias
build_training_episode = build_training_sample
