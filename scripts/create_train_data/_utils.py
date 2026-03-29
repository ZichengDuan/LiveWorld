from __future__ import annotations
"""Data processing internal utilities."""

# === types ===

"""
LiveWorld 数据类型定义
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SampleIndices:
    """
    训练样本的帧索引

    t0: target帧的起始索引
    preceding_indices: 前置帧索引列表 (P)
    target_indices: 目标帧索引列表 (T)
    candidate_indices: 候选帧索引列表 (C)，用于选择参考帧
    """
    t0: int
    preceding_indices: list[int]
    target_indices: list[int]
    candidate_indices: list[int]


EpisodeIndices = SampleIndices  # 兼容别名


@dataclass
class VideoGeometry:
    """
    视频几何信息

    由MapAnything估计得到，包含:
    - frames: RGB帧 (L, H, W, 3)
    - depths: 深度图 (L, H, W)
    - intrinsics: 相机内参 (L, 3, 3)
    - poses_c2w: 相机位姿，camera-to-world (L, 4, 4)
    - masks: 有效区域mask (L, H, W)
    """
    frames: np.ndarray
    depths: np.ndarray
    intrinsics: np.ndarray
    poses_c2w: np.ndarray
    masks: Optional[np.ndarray] = None
    frame_indices: Optional[np.ndarray] = None
    original_size: Optional[tuple[int, int]] = None  # 原始分辨率 (H, W)
    processed_size: Optional[tuple[int, int]] = None  # MapAnything处理后的分辨率

# === distributed ===


import os
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


def get_rank_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def shard_items(items: Sequence[T], rank: int, world_size: int, mode: str = "interleave") -> list[T]:
    """
    Distribute items across ranks.

    Args:
        items: Sequence of items to distribute
        rank: Current rank (0-indexed)
        world_size: Total number of ranks
        mode: Distribution mode
            - "interleave": Round-robin (0,4,8,... for rank 0 with 4 GPUs)
            - "contiguous": Contiguous blocks (0-499 for rank 0 with 2000 items, 4 GPUs)

    Returns:
        List of items assigned to this rank
    """
    if world_size <= 1:
        return list(items)

    if mode == "contiguous":
        # Contiguous block assignment
        n = len(items)
        base_size = n // world_size
        remainder = n % world_size
        # Ranks 0..remainder-1 get one extra item
        if rank < remainder:
            start = rank * (base_size + 1)
            end = start + base_size + 1
        else:
            start = remainder * (base_size + 1) + (rank - remainder) * base_size
            end = start + base_size
        return list(items[start:end])
    else:
        # Interleave (round-robin) - default
        return [item for idx, item in enumerate(items) if idx % world_size == rank]

# === naming ===


from dataclasses import dataclass


@dataclass(frozen=True)
class SampleNaming:
    """Naming scheme for training sample output files."""
    # RGB videos - 两种 preceding 模式
    preceding_rgb_9: str  # 9帧 preceding (clip-conditioned)
    preceding_rgb_1: str  # 1帧 preceding (image-conditioned)
    target_rgb: str
    reference_rgb: str
    reference_scene_rgb: str
    reference_scene_rgb_orig: str  # non-augmented reference scene
    # Scene-masked RGB videos - 两种 preceding 模式
    preceding_scene_rgb_9: str
    preceding_scene_rgb_1: str
    target_scene_rgb: str
    # 前景RGB videos (抠出的动态物体)
    preceding_fg_rgb_9: str
    preceding_fg_rgb_1: str
    target_fg_rgb: str
    # Augmented前景RGB videos (scatter黑点 + random masking)
    preceding_fg_aug_rgb_9: str
    preceding_fg_aug_rgb_1: str
    target_fg_aug_rgb: str
    # Camera parameters (txt files)
    preceding_poses_c2w: str
    target_poses_c2w: str
    preceding_intrinsics: str
    target_intrinsics: str
    # Scene projection videos - 两种 preceding 模式
    preceding_scene_proj_rgb_9: str  # 9帧 preceding scene projection
    preceding_scene_proj_rgb_1: str  # 1帧 preceding scene projection
    target_scene_proj_rgb: str
    target_scene_proj_rgb_orig: str  # non-augmented target scene projection
    # Foreground-only projection (augmented foreground on black)
    preceding_proj_fg_rgb_9: str
    preceding_proj_fg_rgb_1: str
    target_proj_fg_rgb: str
    # Scene+fg overlay projection (fg pixels overlaid on scene in pixel space)
    preceding_scene_proj_fg_overlay_rgb_9: str
    preceding_scene_proj_fg_overlay_rgb_1: str
    target_scene_proj_fg_overlay_rgb: str
    # Scene point cloud
    scene_xyz: str
    scene_rgb: str


# Backward compatibility alias
EpisodeNaming = SampleNaming


_NAMING_SCHEMES = {
    "legacy": SampleNaming(
        preceding_rgb_9="P9_rgb",
        preceding_rgb_1="P1_rgb",
        target_rgb="T_rgb",
        reference_rgb="R_rgb",
        reference_scene_rgb="R_scene_rgb",
        reference_scene_rgb_orig="R_scene_rgb_orig",
        preceding_scene_rgb_9="P9_scene_rgb",
        preceding_scene_rgb_1="P1_scene_rgb",
        target_scene_rgb="T_scene_rgb",
        preceding_fg_rgb_9="P9_fg_rgb",
        preceding_fg_rgb_1="P1_fg_rgb",
        target_fg_rgb="T_fg_rgb",
        preceding_fg_aug_rgb_9="P9_fg_aug_rgb",
        preceding_fg_aug_rgb_1="P1_fg_aug_rgb",
        target_fg_aug_rgb="T_fg_aug_rgb",
        preceding_poses_c2w="P_poses_c2w",
        target_poses_c2w="T_poses_c2w",
        preceding_intrinsics="P_intrinsics",
        target_intrinsics="T_intrinsics",
        preceding_scene_proj_rgb_9="proj_P9_rgb",
        preceding_scene_proj_rgb_1="proj_P1_rgb",
        target_scene_proj_rgb="proj_T_rgb",
        target_scene_proj_rgb_orig="proj_T_rgb_orig",
        preceding_proj_fg_rgb_9="proj_P9_fg_rgb",
        preceding_proj_fg_rgb_1="proj_P1_fg_rgb",
        target_proj_fg_rgb="proj_T_fg_rgb",
        preceding_scene_proj_fg_overlay_rgb_9="proj_P9_fg_overlay_rgb",
        preceding_scene_proj_fg_overlay_rgb_1="proj_P1_fg_overlay_rgb",
        target_scene_proj_fg_overlay_rgb="proj_T_fg_overlay_rgb",
        scene_xyz="scene_xyz",
        scene_rgb="scene_rgb",
    ),
    "paper": SampleNaming(
        preceding_rgb_9="P9_rgb",
        preceding_rgb_1="P1_rgb",
        target_rgb="T_rgb",
        reference_rgb="R_rgb",
        reference_scene_rgb="R_scene_rgb",
        reference_scene_rgb_orig="R_scene_rgb_orig",
        preceding_scene_rgb_9="P9_scene_rgb",
        preceding_scene_rgb_1="P1_scene_rgb",
        target_scene_rgb="T_scene_rgb",
        preceding_fg_rgb_9="P9_fg_rgb",
        preceding_fg_rgb_1="P1_fg_rgb",
        target_fg_rgb="T_fg_rgb",
        preceding_fg_aug_rgb_9="P9_fg_aug_rgb",
        preceding_fg_aug_rgb_1="P1_fg_aug_rgb",
        target_fg_aug_rgb="T_fg_aug_rgb",
        preceding_poses_c2w="P_poses_c2w",
        target_poses_c2w="T_poses_c2w",
        preceding_intrinsics="P_intrinsics",
        target_intrinsics="T_intrinsics",
        preceding_scene_proj_rgb_9="SP9_rgb",
        preceding_scene_proj_rgb_1="SP1_rgb",
        target_scene_proj_rgb="ST_rgb",
        target_scene_proj_rgb_orig="ST_rgb_orig",
        preceding_proj_fg_rgb_9="SP9_fg_rgb",
        preceding_proj_fg_rgb_1="SP1_fg_rgb",
        target_proj_fg_rgb="ST_fg_rgb",
        preceding_scene_proj_fg_overlay_rgb_9="SP9_fg_overlay_rgb",
        preceding_scene_proj_fg_overlay_rgb_1="SP1_fg_overlay_rgb",
        target_scene_proj_fg_overlay_rgb="ST_fg_overlay_rgb",
        scene_xyz="S_xyz",
        scene_rgb="S_rgb",
    ),
    "figure": SampleNaming(
        preceding_rgb_9="preceding_rgb_9",
        preceding_rgb_1="preceding_rgb_1",
        target_rgb="target_rgb",
        reference_rgb="reference_rgb",
        reference_scene_rgb="reference_scene_rgb",
        reference_scene_rgb_orig="reference_scene_rgb_orig",
        preceding_scene_rgb_9="preceding_scene_rgb_9",
        preceding_scene_rgb_1="preceding_scene_rgb_1",
        target_scene_rgb="target_scene_rgb",
        preceding_fg_rgb_9="preceding_fg_rgb_9",
        preceding_fg_rgb_1="preceding_fg_rgb_1",
        target_fg_rgb="target_fg_rgb",
        preceding_fg_aug_rgb_9="preceding_fg_aug_rgb_9",
        preceding_fg_aug_rgb_1="preceding_fg_aug_rgb_1",
        target_fg_aug_rgb="target_fg_aug_rgb",
        preceding_poses_c2w="preceding_poses_c2w",
        target_poses_c2w="target_poses_c2w",
        preceding_intrinsics="preceding_intrinsics",
        target_intrinsics="target_intrinsics",
        preceding_scene_proj_rgb_9="preceding_scene_proj_rgb_9",
        preceding_scene_proj_rgb_1="preceding_scene_proj_rgb_1",
        target_scene_proj_rgb="target_scene_proj_rgb",
        target_scene_proj_rgb_orig="target_scene_proj_rgb_orig",
        preceding_proj_fg_rgb_9="preceding_proj_fg_rgb_9",
        preceding_proj_fg_rgb_1="preceding_proj_fg_rgb_1",
        target_proj_fg_rgb="target_proj_fg_rgb",
        preceding_scene_proj_fg_overlay_rgb_9="preceding_scene_proj_fg_overlay_rgb_9",
        preceding_scene_proj_fg_overlay_rgb_1="preceding_scene_proj_fg_overlay_rgb_1",
        target_scene_proj_fg_overlay_rgb="target_scene_proj_fg_overlay_rgb",
        scene_xyz="scene_xyz",
        scene_rgb="scene_rgb",
    ),
}


def get_sample_naming(style: str) -> SampleNaming:
    """Get naming scheme for training sample output files."""
    if style not in _NAMING_SCHEMES:
        raise ValueError(f"Unknown naming style: {style}")
    return _NAMING_SCHEMES[style]


# Backward compatibility alias
get_episode_naming = get_sample_naming

# === viz ===


from typing import Iterable, Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.create_train_data._geometry import transform_points, unproject_depth_to_points, voxel_indices
from scripts.create_train_data._utils import VideoGeometry


def save_point_cloud_ply(
    points: np.ndarray,
    out_path: str,
    colors: Optional[np.ndarray] = None,
    trajectory_points: Optional[np.ndarray] = None,
    trajectory_color: tuple[int, int, int] = (255, 0, 0),
) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    if colors is not None:
        if colors.shape[0] != points.shape[0] or colors.shape[1] != 3:
            raise ValueError("colors must be Nx3 and match points")
    if trajectory_points is not None:
        if trajectory_points.ndim != 2 or trajectory_points.shape[1] != 3:
            raise ValueError("trajectory_points must be Nx3")

    has_colors = colors is not None
    if has_colors and trajectory_points is not None:
        traj_colors = np.tile(np.array(trajectory_color, dtype=np.uint8), (trajectory_points.shape[0], 1))
        colors_all = np.concatenate([colors.astype(np.uint8), traj_colors], axis=0)
    elif has_colors:
        colors_all = colors.astype(np.uint8)
    else:
        colors_all = None

    vertex_count = points.shape[0] + (trajectory_points.shape[0] if trajectory_points is not None else 0)
    edge_count = 0
    if trajectory_points is not None and trajectory_points.shape[0] > 1:
        edge_count = trajectory_points.shape[0] - 1

    with open(out_path, "w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertex_count}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors_all is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        if edge_count > 0:
            f.write(f"element edge {edge_count}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
        f.write("end_header\n")

        if colors_all is None:
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            if trajectory_points is not None:
                for p in trajectory_points:
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            colors_u8 = np.clip(colors_all, 0, 255).astype(np.uint8)
            for p, c in zip(points, colors_u8[: points.shape[0]]):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            if trajectory_points is not None:
                offset = points.shape[0]
                for p, c in zip(trajectory_points, colors_u8[offset:]):
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

        if edge_count > 0:
            base = points.shape[0]
            for i in range(edge_count):
                v1 = base + i
                v2 = base + i + 1
                f.write(f"{v1} {v2}\n")


def sample_points(
    points: np.ndarray,
    max_points: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def set_axes_equal(ax, points: np.ndarray) -> None:
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = maxs - mins
    max_span = float(np.max(spans))
    if max_span <= 0:
        max_span = 1.0
    centers = (mins + maxs) / 2.0
    half = max_span / 2.0
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)


def collect_scene_points(
    geometry: VideoGeometry,
    indices: Iterable[int],
    voxel_size: float,
    dynamic_masks: Optional[np.ndarray] = None,
) -> np.ndarray:
    points_all = []
    for idx in indices:
        depth = geometry.depths[idx]
        K = geometry.intrinsics[idx]
        c2w = geometry.poses_c2w[idx]
        mask = depth > 0
        if geometry.masks is not None:
            mask = mask & geometry.masks[idx]
        if dynamic_masks is not None:
            mask = mask & (~dynamic_masks[idx])

        points_cam = unproject_depth_to_points(depth, K, mask=mask)
        if points_cam.size == 0:
            continue
        points_world = transform_points(points_cam, c2w)
        points_all.append(points_world)

    if not points_all:
        return np.zeros((0, 3), dtype=np.float32)

    points = np.concatenate(points_all, axis=0)
    if voxel_size > 0:
        vox = voxel_indices(points, voxel_size)
        _, unique_idx = np.unique(vox, axis=0, return_index=True)
        points = points[unique_idx]
    return points.astype(np.float32)


def plot_point_cloud(
    points: np.ndarray,
    out_path: str,
    max_points: int = 200_000,
    seed: Optional[int] = None,
    elev: float = 20.0,
    azim: float = -60.0,
    title: Optional[str] = None,
) -> None:
    if points.size == 0:
        raise ValueError("Point cloud is empty.")

    pts = sample_points(points, max_points=max_points, seed=seed)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        s=0.5,
        c=pts[:, 2],
        cmap="viridis",
        alpha=0.8,
        linewidths=0,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        ax.set_title(title)
    set_axes_equal(ax, pts)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_camera_trajectory(
    poses_c2w: np.ndarray,
    out_path: str,
    stride: int = 1,
    axis_stride: int = 10,
    axis_scale: float = 0.1,
    show_axes: bool = True,
    elev: float = 20.0,
    azim: float = -60.0,
    title: Optional[str] = None,
) -> None:
    if poses_c2w.ndim != 3 or poses_c2w.shape[1:] != (4, 4):
        raise ValueError("poses_c2w must be (N,4,4)")

    indices = np.arange(0, poses_c2w.shape[0], stride)
    poses = poses_c2w[indices]
    centers = poses[:, :3, 3]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color="tab:red", linewidth=1.0)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color="tab:red", s=4)

    if show_axes:
        axis_indices = indices[::axis_stride] if axis_stride > 0 else indices
        axis_poses = poses_c2w[axis_indices]
        for pose in axis_poses:
            origin = pose[:3, 3]
            rot = pose[:3, :3]
            dirs = [rot[:, 0], rot[:, 1], rot[:, 2]]
            colors = ["r", "g", "b"]
            for d, c in zip(dirs, colors):
                ax.quiver(
                    origin[0],
                    origin[1],
                    origin[2],
                    d[0],
                    d[1],
                    d[2],
                    length=axis_scale,
                    normalize=True,
                    color=c,
                    linewidth=0.8,
                )

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        ax.set_title(title)
    set_axes_equal(ax, centers)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_point_cloud_with_trajectory(
    points: np.ndarray,
    poses_c2w: np.ndarray,
    out_path: str,
    max_points: int = 200_000,
    seed: Optional[int] = None,
    traj_stride: int = 1,
    axis_stride: int = 10,
    axis_scale: float = 0.1,
    show_axes: bool = True,
    elev: float = 20.0,
    azim: float = -60.0,
    title: Optional[str] = None,
) -> None:
    if points.size == 0:
        raise ValueError("Point cloud is empty.")
    if poses_c2w.ndim != 3 or poses_c2w.shape[1:] != (4, 4):
        raise ValueError("poses_c2w must be (N,4,4)")

    pts = sample_points(points, max_points=max_points, seed=seed)

    indices = np.arange(0, poses_c2w.shape[0], traj_stride)
    poses = poses_c2w[indices]
    centers = poses[:, :3, 3]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        s=0.5,
        c=pts[:, 2],
        cmap="viridis",
        alpha=0.6,
        linewidths=0,
    )
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color="tab:red", linewidth=1.0)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color="tab:red", s=4)

    if show_axes:
        axis_indices = indices[::axis_stride] if axis_stride > 0 else indices
        axis_poses = poses_c2w[axis_indices]
        for pose in axis_poses:
            origin = pose[:3, 3]
            rot = pose[:3, :3]
            dirs = [rot[:, 0], rot[:, 1], rot[:, 2]]
            colors = ["r", "g", "b"]
            for d, c in zip(dirs, colors):
                ax.quiver(
                    origin[0],
                    origin[1],
                    origin[2],
                    d[0],
                    d[1],
                    d[2],
                    length=axis_scale,
                    normalize=True,
                    color=c,
                    linewidth=0.8,
                )

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        ax.set_title(title)
    set_axes_equal(ax, np.vstack([pts, centers]))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
