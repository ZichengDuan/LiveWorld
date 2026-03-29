"""
LiveWorld 通用工具函数

整合了 LiveWorld 数据处理、推理、训练中常用的工具函数:
- 图像处理: resize, crop, normalize
- 相机/内参: scale_intrinsics, intrinsics 操作
- 轨迹生成: TrajectoryGenerator
- 几何工具: 点云构建、投影、坐标变换
- 视频 I/O: 加载、保存视频
- 深度可视化
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Tuple, Iterable, Set, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R

from .utils import save_video_h264


# ============================================================================
# 图像处理工具
# ============================================================================

def resize_short_edge(image: np.ndarray, short_edge: int) -> np.ndarray:
    """
    将图片短边 resize 到指定长度，保持宽高比

    Args:
        image: 输入图片 (H, W, C) 或 (H, W)
        short_edge: 目标短边长度

    Returns:
        resize 后的图片
    """
    h, w = image.shape[:2]

    if h < w:
        new_h = short_edge
        new_w = int(w * short_edge / h)
    else:
        new_w = short_edge
        new_h = int(h * short_edge / w)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def center_crop(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    对图片进行中心裁剪

    Args:
        image: 输入图片 (H, W, C) 或 (H, W)
        target_h: 目标高度
        target_w: 目标宽度

    Returns:
        裁剪后的图片
    """
    h, w = image.shape[:2]

    if h < target_h or w < target_w:
        raise ValueError(f"Image size ({h}, {w}) is smaller than target ({target_h}, {target_w})")

    start_y = (h - target_h) // 2
    start_x = (w - target_w) // 2

    return image[start_y:start_y + target_h, start_x:start_x + target_w]


def resize_short_edge_and_center_crop(
    image: np.ndarray,
    target_h: int,
    target_w: int
) -> np.ndarray:
    """
    将图片短边 resize 到目标短边长度，然后 center crop 到目标尺寸

    Args:
        image: 输入图片 (H, W, C)
        target_h: 目标高度
        target_w: 目标宽度

    Returns:
        处理后的图片 (target_h, target_w, C)
    """
    target_short = min(target_h, target_w)
    resized = resize_short_edge(image, target_short)
    return center_crop(resized, target_h, target_w)


def resize_image(
    image: np.ndarray,
    size: Union[int, Tuple[int, int]],
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize 图片到指定尺寸

    Args:
        image: 输入图片 (H, W, C) 或 (H, W)
        size: 目标尺寸，int 表示 (size, size)，tuple 表示 (H, W)
        interpolation: 插值方法

    Returns:
        resize 后的图片
    """
    if isinstance(size, int):
        target_h, target_w = size, size
    else:
        target_h, target_w = size

    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def load_image(path: Union[str, Path], rgb: bool = True) -> np.ndarray:
    """
    加载图片

    Args:
        path: 图片路径
        rgb: 是否转换为 RGB 格式

    Returns:
        图片数组 (H, W, C)
    """
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Failed to load image: {path}")
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(image: np.ndarray, path: Union[str, Path], rgb: bool = True) -> None:
    """
    保存图片

    Args:
        image: 图片数组 (H, W, C)
        path: 保存路径
        rgb: 输入是否为 RGB 格式
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if rgb and image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image)


# ============================================================================
# 相机/内参工具
# ============================================================================

def scale_intrinsics(
    K: np.ndarray,
    from_size: Tuple[int, int],
    to_size: Tuple[int, int]
) -> np.ndarray:
    """
    将内参从一个尺寸缩放到另一个尺寸

    Args:
        K: 内参矩阵 (3, 3)
        from_size: 原始尺寸 (H, W)
        to_size: 目标尺寸 (H, W)

    Returns:
        缩放后的内参矩阵 (3, 3)
    """
    from_h, from_w = from_size
    to_h, to_w = to_size

    scale_x = to_w / from_w
    scale_y = to_h / from_h

    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy

    return K_scaled


def scale_intrinsics_batch(
    intrinsics: np.ndarray,
    from_size: Tuple[int, int],
    to_size: Tuple[int, int]
) -> np.ndarray:
    """
    批量缩放内参

    Args:
        intrinsics: 内参矩阵 (N, 3, 3) 或 (3, 3)
        from_size: 原始尺寸 (H, W)
        to_size: 目标尺寸 (H, W)

    Returns:
        缩放后的内参矩阵
    """
    if intrinsics.ndim == 2:
        return scale_intrinsics(intrinsics, from_size, to_size)

    return np.stack([
        scale_intrinsics(K, from_size, to_size)
        for K in intrinsics
    ], axis=0)


def make_intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    构建内参矩阵

    Args:
        fx, fy: 焦距
        cx, cy: 主点

    Returns:
        内参矩阵 (3, 3)
    """
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)


def intrinsics_to_fov(K: np.ndarray, size: Tuple[int, int]) -> Tuple[float, float]:
    """
    从内参计算视场角 (FOV)

    Args:
        K: 内参矩阵 (3, 3)
        size: 图像尺寸 (H, W)

    Returns:
        (fov_x, fov_y) 以角度为单位
    """
    h, w = size
    fx, fy = K[0, 0], K[1, 1]
    fov_x = 2 * np.arctan(w / (2 * fx)) * 180 / np.pi
    fov_y = 2 * np.arctan(h / (2 * fy)) * 180 / np.pi
    return fov_x, fov_y


# ============================================================================
# 相机轨迹生成
# ============================================================================

class TrajectoryGenerator:
    """
    相机轨迹生成器工具类

    输入：初始位姿 (4x4 c2w matrix), 总帧数
    输出：轨迹列表 (N, 4, 4)

    支持的轨迹模式:
    - yaw_sweep_pause_return: 向右转 -> 停顿 -> 转回
    - yaw_sweep: 左右摇摆
    - pitch_sweep: 上下摇摆
    - orbit: 绕点做圆周运动
    - dolly_forward: 向前推进
    - dolly_zoom: 推拉变焦
    - static: 静止不动
    """

    @staticmethod
    def generate(
        initial_c2w: np.ndarray,
        total_frames: int,
        mode: str = 'yaw_sweep',
        **kwargs
    ) -> np.ndarray:
        """
        统一入口函数

        Args:
            initial_c2w: 初始位姿 (4, 4) c2w 矩阵
            total_frames: 总帧数
            mode: 轨迹模式名称
            **kwargs: 传递给具体路径函数的参数

        Returns:
            轨迹数组 (N, 4, 4)
        """
        method = getattr(TrajectoryGenerator, f"_path_{mode}", None)
        if not method:
            raise ValueError(f"Unknown trajectory mode: {mode}")

        return method(initial_c2w, total_frames, **kwargs)

    @staticmethod
    def _path_static(initial_c2w: np.ndarray, total_frames: int, **kwargs) -> np.ndarray:
        """静止不动"""
        return np.stack([initial_c2w.copy() for _ in range(total_frames)], axis=0)

    @staticmethod
    def _path_yaw_sweep_pause_return(
        initial_c2w: np.ndarray,
        total_frames: int,
        max_angle: float = -90,
        pause_ratio: float = 0.2
    ) -> np.ndarray:
        """
        向右转 -> 停顿 -> 转回原位

        Args:
            max_angle: 最大旋转角度，负数向右
            pause_ratio: 停顿时间占总时长的比例
        """
        poses = []

        t1 = int(total_frames * (1 - pause_ratio) / 2)
        t2 = int(total_frames - t1)

        key_frames = [0, t1, t2, total_frames - 1]
        key_angles = [0, max_angle, max_angle, 0]

        for i in range(total_frames):
            current_angle = np.interp(i, key_frames, key_angles)
            r_relative = R.from_euler('y', current_angle, degrees=True).as_matrix()
            new_c2w = initial_c2w.copy()
            new_c2w[:3, :3] = initial_c2w[:3, :3] @ r_relative
            new_c2w[:3, 3] = initial_c2w[:3, 3]
            poses.append(new_c2w)

        return np.array(poses)

    @staticmethod
    def _path_yaw_sweep(
        initial_c2w: np.ndarray,
        total_frames: int,
        max_angle: float = 30,
        num_cycles: float = 1.0
    ) -> np.ndarray:
        """
        左右摇摆轨迹

        Args:
            max_angle: 最大摇摆角度
            num_cycles: 摇摆周期数
        """
        poses = []
        for i in range(total_frames):
            t = i / (total_frames - 1) if total_frames > 1 else 0
            angle = max_angle * np.sin(2 * np.pi * num_cycles * t)
            r_relative = R.from_euler('y', angle, degrees=True).as_matrix()
            new_c2w = initial_c2w.copy()
            new_c2w[:3, :3] = initial_c2w[:3, :3] @ r_relative
            poses.append(new_c2w)
        return np.array(poses)

    @staticmethod
    def _path_pitch_sweep(
        initial_c2w: np.ndarray,
        total_frames: int,
        max_angle: float = 15,
        num_cycles: float = 1.0
    ) -> np.ndarray:
        """
        上下摇摆轨迹

        Args:
            max_angle: 最大摇摆角度
            num_cycles: 摇摆周期数
        """
        poses = []
        for i in range(total_frames):
            t = i / (total_frames - 1) if total_frames > 1 else 0
            angle = max_angle * np.sin(2 * np.pi * num_cycles * t)
            r_relative = R.from_euler('x', angle, degrees=True).as_matrix()
            new_c2w = initial_c2w.copy()
            new_c2w[:3, :3] = initial_c2w[:3, :3] @ r_relative
            poses.append(new_c2w)
        return np.array(poses)

    @staticmethod
    def _path_orbit(
        initial_c2w: np.ndarray,
        total_frames: int,
        radius: float = 0.5,
        height_var: float = 0.1,
        num_cycles: float = 1.0
    ) -> np.ndarray:
        """
        绕初始位置做圆周运动

        Args:
            radius: 圆周半径
            height_var: 高度变化幅度
            num_cycles: 圆周数
        """
        poses = []
        center = initial_c2w[:3, 3].copy()

        for i in range(total_frames):
            t = i / total_frames
            angle = 2 * np.pi * num_cycles * t

            new_c2w = initial_c2w.copy()
            offset_x = radius * np.sin(angle)
            offset_z = radius * (1 - np.cos(angle))
            offset_y = height_var * np.sin(2 * angle)

            new_c2w[:3, 3] = center + np.array([offset_x, offset_y, offset_z])

            yaw = -np.degrees(angle)
            r_relative = R.from_euler('y', yaw, degrees=True).as_matrix()
            new_c2w[:3, :3] = initial_c2w[:3, :3] @ r_relative

            poses.append(new_c2w)
        return np.array(poses)

    @staticmethod
    def _path_dolly_forward(
        initial_c2w: np.ndarray,
        total_frames: int,
        distance: float = 1.0
    ) -> np.ndarray:
        """
        向前推进的轨迹

        Args:
            distance: 推进距离
        """
        poses = []
        for i in range(total_frames):
            t = i / (total_frames - 1) if total_frames > 1 else 0
            new_c2w = initial_c2w.copy()
            forward = initial_c2w[:3, 2]
            new_c2w[:3, 3] = initial_c2w[:3, 3] - forward * distance * t
            poses.append(new_c2w)
        return np.array(poses)

    @staticmethod
    def _path_dolly_zoom(
        initial_c2w: np.ndarray,
        total_frames: int,
        distance: float = 1.0,
        return_back: bool = True
    ) -> np.ndarray:
        """
        推拉变焦轨迹

        Args:
            distance: 推进距离
            return_back: 是否返回
        """
        poses = []
        for i in range(total_frames):
            if return_back:
                t = i / (total_frames - 1) if total_frames > 1 else 0
                # 前半段前进，后半段后退
                if t < 0.5:
                    progress = t * 2
                else:
                    progress = (1 - t) * 2
            else:
                progress = i / (total_frames - 1) if total_frames > 1 else 0

            new_c2w = initial_c2w.copy()
            forward = initial_c2w[:3, 2]
            new_c2w[:3, 3] = initial_c2w[:3, 3] - forward * distance * progress
            poses.append(new_c2w)
        return np.array(poses)


# ============================================================================
# 几何工具 - 点云和投影
# ============================================================================

def unproject_depth_to_points(
    depth: np.ndarray,
    K: np.ndarray,
    mask: Optional[np.ndarray] = None,
    return_pixels: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    深度图反投影为相机坐标系3D点

    Args:
        depth: 深度图 (H, W)
        K: 内参矩阵 (3, 3)
        mask: 有效像素 mask (H, W)
        return_pixels: 是否返回像素坐标

    Returns:
        points: 3D点 (N, 3)
        pixels: (可选) 像素坐标 (N, 2)
    """
    if depth.ndim != 2:
        raise ValueError("depth must be HxW")
    if K.shape != (3, 3):
        raise ValueError("K must be 3x3")

    if mask is None:
        mask = depth > 0
    else:
        mask = mask & (depth > 0)

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        if return_pixels:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)
        return np.zeros((0, 3), dtype=np.float32)

    z = depth[ys, xs].astype(np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    points = np.stack([x, y, z], axis=1)

    if return_pixels:
        pixels = np.stack([xs.astype(np.int32), ys.astype(np.int32)], axis=1)
        return points, pixels
    return points


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    使用4x4变换矩阵变换3D点

    Args:
        points: 3D点 (N, 3)
        transform: 变换矩阵 (4, 4)

    Returns:
        变换后的点 (N, 3)
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    if transform.shape != (4, 4):
        raise ValueError("transform must be 4x4")

    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([points.astype(np.float32), ones], axis=1)
    transformed = (transform @ homo.T).T[:, :3]
    return transformed.astype(np.float32)


def project_points(points: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将相机坐标系3D点投影到像素坐标

    Args:
        points: 3D点 (N, 3)
        K: 内参矩阵 (3, 3)

    Returns:
        uv: 像素坐标 (N, 2)
        z: 深度 (N,)
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    if K.shape != (3, 3):
        raise ValueError("K must be 3x3")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.where(z > 0, (x / z) * fx + cx, np.nan)
        v = np.where(z > 0, (y / z) * fy + cy, np.nan)

    return np.stack([u, v], axis=1), z


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    体素下采样

    Args:
        points: 3D点 (N, 3)
        voxel_size: 体素大小

    Returns:
        下采样后的点 (M, 3)
    """
    if voxel_size <= 0:
        return points.astype(np.float32)
    if points.size == 0:
        return points.astype(np.float32)

    vox = np.floor(points / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(vox, axis=0, return_index=True)
    return points[unique_idx].astype(np.float32)


def voxel_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    计算每个点所属的 voxel 索引

    Args:
        points: 3D点 (N, 3)
        voxel_size: 体素大小

    Returns:
        体素索引 (N, 3)
    """
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    return np.floor(points / voxel_size).astype(np.int32)


# ============================================================================
# 视频 I/O
# ============================================================================

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def load_video_frames(
    video_path: Union[str, Path],
    start: int = 0,
    stride: int = 1,
    max_frames: Optional[int] = None,
    target_size: Optional[Union[int, Tuple[int, int]]] = None,
) -> List[np.ndarray]:
    """
    加载视频帧

    Args:
        video_path: 视频路径
        start: 起始帧索引
        stride: 帧间隔
        max_frames: 最大帧数
        target_size: 目标尺寸，int 表示 (size, size)，tuple 表示 (W, H)

    Returns:
        帧列表 [np.ndarray]
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    resize_wh = None
    if target_size is not None:
        if isinstance(target_size, int):
            resize_wh = (target_size, target_size)
        else:
            resize_wh = (target_size[0], target_size[1])

    frames: List[np.ndarray] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx >= start and (idx - start) % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize_wh is not None:
                frame = cv2.resize(frame, resize_wh, interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
        idx += 1

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from video: {video_path}")
    return frames


def save_video(
    frames: np.ndarray,
    out_path: Union[str, Path],
    fps: float = 16.0
) -> None:
    """
    保存帧序列为 mp4 视频

    Args:
        frames: 帧数组 (N, H, W, C)
        out_path: 输出路径
        fps: 帧率
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_video_h264(out_path, frames, fps=fps)


def get_video_info(video_path: Union[str, Path]) -> dict:
    """
    获取视频信息

    Args:
        video_path: 视频路径

    Returns:
        包含 fps, frame_count, width, height 的字典
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()

    if info['fps'] is None or info['fps'] <= 0:
        info['fps'] = 16.0

    return info


def list_video_files(
    video_dir: Union[str, Path],
    recursive: bool = True
) -> List[Path]:
    """
    列出目录中的视频文件

    Args:
        video_dir: 目录路径
        recursive: 是否递归搜索

    Returns:
        视频文件路径列表
    """
    root = Path(video_dir)
    if not root.is_dir():
        raise NotADirectoryError(root)

    if recursive:
        videos = []
        for ext in VIDEO_EXTS:
            videos.extend(root.rglob(f"*{ext}"))
            videos.extend(root.rglob(f"*{ext.upper()}"))
        return sorted(set(videos))
    else:
        return sorted([p for p in root.iterdir() if p.suffix.lower() in VIDEO_EXTS])


# ============================================================================
# 深度可视化
# ============================================================================

def visualize_depth(
    depth: np.ndarray,
    colormap: int = cv2.COLORMAP_VIRIDIS,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None
) -> np.ndarray:
    """
    深度图可视化

    Args:
        depth: 深度图 (H, W)
        colormap: OpenCV colormap
        min_depth: 最小深度值 (用于归一化)
        max_depth: 最大深度值 (用于归一化)

    Returns:
        彩色深度图 (H, W, 3) RGB
    """
    if min_depth is None:
        min_depth = depth.min()
    if max_depth is None:
        max_depth = depth.max()

    d_norm = (depth - min_depth) / (max_depth - min_depth + 1e-6)
    d_norm = np.clip(d_norm, 0, 1)
    d_vis = cv2.applyColorMap((d_norm * 255).astype(np.uint8), colormap)
    d_vis = cv2.cvtColor(d_vis, cv2.COLOR_BGR2RGB)
    return d_vis


def save_depth_visualization(
    depth: np.ndarray,
    out_path: Union[str, Path],
    colormap: int = cv2.COLORMAP_VIRIDIS
) -> None:
    """
    保存深度可视化图

    Args:
        depth: 深度图 (H, W)
        out_path: 输出路径
        colormap: OpenCV colormap
    """
    d_vis = visualize_depth(depth, colormap)
    save_image(d_vis, out_path)


# ============================================================================
# 点云 I/O
# ============================================================================

def save_point_cloud_ply(
    save_path: Union[str, Path],
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    """
    保存点云为 PLY 文件

    Args:
        save_path: 保存路径
        points: 点坐标 (N, 3)
        colors: 点颜色 (N, 3) uint8 RGB
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure colors are uint8
    if colors.dtype != np.uint8:
        colors = (colors * 255).clip(0, 255).astype(np.uint8)

    num_points = len(points)

    with open(save_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(num_points):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                    f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")

    print(f"Saved point cloud: {save_path} ({num_points} points)")


# ============================================================================
# 数据类型定义
# ============================================================================

@dataclass
class VideoGeometry:
    """
    视频几何信息

    由 MapAnything 估计得到，包含:
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
    original_size: Optional[Tuple[int, int]] = None
    processed_size: Optional[Tuple[int, int]] = None


# ============================================================================
# 便捷函数
# ============================================================================

def load_geometry_npz(path: Union[str, Path]) -> dict:
    """
    加载 geometry.npz 文件

    Args:
        path: npz 文件路径

    Returns:
        包含几何信息的字典
    """
    data = np.load(str(path))
    return {key: data[key] for key in data.files}


def save_geometry_npz(
    path: Union[str, Path],
    depths: np.ndarray,
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    processed_size: Tuple[int, int],
    original_size: Optional[Tuple[int, int]] = None,
    **kwargs
) -> None:
    """
    保存 geometry.npz 文件

    Args:
        path: 输出路径
        depths: 深度图 (N, H, W)
        poses_c2w: 位姿 (N, 4, 4)
        intrinsics: 内参 (N, 3, 3) 或 (3, 3)
        processed_size: 处理后的尺寸 (H, W)
        original_size: 原始尺寸 (H, W)
        **kwargs: 其他要保存的数据
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'depths': depths,
        'poses_c2w': poses_c2w,
        'intrinsics': intrinsics,
        'processed_size': np.array(processed_size),
    }

    if original_size is not None:
        save_dict['original_size'] = np.array(original_size)

    save_dict.update(kwargs)
    np.savez(str(path), **save_dict)


def make_comparison_image(
    images: List[np.ndarray],
    axis: int = 1,
    gap: int = 0,
    gap_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    创建对比图

    Args:
        images: 图片列表
        axis: 拼接方向，0=垂直，1=水平
        gap: 图片间隔
        gap_color: 间隔颜色 (R, G, B)

    Returns:
        拼接后的图片
    """
    if not images:
        raise ValueError("images list is empty")

    if gap > 0:
        h, w = images[0].shape[:2]
        if axis == 1:
            gap_img = np.full((h, gap, 3), gap_color, dtype=np.uint8)
        else:
            gap_img = np.full((gap, w, 3), gap_color, dtype=np.uint8)

        result = []
        for i, img in enumerate(images):
            result.append(img)
            if i < len(images) - 1:
                result.append(gap_img)
        return np.concatenate(result, axis=axis)
    else:
        return np.concatenate(images, axis=axis)


# ============================================================================
# Pipeline 工具函数 (从 pipeline_unified_backbone.py 移入)
# ============================================================================

def voxel_downsample_with_colors(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    voxel_size: float = 0.02,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Downsample point cloud using voxel grid, preserving colors.

    Args:
        points: [N, 3] point coordinates
        colors: [N, 3] point colors (optional)
        voxel_size: Voxel grid size

    Returns:
        Tuple of (downsampled_points, downsampled_colors)
    """
    if len(points) == 0:
        return points, colors

    vox_indices = np.floor(points / voxel_size).astype(np.int64)
    unique_voxels = {}
    for i, idx in enumerate(vox_indices):
        key = tuple(idx)
        if key not in unique_voxels:
            unique_voxels[key] = i

    indices = list(unique_voxels.values())
    downsampled_points = points[indices]
    downsampled_colors = colors[indices] if colors is not None else None

    return downsampled_points, downsampled_colors




def project_points_to_image(
    points_world: np.ndarray,
    pose_c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_size: Tuple[int, int],
    return_depth: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Project 3D world points to 2D image coordinates.

    Args:
        points_world: [N, 3] points in world coordinates
        pose_c2w: [4, 4] camera-to-world pose matrix
        intrinsics: [3, 3] camera intrinsic matrix
        image_size: (height, width) of target image
        return_depth: Whether to return depth values

    Returns:
        Tuple of (pixel_coords [M, 2], valid_mask [N], depths [M] if return_depth)
    """
    H, W = image_size

    pose_w2c = np.linalg.inv(pose_c2w)
    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]

    points_cam = (R @ points_world.T).T + t

    valid_mask = points_cam[:, 2] > 0.01
    points_cam_valid = points_cam[valid_mask]

    if len(points_cam_valid) == 0:
        empty_coords = np.zeros((0, 2), dtype=np.float32)
        if return_depth:
            return empty_coords, valid_mask, np.zeros(0, dtype=np.float32)
        return empty_coords, valid_mask, None

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x = points_cam_valid[:, 0]
    y = points_cam_valid[:, 1]
    z = points_cam_valid[:, 2]

    u = fx * x / z + cx
    v = fy * y / z + cy

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    valid_indices = np.where(valid_mask)[0]
    valid_mask[valid_indices[~in_bounds]] = False

    pixel_coords = np.stack([u[in_bounds], v[in_bounds]], axis=1)

    if return_depth:
        return pixel_coords, valid_mask, z[in_bounds]
    return pixel_coords, valid_mask, None




def generate_blue_noise_tile(size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a blue noise dither tile using void-and-cluster method.

    Args:
        size: Tile size (size x size)
        rng: Random number generator

    Returns:
        Blue noise tile [size, size] with values in [0, 1]
    """
    tile = rng.random((size, size))

    sigma = 1.5
    k_size = int(sigma * 6) | 1

    for _ in range(size * size // 4):
        blurred = gaussian_filter(tile, sigma, mode='wrap')

        max_idx = np.unravel_index(np.argmax(blurred), tile.shape)
        min_idx = np.unravel_index(np.argmin(blurred), tile.shape)

        tile[max_idx], tile[min_idx] = tile[min_idx], tile[max_idx]

    return (tile - tile.min()) / (tile.max() - tile.min() + 1e-8)


# ============================================================================
# Pipeline utility functions (projection, point cloud, IoU, reference selection)
# ============================================================================

def compute_iteration_plan(
    num_frames: int,
    frames_per_iter: int = None,
) -> List[Tuple[int, int, int]]:
    """
    Compute the iteration plan for generating num_frames (T2V mode, no overlap).

    - num_frames = number of frames to GENERATE (excluding input first frame)
    - First iteration (P1): preceding = first frame, generates frames_per_iter frames
    - Subsequent iterations (P9): preceding = last 9 frames, generates frames_per_iter frames
    - Each iteration generates exactly frames_per_iter new frames, NO overlap
    - Final video = first_frame + num_frames generated = num_frames + 1 total
    - num_frames must be divisible by frames_per_iter

    Args:
        num_frames: Frames to generate (not counting input), must be M * frames_per_iter
        frames_per_iter: Model frames per iteration (must be 4N+1, default=num_frames)

    Returns:
        list: List of tuples (output_start, output_end, model_frames)

    Example (frames_per_iter=33, num_frames=66):
        [(0, 33, 33), (33, 66, 33)]  # Each iteration: 33 new frames, NO overlap
        Final video: first_frame + 66 generated = 67 frames
    """
    if frames_per_iter is None:
        frames_per_iter = num_frames
    if frames_per_iter < 1:
        raise ValueError("frames_per_iter must be at least 1")

    if (frames_per_iter - 1) % 4 != 0:
        raise ValueError(f"frames_per_iter must be 4N+1, got {frames_per_iter}")

    if num_frames % frames_per_iter != 0:
        raise ValueError(
            f"num_frames ({num_frames}) must be divisible by "
            f"frames_per_iter ({frames_per_iter})"
        )

    plan = []
    num_iterations = num_frames // frames_per_iter
    for i in range(num_iterations):
        output_start = i * frames_per_iter
        output_end = (i + 1) * frames_per_iter
        plan.append((output_start, output_end, frames_per_iter))

    return plan


# =============================================================================
# Camera Intrinsics Helpers
# =============================================================================

def _safe_frame_index(frame_idx: int, length: int) -> int:
    """Clamp frame index to a valid range for arrays with known length."""
    if length <= 0:
        raise ValueError("length must be positive for safe frame indexing")
    if frame_idx < 0:
        return 0
    if frame_idx >= length:
        return length - 1
    return frame_idx


def compute_depth_scale_factor(
    ref_depth: np.ndarray,
    new_depth: np.ndarray,
    min_valid_pixels: int = 1000,
) -> float:
    """
    Compute scale factor to align new_depth to ref_depth.

    Uses median ratio of overlapping valid pixels to compute robust scale factor.
    new_depth_aligned = new_depth * scale_factor

    Args:
        ref_depth: Reference depth map (H, W) - the "ground truth" scale
        new_depth: New depth map (H, W) - to be scaled
        min_valid_pixels: Minimum number of valid overlapping pixels required

    Returns:
        Scale factor. Returns 1.0 if alignment fails.
    """
    # Find overlapping valid pixels
    valid_mask = (ref_depth > 0) & (new_depth > 0) & np.isfinite(ref_depth) & np.isfinite(new_depth)

    num_valid = valid_mask.sum()
    if num_valid < min_valid_pixels:
        print(f"  [ScaleAlign] Warning: Only {num_valid} valid pixels, need {min_valid_pixels}. Using scale=1.0")
        return 1.0

    ref_vals = ref_depth[valid_mask]
    new_vals = new_depth[valid_mask]

    # Compute median ratio (robust to outliers)
    ratios = ref_vals / (new_vals + 1e-8)

    # Filter extreme ratios (likely from noise/errors)
    ratio_median = np.median(ratios)
    ratio_std = np.std(ratios)
    valid_ratios = ratios[np.abs(ratios - ratio_median) < 3 * ratio_std]

    if len(valid_ratios) < min_valid_pixels // 2:
        scale_factor = ratio_median
    else:
        scale_factor = np.median(valid_ratios)

    print(f"  [ScaleAlign] Computed scale factor: {scale_factor:.4f} (from {num_valid} pixels)")
    return float(scale_factor)

# =============================================================================
# Reference Frame Selection (3D IoU)
# =============================================================================

def _unproject_depth_to_points(
    depth: np.ndarray,
    K: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Unproject depth map to 3D points in camera coordinates."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    if mask is None:
        mask = depth > 0

    u_valid = u[mask]
    v_valid = v[mask]
    z_valid = depth[mask]

    if len(z_valid) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy

    return np.stack([x, y, z_valid], axis=-1).astype(np.float32)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform points by a 4x4 transformation matrix."""
    if points.size == 0:
        return points
    R = transform[:3, :3]
    t = transform[:3, 3]
    return (points @ R.T) + t



def _merge_pointcloud_incremental(
    base_points: Optional[np.ndarray],
    base_colors: Optional[np.ndarray],
    new_points: np.ndarray,
    new_colors: Optional[np.ndarray],
    voxel_size: float,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Merge new points into existing point cloud using voxel occupancy."""
    if base_points is None or len(base_points) == 0:
        return new_points, new_colors
    if new_points is None or len(new_points) == 0:
        return base_points, base_colors

    voxel_size = float(voxel_size)
    base_vox = np.floor(base_points / voxel_size).astype(np.int64)
    new_vox = np.floor(new_points / voxel_size).astype(np.int64)

    dtype = np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    base_view = base_vox.view(dtype).reshape(-1)
    new_view = new_vox.view(dtype).reshape(-1)

    keep_mask = ~np.isin(new_view, base_view)
    if not np.any(keep_mask):
        return base_points, base_colors

    new_view_kept = new_view[keep_mask]
    _, uniq_idx = np.unique(new_view_kept, return_index=True)
    keep_indices = np.flatnonzero(keep_mask)[uniq_idx]

    merged_points = np.concatenate([base_points, new_points[keep_indices]], axis=0)
    if base_colors is not None and new_colors is not None:
        merged_colors = np.concatenate([base_colors, new_colors[keep_indices]], axis=0)
    else:
        merged_colors = base_colors if base_colors is not None else new_colors
    return merged_points, merged_colors


def _voxel_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Compute voxel indices for points."""
    return np.floor(points / voxel_size).astype(np.int64)


def _occupancy_from_frame(
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    voxel_size: float,
    valid_mask: Optional[np.ndarray] = None,
    dynamic_mask: Optional[np.ndarray] = None,
) -> Set[Tuple[int, int, int]]:
    """Compute voxel occupancy set from a single frame's depth map."""
    mask = depth > 0
    if valid_mask is not None:
        mask = mask & valid_mask
    if dynamic_mask is not None:
        mask = mask & (~dynamic_mask)

    points_cam = _unproject_depth_to_points(depth, K, mask=mask)
    if points_cam.size == 0:
        return set()

    points_world = _transform_points(points_cam, c2w)
    vox = _voxel_indices(points_world, voxel_size)
    return set(map(tuple, vox.tolist()))


def _iou_occupancy(a: Set[Tuple[int, int, int]], b: Set[Tuple[int, int, int]]) -> float:
    """Compute IoU between two voxel occupancy sets."""
    if not a and not b:
        return 0.0
    inter = a.intersection(b)
    union = a.union(b)
    if not union:
        return 0.0
    return float(len(inter)) / float(len(union))


@dataclass
class RefSelectionResult:
    """Result of reference frame selection with diagnostic info."""
    indices: List[int] = field(default_factory=list)
    ious: List[float] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.indices)

    def get_status_str(self) -> str:
        """Get a human-readable status string for logging."""
        if self.count > 0:
            return f"ref={self.count}, best_iou={self.stats.get('best_iou', 0):.3f}"
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
    stride: int = 1,
    iou_threshold: float = 0.04,
    max_refs: int = 7,
    valid_masks: Optional[np.ndarray] = None,
    dynamic_masks: Optional[np.ndarray] = None,
    max_iou_frames: int = 0,
) -> RefSelectionResult:
    """
    Select reference frames based on spatial overlap with target frames.

    Following LiveWorld paper (Algorithm 1): For each target frame, find the candidate
    with highest IoU. A candidate is selected as reference if its max IoU with any
    target frame exceeds the threshold.

    Args:
        candidate_indices: Indices of candidate frames
        target_indices: Indices of target frames
        depths: [T, H, W] depth maps for all frames
        intrinsics: [T, 3, 3] intrinsic matrices
        poses_c2w: [T, 4, 4] camera-to-world poses
        voxel_size: Voxel grid size for IoU computation
        stride: Stride for sampling candidates (default 1)
        iou_threshold: Minimum IoU to select a reference frame
        max_refs: Maximum number of reference frames to select
        valid_masks: [T, H, W] valid depth masks (optional)
        dynamic_masks: [T, H, W] dynamic object masks (optional)
        max_iou_frames: Kept for backward compatibility (no longer limits candidates)

    Returns:
        RefSelectionResult with selected indices, IoUs, and stats
    """
    stats = {
        "threshold": iou_threshold,
        "max_refs": max_refs,
        "stride": stride,
        "voxel_size": voxel_size,
        "max_iou_frames": max_iou_frames,
    }

    if max_refs <= 0:
        stats["no_ref_reason"] = "max_refs_zero"
        stats["best_iou"] = 0.0
        return RefSelectionResult([], [], stats)

    target_list = list(target_indices)
    candidates = list(candidate_indices)
    if stride > 1:
        candidates = candidates[::stride]

    stats["num_targets"] = len(target_list)
    stats["num_candidates"] = len(candidates)

    if not candidates:
        stats["no_ref_reason"] = "no_candidates"
        stats["best_iou"] = 0.0
        return RefSelectionResult([], [], stats)

    # Pre-compute occupancy for all target frames
    target_occs = []
    for idx in target_list:
        occ = _occupancy_from_frame(
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
        occ = _occupancy_from_frame(
            depth=depths[idx],
            K=intrinsics[idx],
            c2w=poses_c2w[idx],
            voxel_size=voxel_size,
            valid_mask=None if valid_masks is None else valid_masks[idx],
            dynamic_mask=None if dynamic_masks is None else dynamic_masks[idx],
        )
        candidate_occs[idx] = occ

    # For each candidate, compute max IoU across all target frames
    scored = []
    all_ious = []
    for c_idx in candidates:
        c_occ = candidate_occs[c_idx]
        max_iou = 0.0
        for t_occ in target_occs:
            iou = _iou_occupancy(c_occ, t_occ)
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

    # Prefer earliest frames when multiple candidates pass the threshold
    scored.sort(key=lambda x: x[0])
    selected = scored[:max_refs]
    indices = [s[0] for s in selected]
    ious = [s[1] for s in selected]

    return RefSelectionResult(indices, ious, stats)


# =============================================================================
# Projection Rendering Utilities
# =============================================================================

def render_projection(
    points_world: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    image_size: Tuple[int, int],
    channels: Iterable[str],
    colors: Optional[np.ndarray] = None,
    fill_holes_kernel: int = 0,
    device: str = "cuda",
) -> np.ndarray:
    """Render point cloud projection (GPU-accelerated). Delegates to scripts.dataset_preparation._projection."""
    from scripts.create_train_data._projection import render_projection as _render_proj
    return _render_proj(
        points_world=points_world,
        K=K,
        c2w=c2w,
        image_size=image_size,
        channels=channels,
        colors=colors,
        fill_holes_kernel=fill_holes_kernel,
        device=device,
    )


def scale_intrinsics(K: np.ndarray, scale_x, scale_y) -> np.ndarray:
    """
    Scale intrinsic matrix for different resolutions.

    Args:
        K: [3, 3] intrinsic matrix
        scale_x: Scale factor for x (width), or source (h, w) tuple
        scale_y: Scale factor for y (height), or target (h, w) tuple

    Returns:
        K_scaled: [3, 3] scaled intrinsic matrix
    """
    if isinstance(scale_x, tuple) and isinstance(scale_y, tuple):
        from_h, from_w = scale_x
        to_h, to_w = scale_y
        scale_x = to_w / from_w
        scale_y = to_h / from_h
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy
    return K_scaled


# =============================================================================
# Projection Density Helpers
# =============================================================================

def _project_points_to_pixels(
    points_world: np.ndarray,
    pose_c2w: np.ndarray,
    K: np.ndarray,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D points to image pixels."""
    height, width = image_size
    pose_w2c = np.linalg.inv(pose_c2w)
    R = pose_w2c[:3, :3]
    t_vec = pose_w2c[:3, 3]

    points_cam = (R @ points_world.T).T + t_vec
    z = points_cam[:, 2]
    uv = (K @ points_cam.T).T
    uv = uv[:, :2] / (uv[:, 2:3] + 1e-8)

    valid = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 0.0)
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)

    idx = np.nonzero(valid)[0]
    return idx, u[valid], v[valid], z[valid]


def _compute_projection_density_max_pixels(
    points_world: np.ndarray,
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    target_frames: List[int],
    output_size: Tuple[int, int],
    intrinsics_size: Tuple[int, int],
) -> int:
    """Compute max projected unique pixel count from the first iteration."""
    height, width = output_size
    if points_world is None or len(points_world) == 0:
        return 0

    if intrinsics.ndim == 2:
        intrinsics = np.tile(intrinsics[None], (len(poses_c2w), 1, 1))

    proc_h, proc_w = intrinsics_size
    max_pixels = 0


    for frame_idx in target_frames:
        pose_idx = _safe_frame_index(frame_idx, len(poses_c2w))
        intr_idx = _safe_frame_index(frame_idx, len(intrinsics))
        K_scaled = scale_intrinsics_from_size(intrinsics[intr_idx], (proc_h, proc_w), (height, width))
        idx, u, v, _ = _project_points_to_pixels(points_world, poses_c2w[pose_idx], K_scaled, (height, width))
        if idx.size == 0:
            continue
        flat = v.astype(np.int64) * width + u.astype(np.int64)
        unique_flat = np.unique(flat)
        max_pixels = max(max_pixels, int(unique_flat.size))

    return max_pixels


def _limit_points_by_density(
    points_world: np.ndarray,
    colors: np.ndarray,
    pose_c2w: np.ndarray,
    K: np.ndarray,
    image_size: Tuple[int, int],
    max_pixels: int,
    rng: Optional[np.random.Generator] = None,
    blue_noise: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Limit projected point density by blue-noise sampling."""
    if max_pixels is None or max_pixels <= 0:
        return points_world, colors

    height, width = image_size
    idx, u, v, z = _project_points_to_pixels(points_world, pose_c2w, K, (height, width))
    if idx.size == 0:
        empty_xyz = np.zeros((0, 3), dtype=np.float32)
        empty_rgb = np.zeros((0, 3), dtype=np.uint8) if colors is not None else None
        return empty_xyz, empty_rgb

    flat = v.astype(np.int64) * width + u.astype(np.int64)

    order_pix = np.lexsort((z, flat))
    flat_sorted = flat[order_pix]
    idx_sorted = idx[order_pix]
    z_sorted = z[order_pix]
    unique_flat, first_idx = np.unique(flat_sorted, return_index=True)

    pixel_idx = idx_sorted[first_idx]
    if pixel_idx.size <= max_pixels:
        return points_world[pixel_idx], colors[pixel_idx] if colors is not None else None

    v_pix = unique_flat // width
    u_pix = unique_flat - v_pix * width

    if rng is None:
        rng = np.random.default_rng()

    if blue_noise is not None and blue_noise.size > 0:
        tile_h, tile_w = blue_noise.shape
        off_y = int(rng.integers(0, tile_h))
        off_x = int(rng.integers(0, tile_w))
        noise_val = blue_noise[(v_pix + off_y) % tile_h, (u_pix + off_x) % tile_w]
    else:
        noise_val = rng.random(len(pixel_idx))

    select = np.argpartition(noise_val, max_pixels - 1)[:max_pixels]
    keep_idx = pixel_idx[select]
    return points_world[keep_idx], colors[keep_idx] if colors is not None else None


# =============================================================================
# Reference Frame Retrieval
# =============================================================================
def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel grid downsampling: keep one point per voxel cell.

    Much better than random subsampling for preserving spatial structure.
    Returns the centroid of points in each occupied voxel.
    """
    if len(points) == 0 or voxel_size <= 0:
        return points
    vox_ids = np.floor(points / voxel_size).astype(np.int64)
    # Encode 3D voxel coord to 1D key for fast unique
    mn = vox_ids.min(axis=0)
    vox_ids -= mn
    mx = vox_ids.max(axis=0) + 1
    keys = vox_ids[:, 0] * (mx[1] * mx[2]) + vox_ids[:, 1] * mx[2] + vox_ids[:, 2]
    _, unique_idx = np.unique(keys, return_index=True)
    return points[unique_idx]


def _compute_3d_iou_numpy(points_a: np.ndarray, points_b: np.ndarray, voxel_size: float) -> float:
    if len(points_a) == 0 or len(points_b) == 0:
        return 0.0

    vox_a = np.floor(points_a / voxel_size).astype(np.int32)
    vox_b = np.floor(points_b / voxel_size).astype(np.int32)

    set_a = set(map(tuple, vox_a))
    set_b = set(map(tuple, vox_b))

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0

    return intersection / union


def compute_3d_iou_batched(
    target_points: np.ndarray,
    hist_points_list: List[Tuple[int, np.ndarray]],
    voxel_size: float,
    device: str | torch.device = "cuda",
    max_points: int = 50000,
) -> List[Tuple[int, float]]:
    """Batch compute 3D IoU between a target point cloud and multiple histories.

    Pre-step: voxel downsample all point clouds (same voxel_size as IoU) to
    drastically reduce point count while preserving spatial structure, then
    deterministically trim (if needed) and compute IoU on GPU.

    Args:
        target_points: Target point cloud (N, 3)
        hist_points_list: List of (frame_idx, points) tuples
        voxel_size: Voxel size for discretization
        device: Device for computation
        max_points: Max points per frame after voxel downsample (deterministic
            trim if still too many). 0 = no limit.
    """
    if len(target_points) == 0 or len(hist_points_list) == 0:
        return []

    device = torch.device(device)

    valid_hist = [(idx, pts) for idx, pts in hist_points_list if pts is not None and len(pts) > 0]
    if len(valid_hist) == 0:
        return []

    # Voxel downsample only (preserves spatial structure without lossy truncation).
    target_points_ds = _voxel_downsample(target_points, voxel_size)
    valid_hist_ds = [
        (idx, _voxel_downsample(pts, voxel_size))
        for idx, pts in valid_hist
    ]

    # Try GPU computation with OOM fallback to CPU.
    try:
        return _compute_3d_iou_batched_gpu(target_points_ds, valid_hist_ds, voxel_size, device)
    except torch.cuda.OutOfMemoryError:
        print("    [WARNING] GPU OOM during 3D IoU, falling back to CPU...")
        torch.cuda.empty_cache()
        results = []
        for frame_idx, pts in valid_hist_ds:
            iou = _compute_3d_iou_numpy(target_points_ds, pts, voxel_size)
            results.append((frame_idx, iou))
        return results


def _compute_3d_iou_batched_gpu(
    target_points: np.ndarray,
    valid_hist: List[Tuple[int, np.ndarray]],
    voxel_size: float,
    device: torch.device,
) -> List[Tuple[int, float]]:
    """GPU implementation of batched 3D IoU computation."""
    target_t = torch.as_tensor(target_points, device=device, dtype=torch.float32)
    target_vox = torch.floor(target_t / voxel_size).to(torch.int64)

    all_hist_points = []
    batch_lengths = []
    frame_indices = []

    for frame_idx, pts in valid_hist:
        all_hist_points.append(pts)
        batch_lengths.append(len(pts))
        frame_indices.append(frame_idx)

    all_hist_concat = np.concatenate(all_hist_points, axis=0)
    hist_t = torch.as_tensor(all_hist_concat, device=device, dtype=torch.float32)
    hist_vox = torch.floor(hist_t / voxel_size).to(torch.int64)

    batch_id_list = []
    for batch_idx, length in enumerate(batch_lengths):
        batch_id_list.append(torch.full((length,), batch_idx, device=device, dtype=torch.int64))
    batch_ids_t = torch.cat(batch_id_list)

    num_batches = len(valid_hist)

    all_vox = torch.cat([target_vox, hist_vox], dim=0)
    global_min = all_vox.min(dim=0).values

    target_vox = target_vox - global_min
    hist_vox = hist_vox - global_min

    global_max = torch.cat([target_vox, hist_vox], dim=0).max(dim=0).values
    ranges = (global_max + 1).to(torch.int64)

    ranges_cpu = ranges.detach().cpu().tolist()
    max_index = ranges_cpu[0] * ranges_cpu[1] * ranges_cpu[2]
    if max_index >= torch.iinfo(torch.int64).max // (num_batches + 1):
        # Fall back to numpy for very large index ranges
        results = []
        for frame_idx, pts in valid_hist:
            iou = _compute_3d_iou_numpy(target_points, pts, voxel_size)
            results.append((frame_idx, iou))
        return results

    stride_y = ranges[2]
    stride_x = ranges[1] * stride_y

    target_keys = target_vox[:, 0] * stride_x + target_vox[:, 1] * stride_y + target_vox[:, 2]
    hist_keys = hist_vox[:, 0] * stride_x + hist_vox[:, 1] * stride_y + hist_vox[:, 2]

    target_keys_unique = torch.unique(target_keys)
    target_count = target_keys_unique.numel()

    max_voxel_key = stride_x * ranges[0]
    composite_keys = batch_ids_t * max_voxel_key + hist_keys

    unique_composite, _ = torch.unique(composite_keys, return_inverse=True)

    unique_batch_ids = unique_composite // max_voxel_key
    unique_voxel_keys = unique_composite % max_voxel_key

    hist_counts = torch.zeros(num_batches, device=device, dtype=torch.int64)
    hist_counts.scatter_add_(0, unique_batch_ids, torch.ones_like(unique_batch_ids))

    target_keys_sorted = torch.sort(target_keys_unique).values
    search_idx = torch.searchsorted(target_keys_sorted, unique_voxel_keys)
    valid_idx = search_idx < target_keys_sorted.numel()
    is_in_target = valid_idx & (
        target_keys_sorted[search_idx.clamp(max=target_keys_sorted.numel() - 1)] == unique_voxel_keys
    )

    intersections = torch.zeros(num_batches, device=device, dtype=torch.int64)
    intersections.scatter_add_(0, unique_batch_ids, is_in_target.to(torch.int64))

    unions = target_count + hist_counts - intersections
    ious = intersections.float() / unions.float().clamp(min=1e-8)
    ious = torch.where(unions > 0, ious, torch.zeros_like(ious))

    ious_cpu = ious.cpu().tolist()
    results = [(frame_indices[i], ious_cpu[i]) for i in range(num_batches)]

    return results


def compute_3d_iou(
    points_a: np.ndarray,
    points_b: np.ndarray,
    voxel_size: float = 0.1,
    device: str | torch.device | None = None,
) -> float:
    """Compute 3D IoU between two point clouds."""
    if device is None:
        return _compute_3d_iou_numpy(points_a, points_b, voxel_size)

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return _compute_3d_iou_numpy(points_a, points_b, voxel_size)

    if len(points_a) == 0 or len(points_b) == 0:
        return 0.0

    pts_a = torch.as_tensor(points_a, device=device, dtype=torch.float32)
    pts_b = torch.as_tensor(points_b, device=device, dtype=torch.float32)
    vox_a = torch.floor(pts_a / voxel_size).to(torch.int64)
    vox_b = torch.floor(pts_b / voxel_size).to(torch.int64)

    mins = torch.minimum(vox_a.min(dim=0).values, vox_b.min(dim=0).values)
    vox_a = vox_a - mins
    vox_b = vox_b - mins

    maxs = torch.maximum(vox_a.max(dim=0).values, vox_b.max(dim=0).values)
    ranges = (maxs + 1).to(torch.int64)

    ranges_cpu = ranges.detach().cpu().tolist()
    max_index = ranges_cpu[0] * ranges_cpu[1] * ranges_cpu[2]
    if max_index >= torch.iinfo(torch.int64).max:
        return _compute_3d_iou_numpy(points_a, points_b, voxel_size)

    stride_y = ranges[2]
    stride_x = ranges[1] * stride_y

    keys_a = vox_a[:, 0] * stride_x + vox_a[:, 1] * stride_y + vox_a[:, 2]
    keys_b = vox_b[:, 0] * stride_x + vox_b[:, 1] * stride_y + vox_b[:, 2]

    keys_a = torch.unique(keys_a)
    keys_b = torch.unique(keys_b)

    if keys_a.numel() == 0 or keys_b.numel() == 0:
        return 0.0

    if keys_a.numel() > keys_b.numel():
        keys_a, keys_b = keys_b, keys_a

    keys_a = torch.sort(keys_a).values
    keys_b = torch.sort(keys_b).values

    idx = torch.searchsorted(keys_b, keys_a)
    valid = idx < keys_b.numel()
    hits = valid & (keys_b[idx.clamp(max=keys_b.numel() - 1)] == keys_a)

    intersection = int(hits.sum().item())
    union = int(keys_a.numel() + keys_b.numel() - intersection)
    if union == 0:
        return 0.0

    return intersection / union


def get_visible_points_for_frame(
    points_world: np.ndarray,
    colors: np.ndarray,
    pose_c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_size: Tuple[int, int],
    depth_threshold: float = 100.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return points visible from a specific camera viewpoint."""
    height, width = image_size

    pose_w2c = np.linalg.inv(pose_c2w)
    R = pose_w2c[:3, :3]
    t_vec = pose_w2c[:3, 3]

    points_cam = (R @ points_world.T).T + t_vec

    valid_depth = points_cam[:, 2] > 0.01
    valid_depth &= points_cam[:, 2] < depth_threshold

    K = intrinsics
    points_proj = (K @ points_cam.T).T
    points_proj = points_proj[:, :2] / (points_proj[:, 2:3] + 1e-8)

    valid_x = (points_proj[:, 0] >= 0) & (points_proj[:, 0] < width)
    valid_y = (points_proj[:, 1] >= 0) & (points_proj[:, 1] < height)

    valid = valid_depth & valid_x & valid_y

    return points_world[valid], colors[valid] if colors is not None else None


def _get_visible_points_and_coverage_gpu(
    points_world: np.ndarray,
    poses_c2w: np.ndarray,
    intrinsics_per_frame: np.ndarray,
    frame_indices: List[int],
    image_size: Tuple[int, int],
    depth_threshold: float = 100.0,
    device: str | torch.device = "cuda",
) -> List[Tuple[int, np.ndarray, float, int]]:
    """Batch-compute visible points + pixel coverage for multiple frames on GPU.

    For each frame: project world points → camera → image, filter by depth and
    bounds, count unique covered pixels.  All heavy matrix ops run on GPU;
    only the per-frame visible-point subsets are copied back to CPU.

    Args:
        points_world: (N, 3) world points.
        poses_c2w: (F, 4, 4) camera-to-world for ALL poses (indexed by frame_indices).
        intrinsics_per_frame: (F, 3, 3) per-pose intrinsics (same length as poses_c2w).
        frame_indices: Which pose indices to process.
        image_size: (height, width).
        depth_threshold: Max depth to keep.
        device: GPU device.

    Returns:
        List of (frame_idx, visible_pts_np, coverage, n_unique_pixels) per frame.
    """
    height, width = image_size
    total_pixels = max(1, height * width)
    dev = torch.device(device)

    # Upload points once.
    pts_t = torch.as_tensor(points_world, device=dev, dtype=torch.float32)  # (N, 3)
    N = pts_t.shape[0]

    results = []
    for frame_idx in frame_indices:
        intr_idx = min(frame_idx, len(intrinsics_per_frame) - 1)
        c2w = poses_c2w[frame_idx]
        w2c = np.linalg.inv(c2w)
        R_t = torch.as_tensor(w2c[:3, :3], device=dev, dtype=torch.float32)
        t_t = torch.as_tensor(w2c[:3, 3], device=dev, dtype=torch.float32)
        K_t = torch.as_tensor(intrinsics_per_frame[intr_idx], device=dev, dtype=torch.float32)

        # Transform to camera space: (N, 3)
        pts_cam = pts_t @ R_t.T + t_t  # (N, 3)

        # Depth filter
        z = pts_cam[:, 2]
        valid = (z > 0.01) & (z < depth_threshold)

        # Project to image
        pts_proj = pts_cam @ K_t.T  # (N, 3)
        px = pts_proj[:, :2] / (pts_proj[:, 2:3] + 1e-8)

        # Bounds filter
        valid = valid & (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)

        # Extract visible points (to CPU)
        valid_mask = valid.cpu().numpy()
        visible_pts = points_world[valid_mask]

        if len(visible_pts) == 0:
            results.append((frame_idx, visible_pts, 0.0, 0))
            continue

        # Pixel coverage: count unique pixel IDs among valid points (stay on GPU).
        px_valid = px[valid]
        px_int = torch.floor(px_valid).to(torch.int64)
        pixel_ids = px_int[:, 1] * width + px_int[:, 0]
        n_unique = int(torch.unique(pixel_ids).numel())
        coverage = n_unique / total_pixels

        results.append((frame_idx, visible_pts, coverage, n_unique))

    return results
@dataclass
class BackboneInferenceOptions:
    """Options for iterative LiveWorld inference.

    Key design:
    - Extrinsics (poses): Always from geometry.npz
    - Intrinsics: Estimated by 3D handler (Stream3R / MapAnything)
    - Point cloud: Managed by 3D handler
    - Sky masking: SAM3 per-frame sky segmentation
    """
    # ============ Inference Settings ============
    infer_steps: int = 50
    no_cfg: bool = False
    cpu_offload: bool = False
    seed: int = 42
    fps: int = 16
    # Few-step mode (CausVid-style): predict x0 at each step, then add noise to next step.
    use_few_step: bool = False
    denoising_step_list: Optional[List[float]] = None  # e.g. [1000, 757, 522]

    # ============ Frame Selection ============
    max_reference_frames: int = 7  # 0 = disable reference frames
    max_preceding_frames_first_iter: int = 1  # Preceding frames for iter 0 / P1 mode (0 = no first-frame concat)
    max_preceding_frames_other_iter: int = 9  # Preceding frames for iter 1+ / P9 mode (0 = no preceding)
    preceding_noise_timestep: int = 0  # Add noise to P9 preceding frames (0 = clean, 300 = match training aug)
    limit_projection_density: bool = False
    projection_density_noise_size: int = 64
    sp_context_scale: float = 1.0  # Scale for State Adapter conditioning at inference
    merge_initial_pointcloud: bool = True  # Always include first-iteration points in later projections
    # ============ Point Cloud Backend ============
    pointcloud_backend: str = "stream3r"  # "stream3r" or "map_anything"
    use_icp: bool = True  # Enable ICP alignment in point cloud updates (disable to skip all ICP)

    # ============ Stream3R Point Cloud Update ============
    stream3r_model_path: str = "ckpts/yslan--STream3R"
    stream3r_window_size: Optional[int] = None  # null = causal; integer = window mode (keep first 1 + last N frames)
    stream3r_preprocess_mode: str = "crop"  # "crop" or "pad"
    stream3r_frame_sample_rate: int = 1  # Sample every Nth frame (1 = all, 4 = every 4th)
    stream3r_icp_conf_percentile: float = 70.0   # ICP uses top (100 - this)% points
    stream3r_keep_conf_percentile: float = 5.0  # Final merge uses top (100 - this)% points
    stream3r_icp_threshold: float = 0.05
    stream3r_icp_max_iter: int = 200
    stream3r_min_new_point_dist: float = 0.005
    stream3r_outlier_nb_neighbors: int = 20
    stream3r_outlier_std_ratio: float = 2.0
    stream3r_global_icp_threshold: float = 0.1
    stream3r_global_icp_max_iter: int = 200
    stream3r_global_icp_voxel_size: Optional[float] = None
    stream3r_global_icp_min_fitness: float = 0.01
    stream3r_merge_voxel_size: float = 0.001
    stream3r_update_mode: str = "complete"  # "complete": fill unseen areas; "freeze": never alter historical pose projections
    stream3r_use_3d_dist_filter: bool = True  # Use 3D distance to filter duplicate points during merge
    stream3r_use_dynamic_mask_in_update: bool = True  # Run Qwen+SAM3 to remove dynamic objects during update
    stream3r_consistency_threshold: float = 0.05  # Relative depth tolerance for multi-view consistency

    # ============ MapAnything Point Cloud Update ============
    map_anything_model_path: str = "ckpts/facebook--map-anything"
    map_anything_frame_sample_rate: int = 1
    map_anything_merge_every_n: int = 5
    map_anything_conf_percentile: float = 60.0
    map_anything_overlap_dist: float = 0.02
    map_anything_min_new_point_dist: float = 0.005
    map_anything_icp_threshold: float = 0.05
    map_anything_use_2d_coverage: bool = True  # Use poses/intrinsics for 2D overlap masking
    map_anything_use_dynamic_mask: bool = True  # Remove dynamic/sky points each iteration via SAM3

    # ============ Model Paths ============
    qwen_model_path: str = "ckpts/Qwen--Qwen3-VL-8B-Instruct"  # For dynamic object detection
    sam3_model_path: str = "ckpts/facebook--sam3/sam3.pt"  # For dynamic object segmentation

    # ============ Point Cloud ============
    voxel_size: float = 0.001  # Voxel size for initial point cloud and 3D IoU
    depth_edge_erosion: int = 10  # Erode depth edges by N pixels (0 = disabled)

    # ============ Target Resolution ============
    target_hw: Tuple[int, int] = (720, 1280)  # (H, W) from image_or_video_shape; set automatically

    # ============ 3D IoU Config ============
    iou_device: str = "auto"  # auto, cpu, cuda
    voxel_size_iou: Optional[float] = None  # Voxel size for 3D IoU computation (None = voxel_size * 2)
    max_iou_frames: int = 10  # Backward-compatible no-op (kept in config/API)
    max_iou_points: int = 20000  # Max points per frame for IoU (downsample if more)


