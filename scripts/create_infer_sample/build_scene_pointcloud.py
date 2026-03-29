#!/usr/bin/env python3
"""
Build first-frame scene point cloud from a single image using Depth Anything 3.

Output:
- <output_dir>/<stem>_da3_scene.ply
- <output_dir>/<stem>_da3_scene_meta.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from liveworld.geometry_utils import (
    load_image,
    resize_short_edge_and_center_crop,
    scale_intrinsics,
    save_point_cloud_ply,
)
from scripts.create_train_data._estimators import Stream3REstimator
from scripts.create_train_data._sample_builder import build_scene_point_cloud


def _prepare_image_to_target(image_rgb: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    # Prefer aspect-preserving short-edge resize + center-crop.
    try:
        return resize_short_edge_and_center_crop(image_rgb, target_h, target_w)
    except ValueError:
        pass

    # Fallback: letterbox to avoid distortion and guarantee exact target size.
    h, w = image_rgb.shape[:2]
    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def build_scene_pointcloud(
    image_path: str,
    output_dir: str,
    target_h: int = 480,
    target_w: int = 832,
    voxel_size: float = 0.01,
    depth_max: float | None = None,
) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path, rgb=True)
    image_cropped = _prepare_image_to_target(image, target_h, target_w)

    # Config is passed via function args
    estimator = Stream3REstimator(cfg)
    geometry = estimator.estimate_video_geometry([image_cropped])

    depth_proc = geometry.depths[0]
    intrinsics_proc = geometry.intrinsics[0]
    c2w = geometry.poses_c2w[0]
    processed_frame = geometry.frames[0]
    proc_h, proc_w = geometry.processed_size

    depth = cv2.resize(depth_proc, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    intrinsics = scale_intrinsics(intrinsics_proc, (proc_h, proc_w), (target_h, target_w))
    frame_scaled = cv2.resize(processed_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    points_world, colors = build_scene_point_cloud(
        depth=depth,
        K=intrinsics,
        c2w=c2w,
        rgb=frame_scaled,
        voxel_size=voxel_size,
        depth_max=depth_max,
    )
    if colors is None:
        colors = np.full((len(points_world), 3), 255, dtype=np.uint8)

    stem = Path(image_path).stem
    ply_path = out_dir / f"{stem}_da3_scene.ply"
    meta_path = out_dir / f"{stem}_da3_scene_meta.npz"

    save_point_cloud_ply(ply_path, points_world, colors)
    np.savez_compressed(
        meta_path,
        points_world=points_world.astype(np.float32),
        colors=colors.astype(np.uint8),
        intrinsics=intrinsics.astype(np.float32),
        c2w=c2w.astype(np.float32),
        target_h=int(target_h),
        target_w=int(target_w),
        # DA3 processed-size geometry (needed for geometry.npz export)
        depth_proc=depth_proc.astype(np.float32),
        intrinsics_proc=intrinsics_proc.astype(np.float32),
        processed_size=np.array([proc_h, proc_w]),
    )

    del estimator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ply_path, meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scene pointcloud from first frame image (DA3)")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--output-dir", default="outputs/event_bench/scene_pointclouds")
    parser.add_argument("--target-height", type=int, default=480)
    parser.add_argument("--target-width", type=int, default=832)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--depth-max", type=float, default=None)
    args = parser.parse_args()

    ply_path, meta_path = build_scene_pointcloud(
        image_path=args.image_path,
        output_dir=args.output_dir,
        target_h=args.target_height,
        target_w=args.target_width,
        voxel_size=args.voxel_size,
        depth_max=args.depth_max,
    )
    print(f"[build_scene_pointcloud_da3] ply: {ply_path}")
    print(f"[build_scene_pointcloud_da3] meta: {meta_path}")


if __name__ == "__main__":
    main()
