"""
动态物体深度更新脚本

功能1 - 两张图片模式:
输入：两张图片，第二张和第一张的区别是运动物体的深度不同（相机位置和视角完全一样）
输出：将第二帧中运动物体的点云合并到第一帧的点云中

功能2 - 视频模式:
输入：一个视频，每帧除了前景物体深度不同外，背景完全静止
输出：把每帧的前景物体点云都合并到第一帧点云中，从偏移视角渲染出视频

流程：
1. 用 STream3R 估计深度和内参
2. 用 SAM3 分割前景（动态物体）和背景
3. 计算背景区域的深度尺度差异（用于 scale 对齐）
4. 将前景物体的点云（经过 scale 对齐）合并到第一帧的点云中
5. (视频模式) 从偏移视角渲染累积点云
"""

import sys
import tempfile
from pathlib import Path


import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

from stream3r.models.stream3r import STream3R
from stream3r.models.components.utils.load_fn import load_and_preprocess_images
from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri
from stream3r.stream_session import StreamSession
from scripts.create_train_data._sam3_segmenter import Sam3VideoSegmenter

DEFAULT_CONF_THRESHOLD = 0.10
MIN_CONF_THRESHOLD = 0.0
MAX_CONF_THRESHOLD = 0.60
MIN_SCALE_VALID_PIXELS = 100


def load_image(path: str) -> np.ndarray:
    """加载图片为 RGB numpy array"""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_video_frames(video_path: str) -> list:
    """加载视频所有帧为 RGB numpy array 列表"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


def get_video_fps(video_path: str) -> float:
    """获取视频帧率"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def _normalize_conf_threshold(conf_threshold: float, verbose: bool = True) -> float:
    """Validate confidence threshold for reconstruction."""
    if conf_threshold is None or not np.isfinite(conf_threshold):
        raise ValueError(f"Invalid conf_threshold={conf_threshold}")

    conf_threshold = float(conf_threshold)
    if conf_threshold < MIN_CONF_THRESHOLD or conf_threshold > MAX_CONF_THRESHOLD:
        raise ValueError(
            f"conf_threshold={conf_threshold:.3f} out of range "
            f"[{MIN_CONF_THRESHOLD:.2f}, {MAX_CONF_THRESHOLD:.2f}]"
        )
    return conf_threshold


def load_stream3r_model(device: str = "cuda", model_path: str = "ckpts/yslan--STream3R"):
    """加载 STream3R 模型（只加载一次）"""
    model = STream3R.from_pretrained(model_path)
    model = model.to(device=torch.device(device))
    model.eval()
    return model


def _prepare_stream3r_images_official(
    frames: list[np.ndarray],
    device: str = "cuda",
    mode: str = "crop",
) -> torch.Tensor:
    """Use Stream3R official preprocessing (same path as misc/STream3R/infer_stream3r.py)."""
    if not frames:
        raise ValueError("frames must be non-empty")

    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = []
        for i, frame in enumerate(frames):
            frame_np = np.asarray(frame)
            if frame_np.dtype != np.uint8:
                frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
            path = Path(tmp_dir) / f"frame_{i:05d}.png"
            cv2.imwrite(str(path), cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
            paths.append(str(path))
        images = load_and_preprocess_images(paths, mode=mode)
    return images.to(device=torch.device(device))


def estimate_depth_and_intrinsics(
    image: np.ndarray,
    model,
    device: str = "cuda",
    process_res: int = 518,
    conf_thresh_percentile: float = 40.0,
):
    """
    使用 STream3R 估计单张图片的深度和内参 (独立模式，无 session)

    Args:
        image: RGB image [H, W, 3]
        model: 已加载的 STream3R 模型
        process_res: STream3R 处理分辨率上限 (default: 518)
        conf_thresh_percentile: 保留参数，当前未使用

    Returns:
        depth: [H, W] float32, Z-depth
        intrinsics: [3, 3] float32
        pts3d: None
        c2w: [4, 4] float32, camera to world transform
        conf: [H, W] float32, confidence map
    """

    # Keep process_res for API compatibility; official loader uses fixed target size.
    if process_res != 518:
        print(f"  [Stream3R] Warning: process_res={process_res} ignored; using official preprocessing")

    image_tensor = _prepare_stream3r_images_official([image], device=device, mode="crop")

    with torch.no_grad():
        predictions = model(image_tensor, mode="full")

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], image_tensor.shape[-2:]
    )

    w2c_34 = extrinsic[0, 0].cpu().numpy()
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :] = w2c_34
    c2w = np.linalg.inv(w2c).astype(np.float32)
    intrinsics = intrinsic[0, 0].cpu().numpy().astype(np.float32)

    wp = predictions.get("world_points")
    if isinstance(wp, torch.Tensor):
        wp = wp.cpu().numpy()
    if wp is None:
        raise RuntimeError("[Stream3R] No world_points in predictions")
    if wp.ndim == 5:
        wp = wp[0, 0]
    elif wp.ndim == 4:
        wp = wp[0]

    conf = predictions.get("world_points_conf")
    if isinstance(conf, torch.Tensor):
        conf = conf.cpu().numpy()
    if conf is None:
        conf = np.ones(wp.shape[:2], dtype=np.float32)
    if conf.ndim == 4:
        conf = conf[0, 0]
    elif conf.ndim == 3:
        conf = conf[0]

    proc_h, proc_w = wp.shape[:2]
    w2c = np.linalg.inv(c2w)
    pts_cam = (w2c[:3, :3] @ wp.reshape(-1, 3).T).T + w2c[:3, 3]
    depth = pts_cam[:, 2].reshape(proc_h, proc_w).astype(np.float32)

    pts3d = None

    return depth, intrinsics, pts3d, c2w, conf.astype(np.float32)


def _extract_depth_from_preds(preds: dict, frame_count: int):
    """Extract depth, intrinsics, conf for the latest frame from session predictions.

    Args:
        preds: Dict of numpy predictions (batch dim already squeezed).
        frame_count: Total frames fed so far (1-indexed).

    Returns:
        Same as estimate_depth_and_intrinsics: (depth, intrinsics, None, c2w, conf)
    """

    # Extract current (last) frame predictions from accumulated outputs.
    cur = {}
    for key, val in preds.items():
        if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == frame_count:
            cur[key] = val[-1:]
        else:
            cur[key] = val

    # pose_enc is a torch tensor in raw predictions; it was already converted.
    # We need the raw tensor for pose_encoding_to_extri_intri. Store it separately.
    pose_enc = cur.get("_pose_enc_tensor")
    if pose_enc is not None:
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, cur["_image_shape"]
        )
        w2c_34 = extrinsic[0, -1].cpu().numpy()  # last frame
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :] = w2c_34
        c2w = np.linalg.inv(w2c).astype(np.float32)
        intrinsics = intrinsic[0, -1].cpu().numpy().astype(np.float32)
    else:
        c2w = np.eye(4, dtype=np.float32)
        intrinsics = np.eye(3, dtype=np.float32)

    wp = cur.get("world_points")
    if wp is None:
        raise RuntimeError("[Stream3R] No world_points in predictions")
    if wp.ndim == 4:
        wp = wp[-1:]
    if wp.ndim == 3:
        wp = wp[np.newaxis]

    conf = cur.get("world_points_conf", np.ones(wp.shape[:-1], dtype=np.float32))
    if conf.ndim >= 2 and conf.shape[0] > 1:
        conf = conf[-1:]
    if conf.ndim == 3:
        conf = conf[0]
    elif conf.ndim == 4:
        conf = conf[0, 0]

    proc_h, proc_w = wp[0].shape[:2]
    w2c = np.linalg.inv(c2w)
    pts_cam = (w2c[:3, :3] @ wp[0].reshape(-1, 3).T).T + w2c[:3, 3]
    depth = pts_cam[:, 2].reshape(proc_h, proc_w).astype(np.float32)

    return depth, intrinsics, None, c2w, conf.astype(np.float32)


def estimate_all_depths_with_session(
    frames: list,
    model,
    device: str = "cuda",
    process_res: int = 518,
    verbose: bool = False,
    session=None,
    initial_frames_fed: int = 0,
):
    """Estimate depth for all frames using a StreamSession for temporal consistency.

    If ``session`` is provided, frames are fed into that shared session
    (no creation / clearing). Otherwise a local session is created and
    cleared after use.

    Args:
        frames: List of RGB images [H, W, 3] uint8.
        model: Pre-loaded STream3R model (used only if session is None).
        device: Compute device string.
        process_res: Stream3R processing resolution.
        verbose: Print per-frame info.
        session: Optional pre-existing StreamSession to reuse.
        initial_frames_fed: How many frames the session has already consumed
            (needed to correctly index the accumulated predictions).

    Returns:
        Tuple of (results, final_frames_fed):
        - results: list of (depth, intrinsics, c2w, conf) per frame.
        - final_frames_fed: total frames fed after this call.
    """
    _log = print if verbose else (lambda *a, **k: None)

    # Keep process_res for API compatibility; official loader uses fixed target size.
    if process_res != 518:
        _log(f"  [Stream3R] Warning: process_res={process_res} ignored; using official preprocessing")

    # Preprocess all frames with official Stream3R loader.
    images = _prepare_stream3r_images_official(frames, device=device, mode="crop")

    owns_session = session is None
    if owns_session:
        session = StreamSession(model, mode="causal")
        initial_frames_fed = 0
        _log("  [Session] Created local session")
    else:
        _log(f"  [Session] Reusing shared session (already fed {initial_frames_fed} frames)")

    results = []
    frames_fed = initial_frames_fed

    with torch.no_grad():
        for i in range(images.shape[0]):
            image = images[i:i + 1]
            predictions = session.forward_stream(image)
            frames_fed += 1

            # Convert tensors to numpy, keeping pose_enc as tensor for later.
            preds = {}
            pose_enc_tensor = None
            for key, val in predictions.items():
                if isinstance(val, torch.Tensor):
                    if key == "pose_enc":
                        pose_enc_tensor = val
                    preds[key] = val.cpu().numpy().squeeze(0)
                else:
                    preds[key] = val

            if pose_enc_tensor is not None:
                preds["_pose_enc_tensor"] = pose_enc_tensor
                preds["_image_shape"] = image.shape[-2:]

            depth, intrinsics, _, c2w, conf = _extract_depth_from_preds(preds, frames_fed)
            results.append((depth, intrinsics, c2w, conf))
            _log(f"  [Session] Frame {i} (global #{frames_fed}): depth {depth.shape}")

    if owns_session:
        session.clear()
        _log(f"  [Session] Local session cleared after {len(results)} frames")
    else:
        _log(f"  [Session] Shared session now at {frames_fed} total frames")

    return results, frames_fed


def load_sam3_segmenter(sam3_model_path: str):
    """加载 SAM3 分割器（只加载一次）"""
    segmenter = Sam3VideoSegmenter(
        checkpoint_path=sam3_model_path,
    )
    return segmenter


def segment_with_sam3(
    image: np.ndarray,
    prompts: list,
    segmenter,
) -> np.ndarray:
    """
    使用 SAM3 分割指定物体

    Args:
        image: RGB image [H, W, 3]
        prompts: 要分割的物体名称列表，如 ["car", "person"]
        segmenter: 已加载的 SAM3 分割器

    Returns:
        mask: [H, W] bool, True 表示被分割的物体区域
    """
    if not prompts:
        return np.zeros(image.shape[:2], dtype=bool)

    # SAM3 需要 PIL Image 列表
    pil_image = Image.fromarray(image)

    try:
        masks = segmenter.segment(
            video_path=[pil_image],
            prompts=prompts,
            frame_index=0,
            expected_frames=1,
        )

        if masks.size == 0 or not masks.any():
            mask = np.zeros(image.shape[:2], dtype=bool)
        else:
            mask = masks[0]  # [H, W]
    except Exception as e:
        print(f"  SAM3 segmentation error: {e}")
        mask = np.zeros(image.shape[:2], dtype=bool)

    return mask


def compute_scale_factor(
    depth1: np.ndarray,
    depth2: np.ndarray,
    background_mask: np.ndarray,
    conf1: np.ndarray = None,
    conf2: np.ndarray = None,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    min_valid_pixels: int = MIN_SCALE_VALID_PIXELS,
    verbose: bool = True,
) -> float:
    """
    计算背景区域的深度尺度因子

    由于深度模型对两张图片分别估计，尺度可能不一致。
    我们用背景区域（静态区域）来计算尺度对齐因子。

    Args:
        depth1: 第一张图的深度 [H, W]
        depth2: 第二张图的深度 [H, W]
        background_mask: 背景区域 mask [H, W], True 表示背景
        conf1: 第一张图置信度 [H, W] (optional)
        conf2: 第二张图置信度 [H, W] (optional)
        conf_threshold: 置信度阈值，低于阈值的像素不参与 scale 估计
        min_valid_pixels: 最小有效像素数

    Returns:
        scale: depth2 * scale ≈ depth1 (在背景区域)
    """
    _log = print if verbose else (lambda *a, **k: None)
    conf_threshold = _normalize_conf_threshold(conf_threshold, verbose=verbose)

    # 基础有效区域（不含置信度）
    base_valid = (
        background_mask
        & (depth1 > 0.1)
        & (depth2 > 0.1)
        & np.isfinite(depth1)
        & np.isfinite(depth2)
    )
    base_valid_pixels = int(base_valid.sum())
    if base_valid_pixels < min_valid_pixels:
        raise RuntimeError(
            "Scale estimation failed: too few valid background pixels "
            f"({base_valid_pixels} < {min_valid_pixels})"
        )

    if (conf1 is None) != (conf2 is None):
        raise ValueError("conf1 and conf2 must be both provided or both None")

    use_conf = conf1 is not None and conf2 is not None
    if use_conf and (conf1.shape != depth1.shape or conf2.shape != depth2.shape):
        raise ValueError(
            "confidence shape mismatch: "
            f"conf1={None if conf1 is None else conf1.shape}, "
            f"conf2={None if conf2 is None else conf2.shape}, "
            f"depth1={depth1.shape}, depth2={depth2.shape}"
        )

    if use_conf:
        valid_mask = base_valid & (conf1 > conf_threshold) & (conf2 > conf_threshold)
        label = f"conf>{conf_threshold:.2f}"
    else:
        valid_mask = base_valid
        label = "no_conf"

    num_valid = int(valid_mask.sum())
    if num_valid < min_valid_pixels:
        raise RuntimeError(
            "Scale estimation failed: too few pixels after filtering "
            f"[{label}] ({num_valid} < {min_valid_pixels})"
        )

    d1 = depth1[valid_mask]
    d2 = depth2[valid_mask]
    ratios = d1 / d2
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size < min_valid_pixels:
        raise RuntimeError(
            "Scale estimation failed: too few finite ratios "
            f"[{label}] ({ratios.size} < {min_valid_pixels})"
        )

    # Robustify: remove ratio tails to reduce reflective/glass outliers.
    p5, p95 = np.percentile(ratios, [5, 95])
    robust = ratios[(ratios >= p5) & (ratios <= p95)]
    ratios_used = robust if robust.size >= min_valid_pixels else ratios

    scale = float(np.median(ratios_used))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"Scale estimation produced invalid scale={scale}")

    std = float(np.std(ratios_used))
    _log(
        f"  Background scale [{label}]: scale={scale:.4f}, std={std:.4f}, "
        f"num_pixels={ratios_used.size}"
    )
    return scale


def unproject_to_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    mask: np.ndarray = None,
    conf: np.ndarray = None,
    conf_threshold: float = 0.1,
    filter_depth_outliers: bool = False,
    depth_outlier_std_scale: float = 3.0,
) -> tuple:
    """
    将深度图 unproject 到相机坐标系下的点云

    Args:
        depth: [H, W] Z-depth
        rgb: [H, W, 3] RGB colors
        intrinsics: [3, 3] camera intrinsics
        mask: [H, W] bool, 只提取 mask=True 的点 (optional)
        conf: [H, W] confidence map (optional)
        conf_threshold: 置信度阈值，低于此值的点被过滤 (default: 0.1)
        filter_depth_outliers: 是否过滤深度离群点 (default: False)
        depth_outlier_std_scale: 标准差倍数，mean ± std_scale * std (default: 3.0)

    Returns:
        points: [N, 3] 点云坐标（相机坐标系）
        colors: [N, 3] 点云颜色
    """
    H, W = depth.shape

    # 创建像素网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # 相机坐标系下的点
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)  # [H, W, 3]

    # 有效性 mask
    valid = (z > 0.1) & np.isfinite(z)
    if mask is not None:
        valid = valid & mask
    if conf is not None:
        valid = valid & (conf > conf_threshold)

    # 基于深度的离群点过滤（针对 mask 区域内的点）
    if filter_depth_outliers and mask is not None:
        # 计算 mask 区域内有效深度的统计信息
        valid_depths = z[valid]
        if len(valid_depths) > 10:  # 至少有足够的点才做统计
            # 用 mean ± std_scale * std 过滤离群点
            mean_depth = np.mean(valid_depths)
            std_depth = np.std(valid_depths)
            lower_bound = mean_depth - depth_outlier_std_scale * std_depth
            upper_bound = mean_depth + depth_outlier_std_scale * std_depth
            # 过滤深度离群点
            depth_inlier = (z >= lower_bound) & (z <= upper_bound)
            valid = valid & depth_inlier

    points_valid = points[valid]  # [N, 3]
    colors_valid = rgb[valid]  # [N, 3]

    return points_valid, colors_valid


def save_pointcloud_ply(path: str, points: np.ndarray, colors: np.ndarray):
    """保存点云为 PLY 格式"""
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def save_depth_visualization(depth: np.ndarray, path: str):
    """保存深度图可视化"""
    valid = (depth > 0) & np.isfinite(depth)
    if not valid.any():
        return

    d_min, d_max = depth[valid].min(), depth[valid].max()
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)
    depth_vis = (depth_norm * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
    depth_vis[~valid] = 0
    cv2.imwrite(path, depth_vis)


def render_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple,
    camera_pose: np.ndarray = None,
) -> np.ndarray:
    """
    渲染点云到图像

    Args:
        points: [N, 3] 世界坐标系下的点
        colors: [N, 3] RGB 颜色
        intrinsics: [3, 3] 相机内参
        image_size: (H, W)
        camera_pose: [4, 4] camera-to-world 变换矩阵

    Returns:
        image: [H, W, 3] 渲染的图像
    """
    H, W = image_size

    if camera_pose is not None:
        # world-to-camera
        w2c = np.linalg.inv(camera_pose)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        points_cam = (R @ points.T).T + t
    else:
        points_cam = points

    # 过滤掉相机后面的点
    valid_depth = points_cam[:, 2] > 0.1
    points_cam = points_cam[valid_depth]
    colors_valid = colors[valid_depth]

    if len(points_cam) == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    # 投影到图像平面
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    u = (fx * x / z + cx).astype(np.int32)
    v = (fy * y / z + cy).astype(np.int32)

    # 过滤超出图像范围的点
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid]
    v = v[valid]
    z = z[valid]
    colors_valid = colors_valid[valid]

    if len(u) == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    # 创建深度缓冲区和颜色缓冲区
    depth_buffer = np.full((H, W), np.inf)
    image = np.zeros((H, W, 3), dtype=np.uint8)

    # 按深度排序（远到近）
    sort_idx = np.argsort(-z)
    u = u[sort_idx]
    v = v[sort_idx]
    z = z[sort_idx]
    colors_valid = colors_valid[sort_idx]

    # 渲染（近的点覆盖远的点）
    for i in range(len(u)):
        if z[i] < depth_buffer[v[i], u[i]]:
            depth_buffer[v[i], u[i]] = z[i]
            image[v[i], u[i]] = colors_valid[i]

    return image


def update_dynamic_object_depth(
    image1_path: str,
    image2_path: str,
    dynamic_object_prompts: list,
    output_dir: str,
    sam3_model_path: str = "ckpts/facebook--sam3/sam3.pt",
    device: str = "cuda",
):
    """
    主函数：将第二帧中动态物体的点云更新到第一帧的点云中

    Args:
        image1_path: 第一张图片路径（原始帧）
        image2_path: 第二张图片路径（动态物体移动后）
        dynamic_object_prompts: 动态物体名称列表，如 ["car", "person"]
        output_dir: 输出目录
        sam3_model_path: SAM3 模型路径
        device: 计算设备
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Dynamic Object Depth Update")
    print("=" * 60)
    print(f"Image 1: {image1_path}")
    print(f"Image 2: {image2_path}")
    print(f"Dynamic objects: {dynamic_object_prompts}")
    print(f"Output: {output_dir}")
    print()

    # ========== Step 1: 加载图片和模型 ==========
    print("[Step 1/6] Loading images and models...")
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    assert image1.shape == image2.shape, f"Image shapes must match: {image1.shape} vs {image2.shape}"
    H, W = image1.shape[:2]
    print(f"  Image size: {W}x{H}")

    # 保存输入图片
    cv2.imwrite(str(output_dir / "input_image1.png"), cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "input_image2.png"), cv2.cvtColor(image2, cv2.COLOR_RGB2BGR))

    print("  Loading STream3R model...")
    stream3r_model = load_stream3r_model(device)

    print("  Loading SAM3 segmenter...")
    sam3_segmenter = load_sam3_segmenter(sam3_model_path)

    # ========== Step 2: 估计深度 ==========
    print("\n[Step 2/6] Estimating depth with STream3R...")

    print("  Processing image 1...")
    depth1, intrinsics1, pts3d_1, c2w_1, conf1 = estimate_depth_and_intrinsics(image1, stream3r_model, device)
    print(f"    Depth range: [{depth1[depth1 > 0].min():.2f}, {depth1.max():.2f}]")

    print("  Processing image 2...")
    depth2, intrinsics2, pts3d_2, c2w_2, conf2 = estimate_depth_and_intrinsics(image2, stream3r_model, device)
    print(f"    Depth range: [{depth2[depth2 > 0].min():.2f}, {depth2.max():.2f}]")

    # 保存深度可视化
    save_depth_visualization(depth1, str(output_dir / "depth1.png"))
    save_depth_visualization(depth2, str(output_dir / "depth2.png"))

    # 释放 STream3R 模型
    del stream3r_model
    torch.cuda.empty_cache()

    # ========== Step 3: 分割天空 ==========
    print("\n[Step 3/6] Segmenting sky with SAM3...")
    sky_mask1 = segment_with_sam3(image1, ["sky"], sam3_segmenter)
    sky_mask2 = segment_with_sam3(image2, ["sky"], sam3_segmenter)
    print(f"  Sky pixels in image 1: {sky_mask1.sum()} ({sky_mask1.sum() / sky_mask1.size * 100:.1f}%)")
    print(f"  Sky pixels in image 2: {sky_mask2.sum()} ({sky_mask2.sum() / sky_mask2.size * 100:.1f}%)")

    # ========== Step 4: 分割动态物体 ==========
    print("\n[Step 4/6] Segmenting dynamic objects with SAM3...")
    foreground_mask1 = segment_with_sam3(image1, dynamic_object_prompts, sam3_segmenter)
    foreground_mask2 = segment_with_sam3(image2, dynamic_object_prompts, sam3_segmenter)
    print(f"  Foreground pixels in image 1: {foreground_mask1.sum()} ({foreground_mask1.sum() / foreground_mask1.size * 100:.1f}%)")
    print(f"  Foreground pixels in image 2: {foreground_mask2.sum()} ({foreground_mask2.sum() / foreground_mask2.size * 100:.1f}%)")

    # 释放 SAM3 分割器
    del sam3_segmenter
    torch.cuda.empty_cache()

    # 保存分割可视化
    def save_mask_vis(image, mask, path, color=[255, 0, 0]):
        vis = image.copy()
        overlay = image.copy()
        overlay[mask] = color
        vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0)
        cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    save_mask_vis(image1, foreground_mask1, str(output_dir / "foreground_mask1.png"))
    save_mask_vis(image2, foreground_mask2, str(output_dir / "foreground_mask2.png"))
    save_mask_vis(image1, sky_mask1, str(output_dir / "sky_mask1.png"), color=[0, 0, 255])
    save_mask_vis(image2, sky_mask2, str(output_dir / "sky_mask2.png"), color=[0, 0, 255])

    # 背景 mask = 非前景 且 非天空
    background_mask1 = (~foreground_mask1) & (~sky_mask1)
    background_mask2 = (~foreground_mask2) & (~sky_mask2)

    # ========== Step 4.5: 调整 mask 尺寸以匹配深度图 ==========
    # 深度图尺寸可能与原图不同，需要 resize masks
    depth_H1, depth_W1 = depth1.shape
    depth_H2, depth_W2 = depth2.shape

    if (depth_H1, depth_W1) != (H, W):
        print(f"\n[Step 4.5] Resizing masks to match depth size ({depth_W1}x{depth_H1})...")
        sky_mask1 = cv2.resize(sky_mask1.astype(np.uint8), (depth_W1, depth_H1), interpolation=cv2.INTER_NEAREST).astype(bool)
        foreground_mask1 = cv2.resize(foreground_mask1.astype(np.uint8), (depth_W1, depth_H1), interpolation=cv2.INTER_NEAREST).astype(bool)
        background_mask1 = cv2.resize(background_mask1.astype(np.uint8), (depth_W1, depth_H1), interpolation=cv2.INTER_NEAREST).astype(bool)
        # 也需要 resize image1 用于点云颜色
        image1_resized = cv2.resize(image1, (depth_W1, depth_H1), interpolation=cv2.INTER_LINEAR)
    else:
        image1_resized = image1

    if (depth_H2, depth_W2) != (H, W):
        sky_mask2 = cv2.resize(sky_mask2.astype(np.uint8), (depth_W2, depth_H2), interpolation=cv2.INTER_NEAREST).astype(bool)
        foreground_mask2 = cv2.resize(foreground_mask2.astype(np.uint8), (depth_W2, depth_H2), interpolation=cv2.INTER_NEAREST).astype(bool)
        background_mask2 = cv2.resize(background_mask2.astype(np.uint8), (depth_W2, depth_H2), interpolation=cv2.INTER_NEAREST).astype(bool)
        image2_resized = cv2.resize(image2, (depth_W2, depth_H2), interpolation=cv2.INTER_LINEAR)
    else:
        image2_resized = image2

    # ========== Step 5: 计算深度尺度因子 ==========
    print("\n[Step 5/6] Computing depth scale factor from background...")

    # 用两张图的背景区域交集来计算 scale
    common_background = background_mask1 & background_mask2
    scale = compute_scale_factor(depth1, depth2, common_background)

    # 对第二张图的深度进行 scale 对齐
    depth2_scaled = depth2 * scale
    print(f"  Scaled depth2 range: [{depth2_scaled[depth2_scaled > 0].min():.2f}, {depth2_scaled.max():.2f}]")

    # ========== Step 6: 构建并合并点云 ==========
    print("\n[Step 6/6] Building and merging point clouds...")

    # 第一帧：背景点云（排除天空和前景）
    # 我们用第一帧的背景作为基础
    points_bg1, colors_bg1 = unproject_to_pointcloud(
        depth1, image1_resized, intrinsics1, mask=background_mask1
    )
    print(f"  Image 1 background points: {len(points_bg1)}")

    # 第一帧：前景点云
    points_fg1, colors_fg1 = unproject_to_pointcloud(
        depth1, image1_resized, intrinsics1, mask=foreground_mask1
    )
    print(f"  Image 1 foreground points: {len(points_fg1)}")

    # 第二帧：前景点云（使用 scaled depth）
    # 注意：我们用 depth2_scaled 和 intrinsics1（因为相机参数应该一致）
    # 实际上由于模型可能估计出略微不同的内参，我们用 intrinsics2 然后做变换
    points_fg2, colors_fg2 = unproject_to_pointcloud(
        depth2_scaled, image2_resized, intrinsics1, mask=foreground_mask2  # 使用 intrinsics1 保持一致
    )
    print(f"  Image 2 foreground points (scaled): {len(points_fg2)}")

    # 合并点云：第一帧背景 + 第二帧前景
    # 这样就把运动后的前景物体放到了第一帧的场景中
    merged_points = np.concatenate([points_bg1, points_fg2], axis=0)
    merged_colors = np.concatenate([colors_bg1, colors_fg2], axis=0)
    print(f"  Merged point cloud: {len(merged_points)} points")

    # 也保存完整的第一帧点云（背景+原始前景）作为对比
    complete_points1 = np.concatenate([points_bg1, points_fg1], axis=0)
    complete_colors1 = np.concatenate([colors_bg1, colors_fg1], axis=0)

    # ========== 保存结果 ==========
    print("\n[Saving results...]")

    # 保存点云
    save_pointcloud_ply(str(output_dir / "pointcloud_image1_complete.ply"), complete_points1, complete_colors1)
    save_pointcloud_ply(str(output_dir / "pointcloud_image1_background.ply"), points_bg1, colors_bg1)
    save_pointcloud_ply(str(output_dir / "pointcloud_image1_foreground.ply"), points_fg1, colors_fg1)
    save_pointcloud_ply(str(output_dir / "pointcloud_image2_foreground_scaled.ply"), points_fg2, colors_fg2)
    save_pointcloud_ply(str(output_dir / "pointcloud_merged.ply"), merged_points, merged_colors)

    # 保存元数据
    metadata = {
        "image1_path": str(image1_path),
        "image2_path": str(image2_path),
        "dynamic_object_prompts": dynamic_object_prompts,
        "scale_factor": float(scale),
        "image_size": [W, H],
        "depth1_range": [float(depth1[depth1 > 0].min()), float(depth1.max())],
        "depth2_range": [float(depth2[depth2 > 0].min()), float(depth2.max())],
        "depth2_scaled_range": [float(depth2_scaled[depth2_scaled > 0].min()), float(depth2_scaled.max())],
        "num_points": {
            "image1_background": len(points_bg1),
            "image1_foreground": len(points_fg1),
            "image2_foreground_scaled": len(points_fg2),
            "merged": len(merged_points),
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 保存深度数据
    np.savez(
        output_dir / "depth_data.npz",
        depth1=depth1,
        depth2=depth2,
        depth2_scaled=depth2_scaled,
        intrinsics1=intrinsics1,
        intrinsics2=intrinsics2,
        c2w_1=c2w_1,
        c2w_2=c2w_2,
        foreground_mask1=foreground_mask1,
        foreground_mask2=foreground_mask2,
        background_mask1=background_mask1,
        background_mask2=background_mask2,
        sky_mask1=sky_mask1,
        sky_mask2=sky_mask2,
        scale_factor=scale,
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"  - pointcloud_image1_complete.ply: 第一帧完整点云")
    print(f"  - pointcloud_image1_background.ply: 第一帧背景点云")
    print(f"  - pointcloud_image1_foreground.ply: 第一帧前景点云")
    print(f"  - pointcloud_image2_foreground_scaled.ply: 第二帧前景点云（scaled）")
    print(f"  - pointcloud_merged.ply: 合并后的点云（第一帧背景 + 第二帧前景）")
    print(f"  - depth1.png, depth2.png: 深度可视化")
    print(f"  - foreground_mask1.png, foreground_mask2.png: 前景分割可视化")
    print(f"  - metadata.json: 元数据")
    print(f"  - depth_data.npz: 深度数据")
    print(f"\nScale factor: {scale:.4f}")
    print(f"Merged points: {len(merged_points)}")

    return {
        "merged_points": merged_points,
        "merged_colors": merged_colors,
        "scale_factor": scale,
        "metadata": metadata,
    }


def merge_video_foregrounds(
    video_path: str,
    dynamic_object_prompts: list,
    output_dir: str,
    sam3_model_path: str = "ckpts/facebook--sam3/sam3.pt",
    rotation_angle: float = 45.0,
    depth_outlier_std_scale: float = 3.0,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    device: str = "cuda",
    verbose: bool = True,
    stream3r_model=None,
    sam3_segmenter=None,
    stream3r_session=None,
    stream3r_frames_fed: int = 0,
    video_frames=None,
    video_fps: float = None,
    stride: int = 1,
    save_depth_maps: bool = True,
    save_rendered_frames: bool = True,
    save_rendered_video: bool = True,
    save_final_merged_pointcloud: bool = True,
    fg_mask_erode: int = 0,
    skip_background: bool = False,
):
    """
    视频模式：把视频每帧的前景物体合并到第一帧的点云中，并从偏移视角渲染

    Args:
        video_path: 输入视频路径（当 video_frames 为空时必需）
        video_frames: 可选，直接提供 RGB 帧序列，跳过 mp4 读取
        video_fps: 可选，video_frames 对应帧率
        dynamic_object_prompts: 前景物体名称列表
        output_dir: 输出目录
        sam3_model_path: SAM3 模型路径
        rotation_angle: 相机偏移角度（度）
        depth_outlier_std_scale: 深度离群点过滤的标准差倍数 (default: 3.0)
        conf_threshold: 置信度阈值（必须在合法范围内）
        device: 计算设备
        save_depth_maps: 是否保存每帧深度可视化图
        save_rendered_frames: 是否保存渲染帧 PNG 序列
        save_rendered_video: 是否保存 rendered_merged.mp4
        save_final_merged_pointcloud: 是否保存 final_merged_pointcloud.ply
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log = print if verbose else (lambda *a, **k: None)

    _log("=" * 60)
    _log("Merge Video Foregrounds")
    _log("=" * 60)
    _log(f"Dynamic objects: {dynamic_object_prompts}")
    _log(f"Rotation angle: {rotation_angle}°")
    _log(f"Output: {output_dir}")
    _log()

    effective_conf_threshold = _normalize_conf_threshold(conf_threshold, verbose=verbose)
    _log(f"  Effective confidence threshold: {effective_conf_threshold:.3f}")
    if fg_mask_erode > 0:
        _log("  Note: fg_mask_erode is ignored; foreground points are kept without erosion.")

    # ========== Step 1: 加载输入帧和模型 ==========
    _log("[Step 1] Loading video and models...")
    if video_frames is not None:
        if isinstance(video_frames, np.ndarray):
            frames = [f for f in video_frames]
        else:
            frames = list(video_frames)
        if len(frames) == 0:
            raise ValueError("video_frames is empty")
        if video_fps is not None and np.isfinite(video_fps) and float(video_fps) > 0:
            fps = float(video_fps)
        else:
            fps = 16.0
        source_desc = "in-memory frames"
    else:
        if not video_path:
            raise ValueError("video_path must be provided when video_frames is None")
        frames = load_video_frames(video_path)
        if len(frames) == 0:
            raise ValueError(f"No frames found in video: {video_path}")
        fps = get_video_fps(video_path)
        source_desc = str(video_path)

    H, W = frames[0].shape[:2]

    # Subsample frames if stride > 1 (reduces STream3R + SAM3 cost).
    # Always keep the first and last frame.
    stride = max(1, int(stride))
    if stride > 1:
        n_orig = len(frames)
        indices = list(range(0, n_orig, stride))
        if indices[-1] != n_orig - 1:
            indices.append(n_orig - 1)
        frames = [frames[i] for i in indices]
        _log(f"  Stride={stride}: subsampled {len(frames)}/{n_orig} frames (first+last guaranteed)")

    num_frames = len(frames)
    _log(f"  Source: {source_desc}")
    _log(f"  Frames: {num_frames}, FPS: {fps:.2f}, Size: {W}x{H}")

    _owns_stream3r = stream3r_model is None
    if _owns_stream3r:
        _log("  Loading STream3R model...")
        stream3r_model = load_stream3r_model(device)
    else:
        _log("  Reusing shared STream3R model")

    _owns_sam3 = sam3_segmenter is None
    if _owns_sam3:
        _log("  Loading SAM3 segmenter...")
        sam3_segmenter = load_sam3_segmenter(sam3_model_path)
    else:
        _log("  Reusing shared SAM3 segmenter")

    # ========== Step 2: 用 StreamSession 估计所有帧的深度 ==========
    _log("\n[Step 2] Estimating depth for all frames (session mode)...")

    depth_results, stream3r_frames_fed = estimate_all_depths_with_session(
        frames, stream3r_model, device=device, verbose=verbose,
        session=stream3r_session,
        initial_frames_fed=stream3r_frames_fed,
    )
    _log(f"  Got depth for {len(depth_results)} frames via session (total fed: {stream3r_frames_fed})")

    # ========== Step 3: 逐帧分割 + 点云提取 ==========
    _log(f"\n[Step 3] Segmenting and extracting point clouds...")

    # 创建深度图保存目录（可选）
    depths_dir = output_dir / "depths"
    if save_depth_maps:
        depths_dir.mkdir(exist_ok=True)

    # -- Per-instance temporal segmentation --
    # Call segment_instances_video() once on the full video to get per-instance
    # masks across all frames, enabling per-instance point cloud extraction.
    instance_masks_dict = {}  # {obj_id: (T, H, W) bool}
    _has_instance_seg = hasattr(sam3_segmenter, "segment_instances_video")
    if not _has_instance_seg:
        # LiveWorld segmenter: use segment_instances() which has compatible return type.
        _has_instance_seg = hasattr(sam3_segmenter, "segment_instances")
    if _has_instance_seg and dynamic_object_prompts:
        _log("  Running per-instance temporal segmentation...")
        pil_frames = [Image.fromarray(f) for f in frames]
        if hasattr(sam3_segmenter, "segment_instances_video"):
            instance_masks_dict = sam3_segmenter.segment_instances_video(
                video_path=pil_frames,
                prompts=dynamic_object_prompts,
                frame_index=0,
                expected_frames=num_frames,
            )
        else:
            instance_masks_dict = sam3_segmenter.segment_instances(
                video_path=pil_frames,
                prompts=dynamic_object_prompts,
                frame_index=0,
                expected_frames=num_frames,
            )
        _log(f"  Per-instance segmentation: {len(instance_masks_dict)} instances detected")
        for obj_id, masks_arr in instance_masks_dict.items():
            _log(f"    obj_{obj_id}: shape={masks_arr.shape}, "
                 f"pixels_frame0={masks_arr[0].sum()}")

    # -- Rescale depth maps to video resolution and store per frame --
    depth1_raw, intrinsics1_raw, _, conf1_raw = depth_results[0]
    depth_H_raw, depth_W_raw = depth1_raw.shape
    _log(f"  Raw depth size: {depth_W_raw}x{depth_H_raw}")

    # 把深度、置信度和内参 rescale 到原视频分辨率
    if (depth_H_raw, depth_W_raw) != (H, W):
        _log(f"  Rescaling depth and intrinsics to original size ({W}x{H})...")
        depth1 = cv2.resize(depth1_raw, (W, H), interpolation=cv2.INTER_LINEAR)
        conf1 = cv2.resize(conf1_raw, (W, H), interpolation=cv2.INTER_LINEAR)
        scale_x = W / depth_W_raw
        scale_y = H / depth_H_raw
        intrinsics1 = intrinsics1_raw.copy()
        intrinsics1[0, 0] *= scale_x
        intrinsics1[1, 1] *= scale_y
        intrinsics1[0, 2] *= scale_x
        intrinsics1[1, 2] *= scale_y
    else:
        depth1 = depth1_raw
        intrinsics1 = intrinsics1_raw
        conf1 = conf1_raw

    # Store per-frame depth for per-instance point cloud extraction later.
    depth_maps = [depth1]

    if save_depth_maps:
        save_depth_visualization(depth1, str(depths_dir / "depth_0000.png"))

    # Build foreground mask from per-instance masks or fallback to per-frame SAM3.
    if instance_masks_dict:
        foreground_mask1 = np.zeros((H, W), dtype=bool)
        for obj_id, inst_masks in instance_masks_dict.items():
            foreground_mask1 |= inst_masks[0]
        _log(f"  Foreground mask (from {len(instance_masks_dict)} instances): "
             f"{foreground_mask1.sum()} pixels")
    else:
        _log("  Segmenting foreground...")
        foreground_mask1 = segment_with_sam3(frames[0], dynamic_object_prompts, sam3_segmenter)

    # Background point cloud (only needed for standalone rendering/merged ply).
    points_bg1 = None
    colors_bg1 = None
    if not skip_background:
        _log("  Segmenting sky...")
        sky_mask1 = segment_with_sam3(frames[0], ["sky"], sam3_segmenter)
        background_mask1 = (~foreground_mask1) & (~sky_mask1)
        points_bg1, colors_bg1 = unproject_to_pointcloud(
            depth1, frames[0], intrinsics1, mask=background_mask1, conf=conf1,
            conf_threshold=effective_conf_threshold
        )
        _log(f"  Background points: {len(points_bg1)}")

    points_fg1, colors_fg1 = unproject_to_pointcloud(
        # Foreground points keep all valid masked depths.
        # No confidence filter, no fg-mask erosion.
        depth1, frames[0], intrinsics1, mask=foreground_mask1, conf=None,
        conf_threshold=effective_conf_threshold, filter_depth_outliers=True,
        depth_outlier_std_scale=depth_outlier_std_scale
    )
    _log(f"  Foreground points: {len(points_fg1)}")

    # -- 处理后续帧 --
    all_foreground_points = [points_fg1]
    all_foreground_colors = [colors_fg1]

    for frame_idx in tqdm(range(1, num_frames), desc="Processing frames", disable=not verbose):
        frame = frames[frame_idx]
        depth_i_raw, _, _, _ = depth_results[frame_idx]
        depth_H_i_raw, depth_W_i_raw = depth_i_raw.shape

        if (depth_H_i_raw, depth_W_i_raw) != (H, W):
            depth_i = cv2.resize(depth_i_raw, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            depth_i = depth_i_raw

        depth_maps.append(depth_i)

        if save_depth_maps:
            save_depth_visualization(depth_i, str(depths_dir / f"depth_{frame_idx:04d}.png"))

        # Build fg mask from per-instance masks or fallback to per-frame SAM3.
        if instance_masks_dict:
            fg_mask_i = np.zeros((H, W), dtype=bool)
            for obj_id, inst_masks in instance_masks_dict.items():
                fg_mask_i |= inst_masks[frame_idx]
        else:
            fg_mask_i = segment_with_sam3(frame, dynamic_object_prompts, sam3_segmenter)

        depth_i_scaled = depth_i * 1

        points_fg_i, colors_fg_i = unproject_to_pointcloud(
            # Foreground points keep all valid masked depths.
            # No confidence filter, no fg-mask erosion.
            depth_i_scaled, frame, intrinsics1, mask=fg_mask_i, conf=None,
            conf_threshold=effective_conf_threshold, filter_depth_outliers=True,
            depth_outlier_std_scale=depth_outlier_std_scale
        )

        all_foreground_points.append(points_fg_i)
        all_foreground_colors.append(colors_fg_i)

    _log(f"  Total foreground point sets: {len(all_foreground_points)}")

    # -- Build per-instance per-frame point clouds --
    per_instance_per_frame_points = {}
    per_instance_per_frame_colors = {}
    per_instance_anchor_masks = {}
    if instance_masks_dict:
        _log("  Building per-instance point clouds...")
        for obj_id, inst_masks_TxHxW in instance_masks_dict.items():
            inst_pts_list = []
            inst_cols_list = []
            for t in range(num_frames):
                inst_mask_t = inst_masks_TxHxW[t]
                if not inst_mask_t.any():
                    inst_pts_list.append(np.zeros((0, 3), dtype=np.float32))
                    inst_cols_list.append(np.zeros((0, 3), dtype=np.uint8))
                    continue
                depth_t = depth_maps[t] * 1  # same scale as merged
                pts, cols = unproject_to_pointcloud(
                    depth_t, frames[t], intrinsics1, mask=inst_mask_t,
                    # Per-instance foreground points keep all valid masked depths.
                    # No confidence filter, no fg-mask erosion.
                    conf=None,
                    conf_threshold=effective_conf_threshold,
                    filter_depth_outliers=True,
                    depth_outlier_std_scale=depth_outlier_std_scale,
                )
                inst_pts_list.append(pts)
                inst_cols_list.append(cols)
            per_instance_per_frame_points[obj_id] = inst_pts_list
            per_instance_per_frame_colors[obj_id] = inst_cols_list
            # Store anchor mask (frame 0) for instance reference extraction.
            anchor_mask = inst_masks_TxHxW[0]
            if anchor_mask.any():
                per_instance_anchor_masks[obj_id] = anchor_mask
            total_pts = sum(len(p) for p in inst_pts_list)
            _log(f"    obj_{obj_id}: {total_pts} total points across {num_frames} frames")

    # 释放模型显存 (only if we loaded them ourselves)
    if _owns_stream3r:
        del stream3r_model
    if _owns_sam3:
        del sam3_segmenter
    torch.cuda.empty_cache()

    # ========== Step 4: 渲染每一帧（可选） ==========
    need_rendering = save_rendered_video or save_rendered_frames
    rendered_frames = []

    if need_rendering:
        _log("\n[Step 4] Rendering frames...")

        # 如果 rotation_angle == 0，直接用原视角；否则从上方俯视
        if abs(rotation_angle) < 1e-6:
            # 原视角：点云本身就在相机坐标系下，不需要变换
            c2w_render = None
            _log(f"  Using original viewpoint (no rotation)")
        else:
            # 计算点云中心（用于确定观察目标）
            bg_parts = [points_bg1] if points_bg1 is not None else []
            all_pts = np.concatenate(bg_parts + all_foreground_points, axis=0)
            center = all_pts.mean(axis=0)
            _log(f"  Point cloud center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")

            # 从上方俯视：相机在场景上方，往下看
            # rotation_angle 这里表示俯视角度（60度 = 从上方60度往下看）
            elevation_angle = np.radians(rotation_angle)  # 仰角（从水平面往上的角度）

            # 相机距离场景中心的距离（不要太远）
            cam_distance = center[2] * 2  # 用较近的距离

            # 相机位置：在场景中心的后上方
            # x 不变，y 往上移动，z 往后移动（靠近相机原点）
            cam_pos = np.array([
                center[0],  # x 保持不变
                center[1] - cam_distance * np.sin(elevation_angle),  # y 往上（相机坐标系 y 向下，所以减）
                center[2] - cam_distance * np.cos(elevation_angle),  # z 往后（靠近相机原点）
            ])

            # 构建相机到世界的变换矩阵
            # 相机看向场景中心
            forward = center - cam_pos
            forward = forward / np.linalg.norm(forward)

            # 定义世界坐标系的上方向（相机坐标系 y 向下，所以用 -y）
            world_up = np.array([0, -1, 0])

            # 计算 right 和 up
            right = np.cross(forward, world_up)
            if np.linalg.norm(right) < 1e-6:
                # forward 和 world_up 平行，用 x 轴作为 right
                right = np.array([1, 0, 0])
            else:
                right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)

            # 构建 c2w 矩阵（相机坐标系：x=right, y=down, z=forward）
            c2w_render = np.eye(4)
            c2w_render[:3, 0] = right
            c2w_render[:3, 1] = -up  # 相机 y 轴向下
            c2w_render[:3, 2] = forward
            c2w_render[:3, 3] = cam_pos

            _log(f"  Camera position: [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]")
            _log(f"  Elevation angle: {rotation_angle}°")

        for frame_idx in tqdm(range(num_frames), desc="Rendering", disable=not verbose):
            # 当前帧的前景（不累积）
            current_fg_points = all_foreground_points[frame_idx]
            current_fg_colors = all_foreground_colors[frame_idx]

            # 合并背景和当前帧前景
            if points_bg1 is not None:
                merged_points = np.concatenate([points_bg1, current_fg_points], axis=0)
                merged_colors = np.concatenate([colors_bg1, current_fg_colors], axis=0)
            else:
                merged_points = current_fg_points
                merged_colors = current_fg_colors

            # 渲染（使用原视频分辨率）
            rendered = render_pointcloud(
                merged_points,
                merged_colors,
                intrinsics1,
                (H, W),
                camera_pose=c2w_render,
            )

            rendered_frames.append(rendered)
    else:
        _log("\n[Step 4] Skipping rendering (render outputs disabled)")

    # ========== Step 5: 保存输出 ==========
    _log("\n[Step 5] Saving outputs...")

    # 保存渲染视频（可选）
    if save_rendered_video:
        output_video_path = output_dir / "rendered_merged.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (W, H))

        for frame in rendered_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        _log(f"  Saved: {output_video_path}")

    # 保存最终合并的点云（可选）
    final_fg_points = np.concatenate(all_foreground_points, axis=0)
    final_fg_colors = np.concatenate(all_foreground_colors, axis=0)
    if points_bg1 is not None:
        final_merged_points = np.concatenate([points_bg1, final_fg_points], axis=0)
        final_merged_colors = np.concatenate([colors_bg1, final_fg_colors], axis=0)
    else:
        final_merged_points = final_fg_points
        final_merged_colors = final_fg_colors

    if save_final_merged_pointcloud:
        save_pointcloud_ply(str(output_dir / "final_merged_pointcloud.ply"), final_merged_points, final_merged_colors)
        _log(f"  Saved: {output_dir / 'final_merged_pointcloud.ply'}")

    num_bg = len(points_bg1) if points_bg1 is not None else 0

    # 保存元数据
    metadata = {
        "video_path": str(video_path) if video_path is not None else "<in_memory>",
        "video_source": source_desc,
        "num_frames": num_frames,
        "fps": fps,
        "image_size": [W, H],
        "raw_depth_size": [depth_W_raw, depth_H_raw],
        "rotation_angle": rotation_angle,
        "conf_threshold": float(effective_conf_threshold),
        "dynamic_object_prompts": dynamic_object_prompts,
        "num_background_points": num_bg,
        "num_foreground_points_per_frame": [len(pts) for pts in all_foreground_points],
        "total_merged_points": len(final_merged_points),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    _log(f"  Saved: {output_dir / 'metadata.json'}")

    # 保存每帧渲染结果（可选）
    if save_rendered_frames:
        frames_dir = output_dir / "rendered_frames"
        frames_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(rendered_frames):
            cv2.imwrite(str(frames_dir / f"frame_{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        _log(f"  Saved frames to: {frames_dir}")

    if save_depth_maps:
        _log(f"  Saved depth maps to: {depths_dir}")

    _log("\n" + "=" * 60)
    _log("Done!")
    _log("=" * 60)

    return {
        "rendered_frames": rendered_frames,
        "final_merged_points": final_merged_points,
        "final_merged_colors": final_merged_colors,
        # Foreground-only (no background) for event projection.
        "final_fg_points": final_fg_points,
        "final_fg_colors": final_fg_colors,
        # Per-frame foreground points for finer control.
        "all_foreground_points": all_foreground_points,
        "all_foreground_colors": all_foreground_colors,
        "metadata": metadata,
        # Session state for caller to sync back.
        "stream3r_frames_fed": stream3r_frames_fed,
        # Per-instance per-frame point clouds (empty dicts when instance seg unavailable).
        "per_instance_per_frame_points": per_instance_per_frame_points,
        "per_instance_per_frame_colors": per_instance_per_frame_colors,
        "per_instance_anchor_masks": per_instance_anchor_masks,
    }


def main(
    mode: str = "video",
    # 通用配置
    dynamic_object_prompts: list = None,
    sam3_model_path: str = "ckpts/facebook--sam3/sam3.pt",
    device: str = "cuda",
    # 图片模式配置
    image1_path: str = None,
    image2_path: str = None,
    # 视频模式配置
    video_path: str = None,
    output_dir: str = None,
    rotation_angle: float = 20.0,
    depth_outlier_std_scale: float = 3.0,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
):
    """
    主函数入口

    Args:
        mode: "image" 或 "video"
        dynamic_object_prompts: 动态物体类别名称列表
        sam3_model_path: SAM3 模型路径
        device: 计算设备
        image1_path: 图片模式 - 第一张图片路径
        image2_path: 图片模式 - 第二张图片路径
        video_path: 视频模式 - 视频路径
        output_dir: 输出目录
        rotation_angle: 视频模式 - 相机俯视角度
        depth_outlier_std_scale: 深度离群点过滤的标准差倍数
        conf_threshold: 置信度阈值 (default: 0.10，超范围会直接报错)
    """
    if dynamic_object_prompts is None:
        dynamic_object_prompts = ["car"]

    if mode == "image":
        if output_dir is None:
            output_dir = "outputs/dynamic_depth_update"

        result = update_dynamic_object_depth(
            image1_path=image1_path,
            image2_path=image2_path,
            dynamic_object_prompts=dynamic_object_prompts,
            output_dir=output_dir,
            sam3_model_path=sam3_model_path,
            device=device,
        )

    elif mode == "video":
        if output_dir is None:
            output_dir = "outputs/merge_video_foregrounds"

        result = merge_video_foregrounds(
            video_path=video_path,
            dynamic_object_prompts=dynamic_object_prompts,
            output_dir=output_dir,
            sam3_model_path=sam3_model_path,
            rotation_angle=rotation_angle,
            depth_outlier_std_scale=depth_outlier_std_scale,
            conf_threshold=conf_threshold,
            device=device,
        )

    return result


if __name__ == "__main__":
    # ========== 超参数配置 ==========

    # 模式选择: "image" 或 "video"
    mode = "video"

    # 动态物体类别名称（用于 SAM3 分割）
    dynamic_object_prompts = ["car"]  # 例如: ["car", "person", "dog"]

    # SAM3 模型路径
    sam3_model_path = "ckpts/facebook--sam3/sam3.pt"
    
    # 计算设备
    device = "cuda"

    # 深度离群点过滤的标准差倍数 (mean ± std_scale * std)
    depth_outlier_std_scale = 3.0

    # 置信度阈值 (0.0 = 不过滤, 越大过滤越严格)
    conf_threshold = 0.4

    # ========== 图片模式配置 ==========
    image1_path = "example_imgs/first_frame/car_wild_far.png"
    image2_path = "example_imgs/first_frame/car_wild_near.png"
    image_output_dir = "outputs/dynamic_depth_update"

    # ========== 视频模式配置 ==========
    video_path = "example_imgs/videos/name_car_wild_far_step_unknown_idx_0.mp4"
    video_output_dir = "outputs/event_video_proj"
    rotation_angle = 20  # 从上方俯视角度（度）
    # rotation_angle = 0  # 从上方俯视角度（度）

    # ========== 运行 ==========
    if mode == "image":
        result = main(
            mode=mode,
            dynamic_object_prompts=dynamic_object_prompts,
            sam3_model_path=sam3_model_path,
            device=device,
            image1_path=image1_path,
            image2_path=image2_path,
            output_dir=image_output_dir,
        )
    else:
        result = main(
            mode=mode,
            dynamic_object_prompts=dynamic_object_prompts,
            sam3_model_path=sam3_model_path,
            device=device,
            video_path=video_path,
            output_dir=video_output_dir,
            rotation_angle=rotation_angle,
            depth_outlier_std_scale=depth_outlier_std_scale,
            conf_threshold=conf_threshold,
        )
