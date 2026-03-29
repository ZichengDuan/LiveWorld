from __future__ import annotations
"""
Stream3R 几何估计模块

使用 Stream3R 模型从视频帧估计:
- 深度图 (depth)
- 相机位姿 (extrinsics, w2c -> c2w)
- 相机内参 (intrinsics)

替代 MapAnything，提供更稳定的几何估计。
"""

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from scripts.create_train_data._utils import VideoGeometry
from stream3r.models.stream3r import STream3R
from stream3r.stream_session import StreamSession
from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri



def w2c_to_c2w(extrinsics: np.ndarray) -> np.ndarray:
    """
    将 w2c [N, 3, 4] 转换为 c2w [N, 4, 4]

    w2c 输出的是 world-to-camera (w2c) 变换矩阵
    我们需要转换为 camera-to-world (c2w) 以保持与 MapAnything 输出一致
    """
    N = extrinsics.shape[0]
    if extrinsics.shape[1] == 3:
        # [N, 3, 4] -> [N, 4, 4]
        w2c_4x4 = np.zeros((N, 4, 4), dtype=extrinsics.dtype)
        w2c_4x4[:, :3, :4] = extrinsics
        w2c_4x4[:, 3, 3] = 1.0
    else:
        w2c_4x4 = extrinsics
    c2w = np.linalg.inv(w2c_4x4)
    return c2w


# === geometry estimator ===

"""
Stream3R 几何估计模块

使用 STream3R 模型从视频帧估计:
- 深度图 (原生 depth head 输出)
- 相机位姿 (w2c -> c2w)
- 相机内参 (intrinsics)

流程:
1. Stream3R 在处理分辨率 (proc_h x proc_w) 下推理得到 depth + pose + intrinsics
2. 上采样深度图到原始分辨率 (original_h x original_w)
3. Scale intrinsics 到原始分辨率
4. 返回原始分辨率的 VideoGeometry (下游直接用于反投影得到致密点云)

使用Stream3R进行几何估计。
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image




def _preprocess_frames(
    frames: np.ndarray,
    mode: str = "crop",
    target_size: int = 518,
) -> torch.Tensor:
    """Preprocess numpy frames to STream3R input tensor.

    Args:
        frames: [N, H, W, 3] uint8 RGB
        mode: "crop" or "pad"
        target_size: target side length (default 518)

    Returns:
        [S, 3, H_proc, W_proc] float32 tensor in [0, 1]
    """
    if mode not in {"crop", "pad"}:
        raise ValueError(f"preprocess_mode must be 'crop' or 'pad', got: {mode}")

    resampling = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC

    images = []
    for frame in frames:
        frame_np = np.asarray(frame)
        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)

        img = Image.fromarray(frame_np).convert("RGB")
        width, height = img.size

        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:  # crop
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        img = img.resize((new_width, new_height), resampling)
        img_tensor = TF.to_tensor(img)  # [3, new_height, new_width], float [0,1]

        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img_tensor = img_tensor[:, start_y:start_y + target_size, :]

        if mode == "pad":
            h_padding = target_size - img_tensor.shape[1]
            w_padding = target_size - img_tensor.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img_tensor = torch.nn.functional.pad(
                    img_tensor, (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant", value=1.0,
                )

        images.append(img_tensor)

    images = torch.stack(images)  # [S, 3, H_proc, W_proc]
    return images


class Stream3REstimator:
    """STream3R 模型封装，用于视频几何估计"""

    def __init__(self, config):
        self.config = config


        device = torch.device(config.device)
        if config.device != "cpu" and not torch.cuda.is_available():
            device = torch.device("cpu")
        self.device = device

        model_path = config.model_path
        print(f"Loading STream3R from: {model_path}")
        self.model = STream3R.from_pretrained(model_path).to(device)
        self.model.eval()

    def estimate_video_geometry(
        self,
        frames: List[np.ndarray],
        frame_indices: Optional[List[int]] = None,
    ) -> VideoGeometry:
        """
        估计视频的几何信息

        流程:
        1. Stream3R proc 分辨率推理 → depth, pose, intrinsics
        2. 上采样深度图到原始分辨率
        3. Scale intrinsics 到原始分辨率
        4. 返回原始分辨率的 VideoGeometry

        Args:
            frames: RGB 帧列表, 每帧 [H, W, 3] uint8
            frame_indices: 帧索引列表

        Returns:
            VideoGeometry: 包含原始分辨率的深度、位姿、内参等信息
        """
        if not frames:
            raise ValueError("frames is empty")


        original_h, original_w = frames[0].shape[:2]
        frames_np = np.stack(frames, axis=0) if not isinstance(frames, np.ndarray) else frames

        # Preprocess frames
        mode = self.config.preprocess_mode
        target_size = self.config.target_size
        images = _preprocess_frames(frames_np, mode=mode, target_size=target_size)
        images = images.to(self.device)  # [S, 3, H_proc, W_proc]

        proc_h, proc_w = images.shape[2], images.shape[3]

        # Run inference using StreamSession (streaming with KV cache, constant VRAM)
        window_size = getattr(self.config, "window_size", 32)
        session = StreamSession(self.model, mode="window", window_size=window_size)

        S = images.shape[0]
        with torch.no_grad():
            for i in range(S):
                frame = images[i:i+1]  # [1, 3, H, W]
                predictions = session.forward_stream(frame)

        # predictions now contains accumulated results for all frames

        # Extract poses and intrinsics at proc resolution
        # pose_enc: [B, S, 9] -> extrinsic [B, S, 3, 4] (w2c), intrinsic [B, S, 3, 3]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:]
        )
        extrinsic_np = extrinsic[0].cpu().numpy()  # [S, 3, 4] at proc resolution
        intrinsic_np = intrinsic[0].cpu().numpy()   # [S, 3, 3] at proc resolution

        # Extract native depth and confidence from depth head
        depth_raw = predictions["depth"][0].cpu().numpy()  # [S, H_proc, W_proc, 1]
        depths_proc = depth_raw[..., 0]  # [S, H_proc, W_proc]
        depth_conf = predictions.get("depth_conf")
        if depth_conf is not None:
            conf = depth_conf[0].cpu().numpy()  # [S, H_proc, W_proc]
        else:
            conf = np.ones(depths_proc.shape, dtype=np.float32)

        # Free GPU memory early
        del predictions, images
        torch.cuda.empty_cache()

        N = extrinsic_np.shape[0]

        # Step 2: Upsample depth maps to original resolution
        depths_up = np.zeros((N, original_h, original_w), dtype=np.float32)
        conf_up = np.zeros((N, original_h, original_w), dtype=np.float32)
        for i in range(N):
            depths_up[i] = cv2.resize(depths_proc[i], (original_w, original_h),
                                      interpolation=cv2.INTER_LINEAR)
            conf_up[i] = cv2.resize(conf[i], (original_w, original_h),
                                    interpolation=cv2.INTER_LINEAR)

        # Step 3: Scale intrinsics from proc resolution to original resolution
        scale_x = original_w / proc_w
        scale_y = original_h / proc_h
        intrinsics_up = intrinsic_np.copy()
        for i in range(N):
            intrinsics_up[i, 0, 0] *= scale_x  # fx
            intrinsics_up[i, 1, 1] *= scale_y  # fy
            intrinsics_up[i, 0, 2] *= scale_x  # cx
            intrinsics_up[i, 1, 2] *= scale_y  # cy

        # Convert w2c -> c2w (poses are resolution-independent)
        poses_c2w = w2c_to_c2w(extrinsic_np).astype(np.float32)

        # Build masks from confidence and positive depth at original resolution
        conf_threshold = getattr(self.config, "conf_threshold", 0.0)
        masks = (depths_up > 0) & (conf_up > conf_threshold)

        # Frames at original resolution (just use the input frames directly)
        frames_out = frames_np.copy()

        # print(f"  [Stream3R] Geometry estimated: proc {proc_h}x{proc_w} → "
        #       f"output {original_h}x{original_w}")

        return VideoGeometry(
            frames=frames_out,                          # [N, original_h, original_w, 3]
            depths=depths_up.astype(np.float32),        # [N, original_h, original_w]
            intrinsics=intrinsics_up.astype(np.float32),  # [N, 3, 3] at original resolution
            poses_c2w=poses_c2w,                        # [N, 4, 4]
            masks=masks,
            frame_indices=np.array(
                frame_indices if frame_indices else list(range(N)),
                dtype=np.int32,
            ),
            original_size=(original_h, original_w),
            processed_size=(original_h, original_w),    # 返回的已经是原始分辨率
        )
