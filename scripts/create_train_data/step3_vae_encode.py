#!/usr/bin/env python3
"""
VAE 编码脚本

将训练视频编码为 latent 并保存为单个 .pt 文件 (latents.pt)。

支持的视频类型:
- train_preceding_rgb_9.mp4: 9帧 preceding (clip-conditioned mode)
- train_preceding_rgb_1.mp4: 1帧 preceding (image-conditioned mode)
- train_target_rgb.mp4: 目标帧视频
- train_reference_rgb.mp4: 参考帧视频 (逐帧编码，避免时序关联)
- train_reference_scene_rgb.mp4: 参考帧背景/scene视频 (逐帧编码，避免时序关联)
- train_reference_instance_00.mp4: 参考实例视频 (逐帧编码，避免时序关联)
- train_reference_instance_01.mp4: 参考实例视频 (逐帧编码，避免时序关联)
- train_reference_instance_02.mp4: 参考实例视频 (逐帧编码，避免时序关联)
- train_reference_instance_03.mp4: 参考实例视频 (逐帧编码，避免时序关联)
- train_reference_instance_04.mp4: 参考实例视频 (逐帧编码，避免时序关联)
- train_preceding_scene_proj_rgb_9.mp4: 9帧 preceding 场景投影 (clip-conditioned mode)
- train_preceding_scene_proj_rgb_1.mp4: 1帧 preceding 场景投影 (image-conditioned mode)
- train_target_scene_proj_rgb.mp4: 目标帧场景投影
- train_target_scene_proj_rgb_orig.mp4: 目标帧场景投影 (非augmented原版)
- train_reference_scene_rgb_orig.mp4: 参考帧背景/scene视频 (非augmented原版, 逐帧编码)
- train_preceding_proj_fg_rgb_9.mp4: 9帧 preceding 前景投影 (黑底)
- train_preceding_proj_fg_rgb_1.mp4: 1帧 preceding 前景投影 (黑底)
- train_target_proj_fg_rgb.mp4: 目标帧前景投影 (黑底)

根据 LiveWorld 论文:
- 第一轮 (image-conditioned): 使用 1帧 preceding + 1帧 preceding scene proj
- 后续轮 (clip-conditioned): 使用 9帧 preceding + 9帧 preceding scene proj
训练时随机选择使用哪种 preceding 模式。

输出格式:
- 所有 latent 保存在单个文件 latents.pt 中
- 格式: {"train_preceding_rgb_9": tensor, "train_target_rgb": tensor, ...}

用法:
    python -m scripts.dataset_preparation.step3_vae_encode \\
        --input-root data/liveworld/frame33_fps16_2000 \\
        --video-keys "train_preceding_rgb_9,train_preceding_rgb_1,train_target_rgb,train_reference_rgb,train_reference_scene_rgb,train_reference_instance_00,train_reference_instance_01,train_reference_instance_02,train_reference_instance_03,train_reference_instance_04,train_preceding_scene_proj_rgb_9,train_preceding_scene_proj_rgb_1,train_target_scene_proj_rgb,train_preceding_proj_fg_rgb_9,train_preceding_proj_fg_rgb_1,train_target_proj_fg_rgb"
"""
from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import datetime

from omegaconf import OmegaConf
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG = OmegaConf.load(str(_PROJECT_ROOT / "configs" / "data_preparation.yaml"))
from scripts.create_train_data._utils import get_rank_info, shard_items
from liveworld.wrapper import WanVAEWrapper


LATENTS_FILENAME = "latents.pt"


def get_vae_wrapper(model_name: str):
    print(f"[VAE] Using WanVAEWrapper for model: {model_name}", flush=True)
    return WanVAEWrapper(model_name=model_name)


def _parse_video_keys(raw: str) -> list[str]:
    if not raw:
        return []
    keys = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if item.lower().endswith(".mp4"):
            item = item[:-4]
        keys.append(item)
    return keys


def _list_video_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _load_video(path: Path) -> torch.Tensor:
    array = iio.imread(str(path), plugin="pyav")
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"Unexpected shape for video {path}: {array.shape}")
    tensor = torch.from_numpy(array).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    return tensor


def _preprocess_video(
    tensor: torch.Tensor,
    device: torch.device,
    target_height: int | None = None,
    target_width: int | None = None,
) -> torch.Tensor:
    video = tensor.to(device=device, dtype=torch.float32)
    # Resize spatially if target size is specified
    if target_height is not None and target_width is not None:
        B, C, T, H, W = video.shape
        if H != target_height or W != target_width:
            video = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            video = F.interpolate(video, size=(target_height, target_width), mode="bilinear", align_corners=False)
            video = video.reshape(B, T, C, target_height, target_width).permute(0, 2, 1, 3, 4)
    video = video.div_(255.0).mul_(2.0).sub_(1.0)
    return video.to(dtype=torch.bfloat16)


def _is_all_black_video(tensor: torch.Tensor, threshold: int = 2) -> bool:
    """Check if a uint8 video tensor is effectively all black."""
    if tensor is None or tensor.numel() == 0:
        return True
    if tensor.dtype != torch.uint8:
        tensor = tensor.to(dtype=torch.uint8)
    return int(tensor.max().item()) <= threshold


def _encode_video(model: WanVAEWrapper, video_tensor: torch.Tensor) -> torch.Tensor:
    """Encode video tensor to latent. Input: [1, C, T, H, W], Output: [C, T', H', W']."""
    with torch.no_grad():
        latent = model.encode_to_latent(video_tensor)
    return latent.squeeze(0).cpu().to(dtype=torch.bfloat16)


def _encode_frames_independently(model: WanVAEWrapper, video_tensor: torch.Tensor) -> torch.Tensor:
    """
    Encode each frame independently to avoid temporal correlation in 3D VAE.
    Input: [1, C, T, H, W], Output: [C, T', H', W'] where each frame is encoded separately.

    For reference frames that should not have temporal relationships.
    """
    # video_tensor shape: [1, C, T, H, W]
    num_frames = video_tensor.shape[2]
    latents = []

    with torch.no_grad():
        for t in range(num_frames):
            # Extract single frame: [1, C, 1, H, W]
            frame = video_tensor[:, :, t:t+1, :, :]
            # Encode single frame
            latent = model.encode_to_latent(frame)  # [1, C', 1, H', W']
            latents.append(latent.squeeze(0))  # [C', 1, H', W']

    # Concatenate along time dimension: [C', T', H', W']
    combined = torch.cat(latents, dim=1)  # dim=1 is the time dimension after squeeze
    return combined.cpu().to(dtype=torch.bfloat16)


def _log_message(
    message: str,
    rank: int,
    world_size: int,
    level: str = "INFO",
    quiet_nonzero: bool = False,
) -> None:
    if quiet_nonzero and rank != 0 and level != "ERROR":
        return
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    tag = f"[{ts}][VAE][R{rank}/{world_size}]"
    if level != "INFO":
        tag += f"[{level}]"
    print(f"{tag} {message}", flush=True)


def _log_progress(
    idx: int,
    total: int,
    video_id: str,
    rank: int,
    world_size: int,
    status: str = "",
) -> None:
    """每个 rank 都显示进度"""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    status_str = f" {status}" if status else ""
    print(f"[{ts}][VAE][R{rank}/{world_size}] {idx}/{total} {video_id}{status_str}", flush=True)


def main() -> None:
    cfg = CONFIG
    video_keys = _parse_video_keys(cfg.vae_video_keys)
    if not video_keys:
        raise ValueError("vae_video_keys is empty in config.")

    if not torch.cuda.is_available():
        raise RuntimeError("VAE encoding requires a CUDA-capable GPU.")

    rank, world_size, local_rank = get_rank_info()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    # Read encode target size from CONFIG
    target_h = CONFIG.encode_target_height
    target_w = CONFIG.encode_target_width
    _log_message(f"Loading VAE model... (encode resize: {target_w}x{target_h})", rank, world_size, quiet_nonzero=cfg.get("vae_quiet_nonzero", True))
    model = get_vae_wrapper(cfg.vae_model_path).to(device=device, dtype=torch.bfloat16)
    model.eval()

    root = Path(cfg.get("vae_input_root", None) or cfg.output_root)
    video_dirs = _list_video_dirs(root)
    if cfg.get("vae_max_videos", None) is not None:
        video_dirs = video_dirs[: cfg.get("vae_max_videos", None)]
    video_dirs = shard_items(video_dirs, rank, world_size)

    total_dirs = len(video_dirs)
    # Use tqdm progress bar on rank 0
    if rank == 0:
        iterator = tqdm(enumerate(video_dirs, start=1), total=total_dirs, desc="VAE encode")
    else:
        iterator = enumerate(video_dirs, start=1)
    for idx, video_dir in iterator:
        video_id = video_dir.name

        latents_path = video_dir / LATENTS_FILENAME

        # 需要加载或迁移时才读取文件
        existing_latents: dict[str, torch.Tensor] = {}
        if latents_path.exists():
            try:
                existing_latents = torch.load(latents_path, map_location="cpu", weights_only=True)
            except Exception:
                existing_latents = {}

        # 快速检查：如果 latents.pt 已包含所有可用 mp4 的 key（且无旧 .pt），才跳过
        if cfg.get("vae_skip_existing", True) and latents_path.exists():
            has_old_pt = any((video_dir / f"{key}.pt").exists() for key in video_keys)
            missing_key = False
            for key in video_keys:
                video_path = video_dir / f"{key}.mp4"
                if not video_path.exists():
                    continue
                if key not in existing_latents:
                    missing_key = True
                    break
            if not has_old_pt and not missing_key and existing_latents:
                _log_progress(idx, total_dirs, video_id, rank, world_size, status="skip(exist)")
                continue

        # 检查旧的单独 .pt 文件，合并到 latents 中
        for key in video_keys:
            old_pt_path = video_dir / f"{key}.pt"
            if old_pt_path.exists() and key not in existing_latents:
                try:
                    old_data = torch.load(old_pt_path, map_location="cpu", weights_only=True)
                    if "latent" in old_data:
                        existing_latents[key] = old_data["latent"]
                except Exception:
                    pass

        # 检查是否所有 key 都已编码
        if cfg.get("vae_skip_existing", True):
            all_done = True
            for key in video_keys:
                video_path = video_dir / f"{key}.mp4"
                if not video_path.exists():
                    continue  # 视频不存在，跳过
                if key not in existing_latents:
                    all_done = False
                    break
            if all_done and existing_latents:
                # 删除旧的单独 .pt 文件
                if cfg.get("vae_delete_old_pt", True):
                    for key in video_keys:
                        old_pt_path = video_dir / f"{key}.pt"
                        if old_pt_path.exists():
                            old_pt_path.unlink()
                _log_progress(idx, total_dirs, video_id, rank, world_size, status="skip(exist)")
                continue

        # 编码缺失的 keys
        latents_dict = dict(existing_latents)
        encoded_keys = []
        skipped_keys = []

        for key in video_keys:
            # 如果已存在且 skip_existing，跳过
            if cfg.get("vae_skip_existing", True) and key in latents_dict:
                skipped_keys.append(key)
                continue

            video_path = video_dir / f"{key}.mp4"
            if not video_path.exists():
                # For reference videos, missing is expected (not all samples have references)
                if "reference" not in key:
                    _log_message(f"Missing {video_path}", rank, world_size, level="WARN", quiet_nonzero=cfg.get("vae_quiet_nonzero", True))
                continue

            raw_video = None
            video_tensor = None
            try:
                raw_video = _load_video(video_path)
                if "proj_fg" in key and _is_all_black_video(raw_video):
                    _log_message(
                        f"Skip {key} (foreground proj is all black)",
                        rank,
                        world_size,
                        level="SKIP",
                        quiet_nonzero=cfg.get("vae_quiet_nonzero", True),
                    )
                    continue
                video_tensor = _preprocess_video(raw_video, device, target_h, target_w)

                # For reference videos, encode each frame independently to avoid temporal correlation
                if "reference" in key:
                    latent = _encode_frames_independently(model, video_tensor)
                else:
                    latent = _encode_video(model, video_tensor)

                latents_dict[key] = latent
                encoded_keys.append(key)
            except Exception as exc:
                _log_message(f"Failed {video_path}: {exc}", rank, world_size, level="ERROR", quiet_nonzero=cfg.get("vae_quiet_nonzero", True))
            finally:
                del raw_video, video_tensor

        # 保存合并后的 latents
        if latents_dict:
            try:
                torch.save(latents_dict, latents_path)

                # 删除旧的单独 .pt 文件
                if cfg.get("vae_delete_old_pt", True):
                    for key in video_keys:
                        old_pt_path = video_dir / f"{key}.pt"
                        if old_pt_path.exists():
                            old_pt_path.unlink()

                # 显示进度
                if encoded_keys:
                    status = f"encoded {len(encoded_keys)}: {', '.join(encoded_keys[:3])}"
                    if len(encoded_keys) > 3:
                        status += f"... (+{len(encoded_keys)-3})"
                    _log_progress(idx, total_dirs, video_id, rank, world_size, status=status)
                else:
                    # 没有新编码的 key，说明都已存在，静默跳过（不打印）
                    pass
            except Exception as exc:
                _log_progress(idx, total_dirs, video_id, rank, world_size, status=f"ERROR: {exc}")
        else:
            _log_progress(idx, total_dirs, video_id, rank, world_size, status="skip(no mp4)")

    _log_message("Done", rank, world_size)


if __name__ == "__main__":
    main()