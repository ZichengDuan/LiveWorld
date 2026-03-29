#!/usr/bin/env python3
"""
Plot 3D trajectories from event_bench plans.

Supports any number of combo variants.
Combo IDs are read from each plan.json's ``combo_id`` field (falls back to
``case_type`` for backward compatibility).

Outputs:
- Global overlay plot (all combos in one 3D view)
- Optional frustum animations (global + separate)
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Auto color palette for any number of combos
_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
    "#e84393", "#00cec9", "#6c5ce7", "#fdcb6e",
    "#d63031", "#0984e3", "#00b894", "#636e72",
]


def _color_for_idx(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]


def _load_plan(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _plan_key(plan: Dict) -> str:
    """Get combo_id from plan, fall back to case_type."""
    return plan.get("combo_id", plan.get("case_type", "unknown"))


def _trajectory_from_plan(plan: Dict) -> np.ndarray:
    """Extract trajectory positions, preferring trajectory_frames (world coords)."""
    if "trajectory_frames" in plan and plan["trajectory_frames"]:
        pts = []
        for fr in plan["trajectory_frames"]:
            c2w = np.asarray(fr["c2w"], dtype=np.float32)
            pts.append(c2w[:3, 3])
        return np.stack(pts, axis=0)
    round_plan = plan["round_plan"]
    keys = sorted(round_plan.keys(), key=lambda x: int(x))
    pts: List[List[float]] = []
    for i, k in enumerate(keys):
        r = round_plan[k]
        if i == 0:
            pts.append(r["camera_start"])
        pts.append(r["camera_end"])
    return np.asarray(pts, dtype=np.float32)


def _event_points(event_slots: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in event_slots.items():
        if "position" in v:
            out[k] = np.asarray(v["position"], dtype=np.float32)
    return out


def _downsample_points(points: np.ndarray, colors: Optional[np.ndarray], max_points: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if len(points) <= max_points:
        return points, colors
    idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
    if colors is None:
        return points[idx], None
    return points[idx], colors[idx]


def _load_scene_pointcloud(pointcloud_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    if not pointcloud_path:
        return None, None, "none"
    path = Path(pointcloud_path)
    if not path.exists():
        return None, None, "missing"

    if path.suffix.lower() == ".ply":
        try:
            from plyfile import PlyData
        except Exception:
            return None, None, "plyfile_not_available"
        ply = PlyData.read(str(path))
        v = ply["vertex"]
        points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
        color = None
        if {"red", "green", "blue"}.issubset(v.data.dtype.names):
            color = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
        return points, color, "ply"

    if path.suffix.lower() == ".npz":
        data = np.load(str(path))
        for key in ("points_world", "points", "xyz"):
            if key in data:
                pts = data[key].astype(np.float32)
                col = None
                for ckey in ("colors", "rgb"):
                    if ckey in data:
                        col = data[ckey]
                        if col.dtype != np.uint8:
                            col = np.clip(col, 0, 255).astype(np.uint8)
                        break
                return pts, col, "npz"
        return None, None, "npz_no_points"

    return None, None, "unsupported"


def _proxy_scene_cloud(center: np.ndarray, radius: float, num_r: int = 30, num_t: int = 90, y_offset: float = -0.75) -> np.ndarray:
    rs = np.linspace(0.0, radius, num_r)
    ts = np.linspace(0.0, 2.0 * np.pi, num_t, endpoint=False)
    pts = []
    for r in rs:
        for t in ts:
            x = center[0] + r * np.sin(t)
            z = center[2] + r * np.cos(t)
            y = center[1] + y_offset
            pts.append([x, y, z])
    return np.asarray(pts, dtype=np.float32)


def _compute_scene_bounds(trajs: Dict[str, np.ndarray], event_pts: Dict[str, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, float]:
    all_pts = []
    for t in trajs.values():
        all_pts.append(t)
    for m in event_pts.values():
        for p in m.values():
            all_pts.append(p[None, :])
    cat = np.concatenate(all_pts, axis=0)
    center = np.median(cat, axis=0)
    span = np.percentile(cat[:, [0, 2]], 95, axis=0) - np.percentile(cat[:, [0, 2]], 5, axis=0)
    radius = float(max(1.0, 0.8 * np.max(span)))
    return center, radius


def _set_equal_3d(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins + 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_scene(ax, scene_points: np.ndarray, scene_colors: Optional[np.ndarray], alpha: float = 0.20, s: float = 1.0) -> None:
    if scene_colors is None:
        ax.scatter(scene_points[:, 0], scene_points[:, 1], scene_points[:, 2], s=s, c="gray", alpha=alpha, depthshade=False)
        return
    c = scene_colors.astype(np.float32) / 255.0
    ax.scatter(scene_points[:, 0], scene_points[:, 1], scene_points[:, 2], s=s, c=c, alpha=alpha, depthshade=False)


def _plot_single_case(
    ax,
    label: str,
    color: str,
    traj: np.ndarray,
    epts: Dict[str, np.ndarray],
    scene_points: np.ndarray,
    scene_colors: Optional[np.ndarray],
    show_scene: bool,
    title_suffix: str = "",
    first_c2w: Optional[np.ndarray] = None,
    frustum_scale: float = 0.18,
) -> None:
    if show_scene:
        _plot_scene(ax, scene_points, scene_colors, alpha=0.20, s=1.0)
    if first_c2w is not None:
        _draw_frustum_from_c2w(ax, first_c2w, color="#8e44ad", scale=frustum_scale * 1.5, alpha=1.0, label="DA3 cam0")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, linewidth=2.2, label=label)
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], marker="^", c=color, s=70)
    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], marker="x", c=color, s=80)
    for eid, p in epts.items():
        ax.scatter(p[0], p[1], p[2], marker="*", c="black", s=90)
        ax.text(p[0], p[1], p[2], f" {eid}", fontsize=9)
    for i, p in enumerate(traj):
        ax.text(p[0], p[1], p[2], f"R{i}", fontsize=7, color=color)
    ax.set_title(f"{label}{title_suffix}", fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=28, azim=-62)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return v / n


def _camera_basis_from_look(pos: np.ndarray, look_at: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _normalize(look_at - pos)
    if float(np.linalg.norm(forward)) < 1e-6:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if float(np.linalg.norm(right)) < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, world_up)
    right = _normalize(right)
    up = _normalize(np.cross(right, forward))
    return right, up, forward


def _draw_frustum(
    ax,
    pos: np.ndarray,
    look_at: np.ndarray,
    color: str,
    scale: float = 0.18,
    alpha: float = 0.9,
) -> None:
    right, up, forward = _camera_basis_from_look(pos, look_at)
    _draw_frustum_axes(ax, pos, right, up, forward, color, scale, alpha)


def _draw_frustum_from_c2w(
    ax,
    c2w: np.ndarray,
    color: str,
    scale: float = 0.18,
    alpha: float = 0.9,
    label: str = "",
) -> None:
    """Draw frustum directly from a 4x4 c2w matrix (OpenCV: col2=forward)."""
    pos = c2w[:3, 3]
    right = c2w[:3, 0]
    up = -c2w[:3, 1]
    forward = c2w[:3, 2]
    _draw_frustum_axes(ax, pos, right, up, forward, color, scale, alpha)
    if label:
        ax.text(pos[0], pos[1], pos[2], f"  {label}", fontsize=9, color=color, fontweight="bold")


def _draw_frustum_axes(
    ax,
    pos: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
    color: str,
    scale: float = 0.18,
    alpha: float = 0.9,
) -> None:
    depth = scale
    half_w = scale * 0.55
    half_h = scale * 0.38

    center_far = pos + forward * depth
    c1 = center_far - right * half_w - up * half_h
    c2 = center_far + right * half_w - up * half_h
    c3 = center_far + right * half_w + up * half_h
    c4 = center_far - right * half_w + up * half_h

    corners = [c1, c2, c3, c4]
    for c in corners:
        ax.plot([pos[0], c[0]], [pos[1], c[1]], [pos[2], c[2]], color=color, linewidth=1.6, alpha=alpha)
    for a, b in [(c1, c2), (c2, c3), (c3, c4), (c4, c1)]:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, linewidth=1.2, alpha=alpha)
    tip = pos + forward * (depth * 1.35)
    ax.plot([pos[0], tip[0]], [pos[1], tip[1]], [pos[2], tip[2]], color=color, linewidth=2.0, alpha=alpha)


def _interpolate_camera_track(
    round_plan: Dict[str, Dict],
    steps_per_round: int,
    trajectory_frames: Optional[List[Dict]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if trajectory_frames:
        pos_list: List[np.ndarray] = []
        look_list: List[np.ndarray] = []
        for fr in trajectory_frames:
            c2w = np.asarray(fr["c2w"], dtype=np.float32)
            p = c2w[:3, 3]
            fwd = c2w[:3, 2]
            look = p + fwd
            pos_list.append(p)
            look_list.append(look)
        if pos_list:
            return np.stack(pos_list, axis=0), np.stack(look_list, axis=0)

    keys = sorted(round_plan.keys(), key=lambda x: int(x))
    pos_list: List[np.ndarray] = []
    look_list: List[np.ndarray] = []
    for ridx, k in enumerate(keys):
        r = round_plan[k]
        start = np.asarray(r["camera_start"], dtype=np.float32)
        end = np.asarray(r["camera_end"], dtype=np.float32)
        look = np.asarray(r["look_at"], dtype=np.float32)
        for s in range(steps_per_round):
            t = float(s) / float(steps_per_round)
            p = start * (1.0 - t) + end * t
            pos_list.append(p)
            look_list.append(look)
    last = round_plan[keys[-1]]
    pos_list.append(np.asarray(last["camera_end"], dtype=np.float32))
    look_list.append(np.asarray(last["look_at"], dtype=np.float32))
    return np.stack(pos_list, axis=0), np.stack(look_list, axis=0)


def _figure_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf[:, :, :3].copy()


def _write_mp4(path: Path, frames: List[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7) as writer:
        for fr in frames:
            writer.append_data(fr)


def _write_gif(path: Path, frames: List[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / max(1, fps)
    imageio.mimsave(str(path), frames, duration=duration)


def _render_global_animation(
    out_dir: Path,
    combo_order: List[str],
    combo_colors: Dict[str, str],
    trajs: Dict[str, np.ndarray],
    epts: Dict[str, Dict[str, np.ndarray]],
    combo_to_plan: Dict[str, Dict],
    scene_pts: np.ndarray,
    scene_cols: Optional[np.ndarray],
    bounds: np.ndarray,
    fps: int,
    steps_per_round: int,
    show_scene: bool,
    frustum_scale: float,
    save_gif: bool,
    first_c2w: Optional[np.ndarray] = None,
) -> None:
    tracks = {}
    for cid in combo_order:
        p, l = _interpolate_camera_track(
            combo_to_plan[cid]["round_plan"],
            steps_per_round=steps_per_round,
            trajectory_frames=combo_to_plan[cid].get("trajectory_frames"),
        )
        tracks[cid] = (p, l)
    total_frames = max(v[0].shape[0] for v in tracks.values())

    frames: List[np.ndarray] = []
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    for fi in range(total_frames):
        ax.cla()
        if show_scene:
            _plot_scene(ax, scene_pts, scene_cols, alpha=0.14, s=0.8)
        if first_c2w is not None:
            _draw_frustum_from_c2w(ax, first_c2w, color="#8e44ad", scale=frustum_scale * 1.5, alpha=0.6, label="DA3 cam0")
        for cid in combo_order:
            col = combo_colors[cid]
            t = trajs[cid]
            ax.plot(t[:, 0], t[:, 1], t[:, 2], color=col, linewidth=1.2, alpha=0.35)
            pos, look = tracks[cid]
            cur_idx = min(fi, pos.shape[0] - 1)
            path = pos[: cur_idx + 1]
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color=col, linewidth=2.3, alpha=0.95)
            p_now = pos[cur_idx]
            l_now = look[cur_idx]
            ax.scatter(p_now[0], p_now[1], p_now[2], c=col, s=30)
            _draw_frustum(ax, p_now, l_now, color=col, scale=frustum_scale, alpha=0.9)
            for eid, ep in epts[cid].items():
                ax.scatter(ep[0], ep[1], ep[2], marker="*", c="black", s=70, alpha=0.9)
                ax.text(ep[0], ep[1], ep[2], eid, fontsize=8)
        ax.set_title(f"Global Camera Frustums | frame {fi+1}/{total_frames}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=28, azim=-62)
        _set_equal_3d(ax, bounds)
        frames.append(_figure_to_rgb(fig))
    plt.close(fig)

    _write_mp4(out_dir / "trajectories_global_frustum.mp4", frames, fps=fps)
    if save_gif:
        _write_gif(out_dir / "trajectories_global_frustum.gif", frames, fps=fps)


def _render_separate_animations(
    out_dir: Path,
    combo_order: List[str],
    combo_colors: Dict[str, str],
    trajs: Dict[str, np.ndarray],
    epts: Dict[str, Dict[str, np.ndarray]],
    combo_to_plan: Dict[str, Dict],
    scene_pts: np.ndarray,
    scene_cols: Optional[np.ndarray],
    bounds: np.ndarray,
    fps: int,
    steps_per_round: int,
    show_scene: bool,
    frustum_scale: float,
    save_gif: bool,
    first_c2w: Optional[np.ndarray] = None,
) -> None:
    for cid in combo_order:
        pos, look = _interpolate_camera_track(
            combo_to_plan[cid]["round_plan"],
            steps_per_round=steps_per_round,
            trajectory_frames=combo_to_plan[cid].get("trajectory_frames"),
        )
        total_frames = pos.shape[0]
        frames: List[np.ndarray] = []
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for fi in range(total_frames):
            ax.cla()
            if show_scene:
                _plot_scene(ax, scene_pts, scene_cols, alpha=0.14, s=0.8)
            if first_c2w is not None:
                _draw_frustum_from_c2w(ax, first_c2w, color="#8e44ad", scale=frustum_scale * 1.5, alpha=0.6, label="DA3 cam0")
            col = combo_colors[cid]
            t = trajs[cid]
            ax.plot(t[:, 0], t[:, 1], t[:, 2], color=col, linewidth=1.3, alpha=0.35)
            path = pos[: fi + 1]
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color=col, linewidth=2.5, alpha=0.98)
            p_now = pos[fi]
            l_now = look[fi]
            ax.scatter(p_now[0], p_now[1], p_now[2], c=col, s=36)
            _draw_frustum(ax, p_now, l_now, color=col, scale=frustum_scale, alpha=0.95)
            for eid, ep in epts[cid].items():
                ax.scatter(ep[0], ep[1], ep[2], marker="*", c="black", s=75, alpha=0.9)
                ax.text(ep[0], ep[1], ep[2], eid, fontsize=8)
            ax.set_title(f"{cid} Camera Frustum | frame {fi+1}/{total_frames}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=28, azim=-62)
            _set_equal_3d(ax, bounds)
            frames.append(_figure_to_rgb(fig))
        plt.close(fig)

        _write_mp4(out_dir / f"trajectory_{cid}_frustum.mp4", frames, fps=fps)
        if save_gif:
            _write_gif(out_dir / f"trajectory_{cid}_frustum.gif", frames, fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot event_bench 3D trajectories")
    parser.add_argument(
        "--plans-glob",
        default="outputs/event_bench/dog_bowl_case*_seed42/plan.json",
        help="Glob for plan.json files",
    )
    parser.add_argument(
        "--scene-pointcloud", default="",
        help="Optional scene pointcloud path (.ply/.npz)",
    )
    parser.add_argument("--output-dir", default="outputs/event_bench/visualizations")
    parser.add_argument("--max-scene-points", type=int, default=80000)
    parser.add_argument("--no-scene", action="store_true")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--animate-separate", action="store_true")
    parser.add_argument("--animate-gif", action="store_true")
    parser.add_argument("--animate-fps", type=int, default=10)
    parser.add_argument("--animate-steps-per-round", type=int, default=16)
    parser.add_argument("--frustum-scale", type=float, default=0.18)
    parser.add_argument("--animate-max-scene-points", type=int, default=25000)
    args = parser.parse_args()

    plan_paths = sorted(glob.glob(args.plans_glob))
    if not plan_paths:
        raise FileNotFoundError(f"No plans found for glob: {args.plans_glob}")

    plans = [_load_plan(p) for p in plan_paths]

    # Build combo_id -> plan mapping (dynamic, not hardcoded)
    combo_to_plan: Dict[str, Dict] = {}
    for p in plans:
        key = _plan_key(p)
        combo_to_plan[key] = p
    combo_order = list(combo_to_plan.keys())
    combo_colors = {cid: _color_for_idx(i) for i, cid in enumerate(combo_order)}

    trajs = {cid: _trajectory_from_plan(combo_to_plan[cid]) for cid in combo_order}
    epts = {cid: _event_points(combo_to_plan[cid].get("event_slots", {})) for cid in combo_order}

    # DA3 first-frame c2w
    first_c2w = None
    for p in plans:
        if "first_c2w" in p:
            first_c2w = np.asarray(p["first_c2w"], dtype=np.float32)
            break

    scene_pts, scene_cols, scene_source = _load_scene_pointcloud(args.scene_pointcloud)
    if scene_pts is None:
        center, radius = _compute_scene_bounds(trajs, epts)
        scene_pts = _proxy_scene_cloud(center=center, radius=radius)
        scene_cols = None
        scene_source = "proxy"
    scene_pts, scene_cols = _downsample_points(scene_pts, scene_cols, args.max_scene_points)

    all_bounds = [scene_pts]
    all_bounds.extend(list(trajs.values()))
    for em in epts.values():
        for p in em.values():
            all_bounds.append(p[None, :])
    bounds = np.concatenate(all_bounds, axis=0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Global overlay
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    if not args.no_scene:
        _plot_scene(ax, scene_pts, scene_cols, alpha=0.18, s=1.0)
    for cid in combo_order:
        t = trajs[cid]
        col = combo_colors[cid]
        ax.plot(t[:, 0], t[:, 1], t[:, 2], color=col, linewidth=2.4, label=cid)
        ax.scatter(t[0, 0], t[0, 1], t[0, 2], marker="^", c=col, s=70)
        ax.scatter(t[-1, 0], t[-1, 1], t[-1, 2], marker="x", c=col, s=80)
    for cid in combo_order:
        for eid, p in epts[cid].items():
            ax.scatter(p[0], p[1], p[2], marker="*", c="black", s=90)
            ax.text(p[0], p[1], p[2], f"{eid}", fontsize=9)
    if first_c2w is not None:
        _draw_frustum_from_c2w(ax, first_c2w, color="#8e44ad", scale=args.frustum_scale * 1.5, alpha=1.0, label="DA3 cam0")
    ax.set_title(f"Global 3D Trajectories (scene={scene_source})", fontsize=13)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=28, azim=-62)
    _set_equal_3d(ax, bounds)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectories_global_overlay.png", dpi=220)
    plt.close(fig)

    if args.animate:
        ani_pts, ani_cols = _downsample_points(scene_pts, scene_cols, args.animate_max_scene_points)
        _render_global_animation(
            out_dir=out_dir,
            combo_order=combo_order,
            combo_colors=combo_colors,
            trajs=trajs,
            epts=epts,
            combo_to_plan=combo_to_plan,
            scene_pts=ani_pts,
            scene_cols=ani_cols,
            bounds=bounds,
            fps=args.animate_fps,
            steps_per_round=args.animate_steps_per_round,
            show_scene=not args.no_scene,
            frustum_scale=args.frustum_scale,
            save_gif=args.animate_gif,
            first_c2w=first_c2w,
        )
        if args.animate_separate:
            _render_separate_animations(
                out_dir=out_dir,
                combo_order=combo_order,
                combo_colors=combo_colors,
                trajs=trajs,
                epts=epts,
                combo_to_plan=combo_to_plan,
                scene_pts=ani_pts,
                scene_cols=ani_cols,
                bounds=bounds,
                fps=args.animate_fps,
                steps_per_round=args.animate_steps_per_round,
                show_scene=not args.no_scene,
                frustum_scale=args.frustum_scale,
                save_gif=args.animate_gif,
                first_c2w=first_c2w,
            )

    print(f"[plot_trajectories_3d] Saved overlay plot to: {out_dir / 'trajectories_global_overlay.png'}")


if __name__ == "__main__":
    main()
