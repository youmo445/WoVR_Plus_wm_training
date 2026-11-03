"""Camera-ray and end-effector projection utilities derived from DiffSynth-Studio."""
from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Tuple
import numpy as np
import cv2
from PIL import Image

ColorListLeft = [(0, 0, 255), (255, 255, 0), (0, 255, 255)]

ColorListRight = [(255, 0, 255), (255, 0, 0), (0, 255, 0)]

EndEffectorPts = np.array(
    [[0, 0, 0, 1], [0.1, 0, 0, 1], [0, 0.1, 0, 1], [0, 0, 0.1, 1]], dtype=np.float32
)

Gripper2EEFCvt = np.array(
    [[1, 0, 0, 0.2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
)

def quaternion_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    return (
        np.stack(
            [
                1.0 - (tyy + tzz),
                txy - twz,
                txz + twy,
                txy + twz,
                1.0 - (txx + tzz),
                tyz - twx,
                txz - twy,
                tyz + twx,
                1.0 - (txx + tyy),
            ],
            axis=-1,
        )
        .reshape(quat.shape[:-1] + (3, 3))
        .astype(np.float32)
    )

def get_transformation_matrix_from_quat(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    mat = np.tile(np.eye(4, dtype=np.float32), (pose.shape[0], 1, 1))
    mat[:, :3, :3] = quaternion_xyzw_to_matrix(pose[:, 3:7])
    mat[:, :3, 3] = pose[:, :3]
    return mat

def grip_to_left_color(v: float) -> Tuple[int, int, int]:
    v = float(np.clip(v, 0.0, 1.0))
    return (int(40 + 140 * v), int(120 + 120 * v), int(40 + 40 * (1 - v)))

def grip_to_right_color(v: float) -> Tuple[int, int, int]:
    v = float(np.clip(v, 0.0, 1.0))
    return (int(120 + 135 * v), int(30 + 80 * v), int(30 + 80 * v))

def compute_depth_radius(
    depth: float,
    radius: int,
    radius_mode: str = "constant",
    min_radius: int = 20,
    max_radius: int = 60,
    ref_depth: float = 0.30,
    near_depth: float = 0.19,
    far_depth: float = 0.69,
) -> int:
    if radius_mode == "constant":
        return int(radius)
    safe_depth = max(float(depth), 1e-4)
    if radius_mode == "perspective":
        dynamic_radius = float(radius) * float(ref_depth) / safe_depth
    elif radius_mode == "depth_norm":
        alpha = (float(far_depth) - safe_depth) / (float(far_depth) - float(near_depth) + 1e-8)
        dynamic_radius = float(min_radius) + float(np.clip(alpha, 0.0, 1.0)) * (
            float(max_radius) - float(min_radius)
        )
    else:
        raise ValueError(f"Unsupported radius_mode: {radius_mode}")
    return int(round(np.clip(dynamic_radius, min_radius, max_radius)))

def generate_traj_map(
    abs_actions: np.ndarray,
    w2c: np.ndarray,
    intrinsic: np.ndarray,
    h: int,
    w: int,
    radius: int = 40,
    radius_mode: str = "perspective",
    min_radius: int = 14,
    max_radius: int = 58,
    ref_depth: float = 0.30,
    near_depth: float = 0.19,
    far_depth: float = 0.69,
) -> List[np.ndarray]:
    """Original-style projection: abs_actions and w2c must be in the same per-camera world frame."""
    abs_actions = np.asarray(abs_actions, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    w2c = np.asarray(w2c, dtype=np.float32)
    if w2c.ndim == 2:
        w2c = np.repeat(w2c[None], abs_actions.shape[0], axis=0)

    ee_key_pts = EndEffectorPts.reshape(1, 4, 4).transpose(0, 2, 1)
    cvt_matrix = Gripper2EEFCvt.reshape(1, 4, 4)
    pose_l_mat = get_transformation_matrix_from_quat(abs_actions[:, 0:7])
    pose_r_mat = get_transformation_matrix_from_quat(abs_actions[:, 8:15])
    ee2cam_l = np.matmul(np.matmul(w2c, pose_l_mat), cvt_matrix)
    ee2cam_r = np.matmul(np.matmul(w2c, pose_r_mat), cvt_matrix)
    pts_l = np.matmul(ee2cam_l, ee_key_pts)
    pts_r = np.matmul(ee2cam_r, ee_key_pts)

    intr = intrinsic.reshape(1, 3, 3)
    uvs_l = np.matmul(intr, pts_l[:, :3, :])
    uvs_l = (uvs_l / (pts_l[:, 2:3, :] + 1e-8))[:, :2, :].transpose(0, 2, 1).astype(np.int64)
    uvs_r = np.matmul(intr, pts_r[:, :3, :])
    uvs_r = (uvs_r / (pts_r[:, 2:3, :] + 1e-8))[:, :2, :].transpose(0, 2, 1).astype(np.int64)

    img_list: List[np.ndarray] = []
    for i in range(abs_actions.shape[0]):
        img = np.zeros((h, w, 3), dtype=np.uint8) + 50
        color_l = grip_to_left_color(abs_actions[i, 7].item() / 120.0)
        color_r = grip_to_right_color(abs_actions[i, 15].item() / 120.0)

        for points, pts, color in zip(
            [uvs_l[i], uvs_r[i]], [pts_l[i], pts_r[i]], [color_l, color_r]
        ):
            if float(pts[2, 0]) <= 1e-6:
                continue
            base = np.asarray(points[0])
            if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                continue
            radius_i = compute_depth_radius(
                float(pts[2, 0]),
                radius,
                radius_mode,
                min_radius,
                max_radius,
                ref_depth,
                near_depth,
                far_depth,
            )
            if cv2 is not None:
                cv2.circle(img, tuple(base), radius_i, color, -1)
            else:
                from PIL import ImageDraw

                pil = Image.fromarray(img)
                d = ImageDraw.Draw(pil)
                x, y = int(base[0]), int(base[1])
                d.ellipse(
                    [x - radius_i, y - radius_i, x + radius_i, y + radius_i], fill=tuple(color)
                )
                img = np.asarray(pil)

        for points, pts, colors in zip(
            [uvs_l[i], uvs_r[i]], [pts_l[i], pts_r[i]], [ColorListLeft, ColorListRight]
        ):
            if float(pts[2, 0]) <= 1e-6:
                continue
            base = np.asarray(points[0])
            if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                continue
            for point_idx, point in enumerate(points):
                if point_idx == 0 or float(pts[2, point_idx]) <= 1e-6:
                    continue
                if cv2 is not None:
                    cv2.line(img, tuple(base), tuple(point[:2]), colors[point_idx - 1], 8)
                else:
                    from PIL import ImageDraw

                    pil = Image.fromarray(img)
                    ImageDraw.Draw(pil).line(
                        [tuple(map(int, base)), tuple(map(int, point[:2]))],
                        fill=tuple(colors[point_idx - 1]),
                        width=8,
                    )
                    img = np.asarray(pil)
        img_list.append(img)
    return img_list

def generate_raymap(
    intrinsic: np.ndarray, c2w: np.ndarray, height: int, width: int
) -> Tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    xs = np.arange(width, dtype=np.float32) + 0.5
    ys = np.arange(height, dtype=np.float32) + 0.5
    xx, yy = np.meshgrid(xs, ys)
    dirs = np.stack([(xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)], axis=-1)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    rays_d = dirs @ R.T
    rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)
    rays_o = np.broadcast_to(t, rays_d.shape).copy()
    return rays_o.astype(np.float32), rays_d.astype(np.float32)

def to_uint8_img(value: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    value = (value - vmin) / (vmax - vmin + 1e-8)
    return (np.clip(value, 0.0, 1.0) * 255).astype(np.uint8)

def write_video_rgb(frames_rgb: Sequence[np.ndarray], output_path: Path, fps: float = 5.0) -> None:
    import imageio.v2 as imageio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=fps)
    try:
        for f in frames_rgb:
            writer.append_data(np.asarray(f, dtype=np.uint8))
    finally:
        writer.close()
