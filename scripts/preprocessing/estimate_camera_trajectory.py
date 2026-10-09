"""Estimate the camera trajectory of every recording with RGB-D DROID-SLAM.

SLAM runs on the full recording (including the frames before the episode start). The manipulated
object and the hand are hidden in the episode frames (random RGB noise, zero depth), so that only
the static scene drives the camera estimate.

Writes `extrinsics_droid_all` [T_full, 4, 4] and `extrinsics_droid` [T, 4, 4] (from the episode
start): camera poses in the SLAM world frame (world <- camera).
"""
import os
import sys
from types import SimpleNamespace

import cv2
import hydra
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from egoavflow.preprocessing.common import DROID_SLAM_ROOT, episode_range, load_rgbd_pngs, resolve_data_dirs, save_array

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def hide_dynamic_regions(rgb_frames, depth_frames, mask, use_rgb_noise, use_fake_depth):
    """Replace masked pixels with random colors and zero depth (in place)."""
    if use_rgb_noise:
        noise = np.random.randint(0, 256, size=rgb_frames.shape, dtype=np.uint8)
        rgb_frames[:] = np.where(np.repeat(mask[..., None], rgb_frames.shape[-1], axis=-1) == 1, noise, rgb_frames)
    if use_fake_depth:
        depth_frames[:] = np.where(mask == 1, 0, depth_frames)


def image_stream(rgb_frames, depth_frames, K, use_depth=True):
    """Frames resized to about 384x512 (cropped to a multiple of 8) with the scaled intrinsics."""
    for frame_idx, (rgb_frame, depth_frame) in enumerate(zip(rgb_frames, depth_frames)):
        image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
        image = cv2.resize(image, (w1, h1))
        image = image[: h1 - h1 % 8, : w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1).cuda()

        depth = torch.as_tensor(depth_frame, dtype=torch.float32)
        depth = F.interpolate(depth[None, None], (h1, w1)).squeeze()
        depth = depth[: h1 - h1 % 8, : w1 - w1 % 8].cuda()

        intrinsics = torch.as_tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        intrinsics[0::2] *= w1 / w0
        intrinsics[1::2] *= h1 / h0
        intrinsics = intrinsics.cuda()

        if use_depth:
            yield frame_idx, image[None], depth, intrinsics
        else:
            yield frame_idx, image[None], intrinsics


def save_trajectory_plot(extrinsics, output_path):
    positions = extrinsics[:, :3, 3]
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=np.arange(len(positions)), cmap="viridis", s=50, alpha=0.7)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "k-", alpha=0.5, linewidth=2)
    for i in range(0, len(positions), max(1, len(positions) // 10)):
        for axis, color in zip(np.eye(3), ["r", "g", "b"]):
            end = positions[i] + 0.1 * extrinsics[i][:3, :3] @ axis
            ax.plot(*zip(positions[i], end), color + "-", linewidth=2, alpha=0.8)
    max_range = (positions.max(axis=0) - positions.min(axis=0)).max() / 2.0
    mid = (positions.max(axis=0) + positions.min(axis=0)) * 0.5
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Camera trajectory ({len(extrinsics)} poses)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_droid(cfg, rgb_frames, depth_frames, K):
    from droid import Droid

    args = SimpleNamespace(weights=cfg.droid_checkpoint, **OmegaConf.to_container(cfg.droid))
    droid = None
    for frame_idx, image, depth, intrinsics in image_stream(rgb_frames, depth_frames, K):
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        droid.track(frame_idx, image, depth, intrinsics=intrinsics)
    # [T, 7]: x, y, z, qx, qy, qz, qw
    traj_est = droid.terminate(image_stream(rgb_frames, depth_frames, K, use_depth=False))

    extrinsics = np.tile(np.eye(4), (len(traj_est), 1, 1))
    extrinsics[:, :3, :3] = Rotation.from_quat(traj_est[:, 3:]).as_matrix()
    extrinsics[:, :3, 3] = traj_est[:, :3]
    return extrinsics


@hydra.main(version_base=None, config_path="../../config/preprocessing", config_name="camera_trajectory")
def main(cfg: DictConfig):
    sys.path.insert(0, os.path.join(DROID_SLAM_ROOT, "droid_slam"))
    torch.multiprocessing.set_start_method("spawn", force=True)
    K = np.array(cfg.camera.intrinsics)

    for data_dir in resolve_data_dirs(cfg.data_dirs):
        root = zarr.open(data_dir, mode="a")
        for episode_idx in tqdm(episode_range(data_dir, cfg.episode_start, cfg.episode_end), desc=os.path.basename(data_dir)):
            key = f"episode_{episode_idx}"
            if key not in root or "sam_mask_sequence_multi_obj" not in root[key]:
                print(f"{key}: no episode data (run build_episodes.py first), skipping")
                continue
            episode = root[key]
            episode_dir = os.path.join(data_dir, key)
            rgb_frames, depth_frames = load_rgbd_pngs(episode_dir)
            depth_frames = depth_frames / 1000.0
            start = int(episode["all_detected_frame_index"][()])

            mask = episode["sam_mask_sequence_multi_obj"][:]  # [T, objects + hand, H, W]
            mask = np.any(mask[:, list(cfg.dynamic_mask_channels)] == 1, axis=1)
            assert rgb_frames[start:].shape[0] == mask.shape[0]
            np.random.seed(cfg.seed + episode_idx)
            hide_dynamic_regions(rgb_frames[start:], depth_frames[start:], mask, cfg.use_rgb_noise, cfg.use_fake_depth)

            extrinsics = run_droid(cfg, rgb_frames, depth_frames, K)
            save_array(episode, "extrinsics_droid", extrinsics[start:])
            save_array(episode, "extrinsics_droid_all", extrinsics)
            save_trajectory_plot(extrinsics, os.path.join(episode_dir, "camera_trajectory.png"))
            distance = np.sum(np.linalg.norm(np.diff(extrinsics[:, :3, 3], axis=0), axis=1))
            print(f"{key}: {len(extrinsics)} poses, trajectory length {distance:.3f} m")


if __name__ == "__main__":
    main()
