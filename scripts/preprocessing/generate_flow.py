"""Track the query points (and points sampled on the object masks) and lift them to 3D.

1. The query points clicked in episode 0 (`annotate_query_points.py`) are transferred to the first
   frame of every other episode by DIFT feature matching (Stable Diffusion 2.1 features).
2. The points are tracked with online CoTracker3 at 256x256 and lifted to 3D with the recorded depth
   (camera frame).
3. A point is marked invisible when its depth deviates from the mean depth of its object mask by
   more than `mask_point_tracking_threshold`.

Writes per episode:
    task_description              (1,)                language description (used as the DIFT prompt)
    dift_points                   (N, 2)              query points in the first frame (640x480 pixels)
    dift_pixel_tracking_sequence  (N, T, 3)           2D tracks in the 256x256 tracking image + visibility
    dift_point_tracking_sequence  (N, T, 4)           3D tracks (camera frame, m) + visibility
    mask_pixel_tracking_sequence  (K, M, T, 3)        2D tracks of M points sampled on each object mask
    mask_point_tracking_sequence  (K, M, T, 4)        their 3D tracks + visibility
    intrinsics (3, 3), rgbd_hw (2,)
"""
import os

import cv2
import hydra
import matplotlib
import numcodecs
import numpy as np
import torch
import zarr
from omegaconf import DictConfig
from tqdm import tqdm

from egoavflow.common.utility.projection import fill_depth_zeros, get_pixel_queries
from egoavflow.common.utility.viz import viz_point_tracking_flow
from egoavflow.common.utility.zarr import parallel_reading
from egoavflow.preprocessing.common import episode_range, resolve_data_dirs, save_array
from egoavflow.preprocessing.dift import SDFeaturizer, get_corresponding_points, get_dift_features

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIFT_IMAGE_SIZE = 768


def track_points(tracker, video, points_xy):
    """Online CoTracker3 over the whole video. Returns tracks [N, T, 2] and visibility [N, T]."""
    queries = torch.zeros((1, len(points_xy), 3), device=video.device)
    queries[0, :, 1] = torch.from_numpy(points_xy[:, 0]).float().to(video.device)
    queries[0, :, 2] = torch.from_numpy(points_xy[:, 1]).float().to(video.device)
    tracker(video_chunk=video, is_first_step=True, queries=queries)
    for ind in range(0, video.shape[1] - tracker.step, tracker.step):
        pred_tracks, pred_visibility = tracker(video_chunk=video[:, ind: ind + tracker.step * 2], one_frame=False)
    tracks = pred_tracks[0].permute(1, 0, 2).cpu().numpy()
    visibles = pred_visibility[0].permute(1, 0).cpu().numpy()
    return tracks, visibles


def lift_to_3d(tracks, depths, intrinsics):
    """Back-project [N, T, 2] pixel tracks with per-frame depth maps into camera-frame points [N, T, 3]."""
    traj_3d = []
    for t in range(tracks.shape[1]):
        pts = get_pixel_queries(
            torch.from_numpy(tracks[:, t]).float(),
            torch.from_numpy(depths[t]).float(),
            torch.from_numpy(intrinsics).float(),
            torch.eye(4),
        ).numpy()
        traj_3d.append(pts[..., -3:])
    return np.stack(traj_3d, axis=1)


def sample_mask_point_clouds(masks, depths, intrinsics, num_points):
    """`num_points` random 3D points on every object mask of every frame: [K, T, num_points, 3]."""
    T, num_obj = masks.shape[:2]
    clouds = np.zeros((num_obj, T, num_points, 3), dtype=np.float32)
    for t in range(T):
        for obj_idx in range(num_obj):
            ys, xs = np.where(masks[t, obj_idx] > 0)
            if len(xs) == 0:
                continue
            valid = depths[t][ys, xs] > 0
            if not np.any(valid):
                continue
            xs, ys = xs[valid], ys[valid]
            idx = np.random.choice(len(xs), num_points, replace=len(xs) < num_points)
            xy = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
            queries = get_pixel_queries(
                torch.from_numpy(xy).float(),
                torch.from_numpy(depths[t]).float().unsqueeze(0),
                torch.from_numpy(intrinsics).float().unsqueeze(0),
                torch.eye(4).unsqueeze(0),
            )
            clouds[obj_idx, t] = queries[:, 1:4].numpy()
    return clouds


def filter_by_mask_depth(traj_3d, mask_clouds_t, threshold):
    """Visibility mask of the points [N, 3] whose depth is within `threshold` of the mask's mean depth."""
    valid_points = mask_clouds_t[np.any(mask_clouds_t != 0, axis=1)]
    if len(valid_points) == 0:
        return None
    return np.abs(traj_3d[:, 2] - np.mean(valid_points[:, 2])) <= threshold


def rescale_points(points, sx, sy):
    """Scale pixel coordinates in place of a copy, keeping the dtype (integer points are truncated)."""
    points = points.copy()
    points[:, 0] = points[:, 0] * sx
    points[:, 1] = points[:, 1] * sy
    return points


class DIFTTransfer:
    """Transfers the query points of the first frame of episode 0 to the first frame of other episodes."""

    def __init__(self, cfg, root, tracking_hw, rgbd_hw):
        torch.manual_seed(cfg.seed)
        self.featurizer = SDFeaturizer(cfg.sd_model_path)
        self.ensemble_size = cfg.dift_ensemble_size
        self.rgbd_hw = rgbd_hw
        self.prompt = root["episode_0/info"][0]["task_description"]
        frame0 = root["episode_0/camera_0/rgb"][0]
        frame0 = cv2.resize(frame0, (tracking_hw[1], tracking_hw[0]))
        h, w = rgbd_hw
        self.source_points = rescale_points(root["episode_0/dift_points"][:], DIFT_IMAGE_SIZE / w, DIFT_IMAGE_SIZE / h)
        self.source_ft, self.source_img = get_dift_features(
            frame0, self.featurizer, self.prompt, self.ensemble_size, DIFT_IMAGE_SIZE
        )

    def __call__(self, first_frame, viz_path):
        target_ft, target_img = get_dift_features(first_frame, self.featurizer, self.prompt, self.ensemble_size, DIFT_IMAGE_SIZE)
        points = get_corresponding_points(self.source_ft, self.source_points, target_ft, DIFT_IMAGE_SIZE)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(self.source_img)
        axes[1].imshow(target_img)
        axes[0].set_title("Episode 0")
        axes[1].set_title("This episode")
        for idx, (src, tgt) in enumerate(zip(self.source_points, points)):
            color = ["r", "g", "b"][idx % 3]
            axes[0].scatter(src[0], src[1], c=color, s=80)
            axes[1].scatter(tgt[0], tgt[1], c=color, s=80)
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()

        h, w = self.rgbd_hw
        return rescale_points(points, w / DIFT_IMAGE_SIZE, h / DIFT_IMAGE_SIZE)


@torch.inference_mode()
def process_episode(cfg, data_dir, root, episode_idx, tracker, dift_transfer):
    key = f"episode_{episode_idx}"
    episode = root[key]
    episode_dir = os.path.join(data_dir, key)
    device = torch.device("cuda")
    # DIFT noise and the mask point sampling are random
    np.random.seed(cfg.seed + episode_idx)
    torch.manual_seed(cfg.seed + episode_idx)
    rgbd_hw = [cfg.camera.height, cfg.camera.width]
    tracking_hw = list(cfg.tracking_hw)
    rgbd_K = np.array(cfg.camera.intrinsics)
    tracking_K = rgbd_K.copy()
    tracking_K[0, :] *= (tracking_hw[1] - 1) / (rgbd_hw[1] - 1)
    tracking_K[1, :] *= (tracking_hw[0] - 1) / (rgbd_hw[0] - 1)

    task_description = episode["info"][0]["task_description"]
    task_arr = episode.create_dataset(
        "task_description", shape=(1,), dtype=object, object_codec=numcodecs.VLenUTF8(), overwrite=True
    )
    task_arr[0] = task_description

    frames = parallel_reading(group=episode["camera_0"], array_name="rgb")
    depths = fill_depth_zeros(parallel_reading(group=episode["camera_0"], array_name="depth").astype(np.float32) / 1000.0)
    tracking_frames = np.array([cv2.resize(image, (tracking_hw[1], tracking_hw[0])) for image in frames])
    tracking_depths = np.array([cv2.resize(depth, (tracking_hw[1], tracking_hw[0])) for depth in depths])

    if episode_idx == 0:
        dift_points = episode["dift_points"][:]
    else:
        dift_points = dift_transfer(tracking_frames[0], os.path.join(episode_dir, f"episode_{episode_idx}_points.png"))
        save_array(episode, "dift_points", dift_points)

    video = torch.tensor(tracking_frames).permute(0, 3, 1, 2)[None].float().to(device)
    points_xy = np.zeros_like(dift_points, dtype=np.float32)
    points_xy[:, 0] = dift_points[:, 0] * (tracking_hw[1] / rgbd_hw[1])
    points_xy[:, 1] = dift_points[:, 1] * (tracking_hw[0] / rgbd_hw[0])
    tracks, visibles = track_points(tracker, video, points_xy)
    tracks[:, :, 0] = tracks[:, :, 0].clip(0, tracking_hw[1] - 1)
    tracks[:, :, 1] = tracks[:, :, 1].clip(0, tracking_hw[0] - 1)
    pixel_tracks = np.concatenate([tracks, np.expand_dims(visibles, axis=-1).astype(np.float32)], axis=-1)
    traj_3d = np.concatenate(
        [lift_to_3d(tracks, tracking_depths, tracking_K), np.expand_dims(visibles, axis=-1).astype(np.float32)], axis=-1
    )
    save_array(episode, "dift_pixel_tracking_sequence", pixel_tracks)
    save_array(episode, "intrinsics", rgbd_K)
    save_array(episode, "rgbd_hw", rgbd_hw)

    masks = episode["sam_mask_sequence_multi_obj"][:]
    num_obj = masks.shape[1] - 1 if masks.shape[1] > 1 else masks.shape[1]  # drop the hand
    mask_clouds = sample_mask_point_clouds(masks[:, :num_obj], depths, rgbd_K, cfg.num_points_per_mask)

    # query points belong to the objects according to `query_point_object_ranges`
    pixel_tracks_viz = pixel_tracks.copy()
    for t in range(traj_3d.shape[1]):
        for obj_idx in range(num_obj):
            start, end = cfg.query_point_object_ranges[obj_idx]
            keep = filter_by_mask_depth(traj_3d[start:end, t, :3], mask_clouds[obj_idx, t], cfg.mask_point_tracking_threshold)
            if keep is not None:
                traj_3d[start:end, t, 3][~keep] = 0.0
                pixel_tracks_viz[start:end, t, 2][~keep] = 0.0
    save_array(episode, "dift_point_tracking_sequence", traj_3d)

    if cfg.track_mask:
        first_masks = np.array([
            cv2.resize(m.astype(np.uint8), (tracking_hw[1], tracking_hw[0]), interpolation=cv2.INTER_NEAREST)
            for m in masks[0]
        ])
        T = traj_3d.shape[1]
        mask_tracks_3d = np.zeros((num_obj, cfg.num_mask_tracking_points, T, 4), dtype=np.float32)
        mask_tracks_2d = []
        for obj_idx in range(num_obj):
            ys, xs = np.where(first_masks[obj_idx] > 0)
            if len(xs) == 0:
                continue
            idx = np.random.choice(len(xs), cfg.num_mask_tracking_points, replace=len(xs) < cfg.num_mask_tracking_points)
            tracks, visibles = track_points(tracker, video, np.stack([xs[idx], ys[idx]], axis=1))
            tracks[:, :, 0] = tracks[:, :, 0].clip(0, tracking_hw[1] - 1)
            tracks[:, :, 1] = tracks[:, :, 1].clip(0, tracking_hw[0] - 1)
            obj_3d = np.concatenate([lift_to_3d(tracks, tracking_depths, tracking_K), visibles[:, :, None]], axis=-1)
            for t in range(T):
                keep = filter_by_mask_depth(obj_3d[:, t, :3], mask_clouds[obj_idx, t], cfg.mask_point_tracking_threshold)
                if keep is not None:
                    obj_3d[:, t, 3][~keep] = 0.0
            mask_tracks_3d[obj_idx] = obj_3d
            mask_tracks_2d.append(np.concatenate([tracks, visibles[:, :, None]], axis=-1))
        save_array(episode, "mask_point_tracking_sequence", mask_tracks_3d)
        save_array(episode, "mask_pixel_tracking_sequence", np.stack(mask_tracks_2d))
        if cfg.save_video and len(mask_tracks_2d) > 0:
            viz_point_tracking_flow(
                tracking_frames, np.concatenate(mask_tracks_2d, axis=0),
                os.path.join(episode_dir, f"episode_{episode_idx}_mask_point_tracking.mp4"), output_format="mp4",
            )

    if cfg.save_video:
        viz_point_tracking_flow(
            tracking_frames, pixel_tracks_viz,
            os.path.join(episode_dir, f"episode_{episode_idx}_query_point_tracking.mp4"), output_format="mp4",
        )
    torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="../../config/preprocessing", config_name="flow")
def main(cfg: DictConfig):
    tracker = None
    for data_dir in resolve_data_dirs(cfg.data_dirs):
        root = zarr.open(data_dir, mode="a")
        if "episode_0" not in root or "dift_points" not in root["episode_0"]:
            raise RuntimeError(f"{data_dir}/episode_0 has no query points; run annotate_query_points.py first")
        episodes = [
            i for i in episode_range(data_dir, cfg.episode_start, cfg.episode_end)
            if f"episode_{i}" in root and "sam_mask_sequence_multi_obj" in root[f"episode_{i}"]
        ]
        if tracker is None:
            from cotracker.predictor import CoTrackerOnlinePredictor

            tracker = CoTrackerOnlinePredictor(checkpoint=cfg.cotracker_checkpoint).to("cuda")
        dift_transfer = None
        if any(i > 0 for i in episodes):
            dift_transfer = DIFTTransfer(cfg, root, list(cfg.tracking_hw), [cfg.camera.height, cfg.camera.width])
        for episode_idx in tqdm(episodes, desc=os.path.basename(data_dir)):
            process_episode(cfg, data_dir, root, episode_idx, tracker, dift_transfer)


if __name__ == "__main__":
    main()
