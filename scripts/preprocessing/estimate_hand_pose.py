"""Estimate the right-hand gripper pose in every recorded frame with HaMeR.

For every frame, the person is detected with ViTDet, the hand keypoints with ViTPose, and the MANO
hand with HaMeR. The wrist pose is refined by minimizing the reprojection error of the 21 hand
joints, and its metric translation is recovered from the depth at the wrist. A parallel-gripper
frame is attached to the hand (origin between the thumb and index-finger bases, axes from a plane
fit to the thumb and index joints).

Writes episode_<i>/data_dict.pkl (frames in which a hand was detected) and episode_<i>/video.mp4.
"""
import os
import pickle
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from scipy.optimize import least_squares
from tqdm import tqdm

from egoavflow.common.utility.projection import fill_depth_zeros
from egoavflow.preprocessing.common import HAMER_ROOT, episode_range, load_rgbd_pngs, resolve_data_dirs

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
VITDET_CHECKPOINT = (
    "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
)


def detect_spikes_in_trajectory(trajectory, threshold_factor=3.0, window_size=5):
    """Frame indices of spikes in a (N, 3) translation trajectory (median absolute deviation test)."""
    if len(trajectory) < window_size + 2:
        return []
    trajectory = np.array(trajectory)
    N = len(trajectory)
    spike_indices = []

    # large jumps between consecutive frames
    diff_magnitudes = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    median_diff = np.median(diff_magnitudes)
    mad = np.median(np.abs(diff_magnitudes - median_diff))
    if mad > 0:
        threshold = median_diff + threshold_factor * mad
    else:
        threshold = median_diff + threshold_factor * np.std(diff_magnitudes)
    for i in range(len(diff_magnitudes)):
        if diff_magnitudes[i] > threshold:
            spike_indices.append(i + 1)

    # outliers with respect to the local median
    for i in range(N):
        local_indices = list(range(max(0, i - window_size), min(N, i + window_size + 1)))
        local_indices.remove(i)
        if len(local_indices) == 0:
            continue
        local_median = np.median(trajectory[local_indices], axis=0)
        dist = np.linalg.norm(trajectory[i] - local_median)
        local_dists = [np.linalg.norm(trajectory[j] - local_median) for j in local_indices]
        local_median_dist = np.median(local_dists)
        local_mad = np.median(np.abs(np.array(local_dists) - local_median_dist))
        if local_mad > 0:
            local_threshold = local_median_dist + threshold_factor * local_mad
        else:
            local_std = np.std(local_dists) if len(local_dists) > 1 else 0
            local_threshold = local_median_dist + threshold_factor * local_std if local_std > 0 else float("inf")
        if dist > local_threshold and i not in spike_indices:
            spike_indices.append(i)

    return sorted(spike_indices)


def camera_to_wrist(P_c, R_cw, t_cw):
    return (R_cw.T @ (P_c - t_cw[None, :]).T).T


def build_gripper_frame(joints, R_hand, eps=1e-6):
    """Parallel-gripper frame from the 21 MANO joints.

    The origin is the midpoint of the thumb (2) and index (5) bases, z is the normal of a plane fit to
    the thumb and index joints (2-8) oriented along the z-axis of `R_hand`, y points from the index base
    to the thumb base (projected onto the plane), and x = y x z.
    """
    J = torch.as_tensor(joints, dtype=torch.float32)
    thumb_base = J[2]
    index_base = J[5]
    origin = (thumb_base + index_base) / 2

    plane_points = torch.stack([J[2], J[3], J[4], J[5], J[6], J[7], J[8]]).cpu().numpy()
    centered = plane_points - np.mean(plane_points, axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    plane_normal = Vt[-1, :]
    plane_normal = plane_normal / (np.linalg.norm(plane_normal) + eps)

    R_hand_z = torch.as_tensor(np.asarray(R_hand)[:, 2], dtype=torch.float32).to(J.device)
    if torch.dot(torch.from_numpy(plane_normal).to(J.device).float(), R_hand_z) < 0:
        plane_normal = -plane_normal
    plane_normal = torch.from_numpy(plane_normal).to(J.device).float()

    y = J[2] - J[5]
    y = y / (y.norm() + eps)
    y = y - torch.dot(y, plane_normal) * plane_normal
    y = y / (y.norm() + eps)
    z = plane_normal
    x = torch.cross(y, z)
    x = x / (x.norm() + eps)

    return origin.cpu().numpy(), torch.stack([x, y, z], dim=1).cpu().numpy()


def project(K, pts_cam):
    pts_cam = np.asarray(pts_cam)
    if pts_cam.ndim == 1:
        pts_cam = pts_cam.reshape(1, -1)
    uvw = (K @ pts_cam.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def optimize_R_cw_t_cw(P_c, uv, K, R_cw, t_cw):
    """Refine the wrist pose (wrist -> camera) by minimizing the reprojection error of the joints."""
    P_c, uv, K = np.array(P_c), np.array(uv), np.array(K)
    t_cw = t_cw.reshape(3)
    P_w = camera_to_wrist(P_c, R_cw, t_cw)
    r0, _ = cv2.Rodrigues(R_cw)

    def residual(x):
        R_var, _ = cv2.Rodrigues(x[:3])
        P_cam = (R_var @ P_w.T).T + x[3:6][None, :]
        return (project(K, P_cam) - uv).ravel()

    res = least_squares(residual, np.hstack([r0.ravel(), t_cw.copy()]), method="lm", xtol=1e-8, ftol=1e-8, gtol=1e-8)
    R_opt, _ = cv2.Rodrigues(res.x[:3])
    return R_opt, res.x[3:6]


def optimize_R_cw_fixed_t(P_c, uv, K, R_cw, t_cw_fixed):
    """Refine only the wrist rotation for a fixed wrist translation."""
    P_c, uv, K = np.array(P_c), np.array(uv), np.array(K)
    t_cw_fixed = np.array(t_cw_fixed).reshape(3)
    P_w = camera_to_wrist(P_c, R_cw, t_cw_fixed)
    r0, _ = cv2.Rodrigues(R_cw)

    def residual(x):
        R_var, _ = cv2.Rodrigues(x[:3])
        P_cam = (R_var @ P_w.T).T + t_cw_fixed[None, :]
        return (project(K, P_cam) - uv).ravel()

    res = least_squares(residual, r0.ravel(), method="lm", xtol=1e-8, ftol=1e-8, gtol=1e-8)
    R_opt, _ = cv2.Rodrigues(res.x[:3])
    return R_opt


def gripper_pose_in_camera(joints_cam, R_cw, t_cw):
    """Gripper pose (camera frame) for the hand joints moved rigidly to the wrist pose (R_cw, t_cw)."""
    joints_translated = (t_cw - joints_cam[0]) + joints_cam
    joints_w = camera_to_wrist(joints_translated, R_cw, t_cw)
    t_wg, R_wg = build_gripper_frame(joints_w, R_hand=np.eye(3))
    return (R_cw @ t_wg.T) + t_cw, R_cw @ R_wg


def draw_axes(frame, K, origin, R, length_px=50, thickness=3):
    length = length_px * origin[2] / K[0, 0]
    pts = project(K, np.stack([np.zeros(3), R[:, 0] * length, R[:, 1] * length, R[:, 2] * length]) + origin)
    o = tuple(pts[0].astype(int))
    for p, color in zip(pts[1:], [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        cv2.line(frame, o, tuple(p.astype(int)), color, thickness)


class HandPoseEstimator:
    def __init__(self, cfg):
        sys.path.insert(0, HAMER_ROOT)
        from detectron2.config import LazyConfig

        import hamer
        from hamer.models import DEFAULT_CHECKPOINT, load_hamer
        from hamer.utils.renderer import Renderer
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        from vitpose_model import ViTPoseModel

        self.cfg = cfg
        self.device = torch.device("cuda")
        self.model, self.model_cfg = load_hamer(DEFAULT_CHECKPOINT)
        self.model = self.model.to(self.device).eval()

        detectron2_cfg = LazyConfig.load(str(Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"))
        detectron2_cfg.train.init_checkpoint = VITDET_CHECKPOINT
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        self.keypoint_detector = ViTPoseModel(self.device)
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)

    @torch.inference_mode()
    def process_episode(self, episode_dir):
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.utils import recursive_to
        from hamer.utils.renderer import cam_crop_to_full

        cfg = self.cfg
        K = np.array(cfg.camera.intrinsics, dtype=np.float32)
        rgb_frames, depth_frames = load_rgbd_pngs(episode_dir)
        h, w = rgb_frames.shape[1:3]
        writer = None
        if cfg.save_video:
            writer = cv2.VideoWriter(str(episode_dir / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))

        traj = {k: [] for k in [
            "joint_c_3d_trajectory", "joint_pixel_2d_trajectory",
            "R_cw_opt_trajectory", "t_cw_opt_trajectory", "R_cw_opt_depth_trajectory", "t_cw_opt_depth_trajectory",
            "R_cg_opt_trajectory", "t_cg_opt_trajectory", "R_cg_opt_depth_trajectory", "t_cg_opt_depth_trajectory",
            "hand_detected_frame_indices", "bbox_trajectory", "hand_side_trajectory",
        ]}

        for frame_idx, img_rgb in enumerate(tqdm(rgb_frames, desc=episode_dir.name)):
            img_cv2 = img_rgb[:, :, ::-1].copy()
            depth_frame = fill_depth_zeros(depth_frames[frame_idx].astype(np.float32) / 1000.0)

            det_instances = self.detector(img_cv2)["instances"]
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores = det_instances.scores[valid_idx].cpu().numpy()
            vitposes_out = self.keypoint_detector.predict_pose(
                img_rgb, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
            )

            bboxes, is_right = [], []
            for vitposes in vitposes_out:
                for keyp, right_flag, side in [(vitposes["keypoints"][-42:-21], 0, "left"),
                                               (vitposes["keypoints"][-21:], 1, "right")]:
                    valid = keyp[:, 2] > 0.5
                    if sum(valid) > 3 and cfg.hand_side in ("auto", side):
                        bboxes.append([keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()])
                        is_right.append(right_flag)
            if len(bboxes) == 0:
                continue

            boxes = np.stack(bboxes)
            right = np.stack(is_right)
            if len(bboxes) > 1:
                print(f"Warning: {len(bboxes)} hands detected at frame {frame_idx}; keeping the first right hand")
                boxes = boxes[right == 1][:1]
                right = right[right == 1][:1]

            dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right, rescale_factor=cfg.rescale_factor)
            batch = recursive_to(next(iter(torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False))), self.device)
            out = self.model(batch)

            pred_cam = out["pred_cam"].clone()
            pred_cam[:, 1] = (2 * batch["right"] - 1) * pred_cam[:, 1]
            scaled_f = torch.tensor(K[0, 0])
            pred_cam_t_full = cam_crop_to_full(
                pred_cam, batch["box_center"].float(), batch["box_size"].float(), batch["img_size"].float(), scaled_f
            ).cpu().numpy()

            # a single right hand per frame
            i = 0
            joints_2d = out["pred_keypoints_2d"].cpu().numpy()[i].copy()
            joints_3d = out["pred_keypoints_3d"].cpu().numpy()[i].copy()
            if not bool(batch["right"][i]):
                joints_2d[:, 0] *= -1
                joints_3d[:, 0] *= -1
            joints_2d = joints_2d * batch["box_size"][i].cpu().numpy() + batch["box_center"][i].cpu().numpy()
            # camera-frame joints; the scale is only valid up to the projection
            joints_cam = joints_3d + pred_cam_t_full[i]
            R_hand = out["pred_mano_params"]["global_orient"].cpu().numpy()[i][0]

            R_cw_opt, t_cw_opt = optimize_R_cw_t_cw(joints_cam, joints_2d, K, R_hand, joints_cam[0])
            t_cg_opt, R_cg_opt = gripper_pose_in_camera(joints_cam, R_cw_opt, t_cw_opt)

            # metric scale from the depth at the projected wrist (plus the wrist thickness)
            u, v = np.round(project(K, t_cw_opt[None, :])[0]).astype(int)
            u = np.clip(u, 0, depth_frame.shape[1] - 1)
            v = np.clip(v, 0, depth_frame.shape[0] - 1)
            depth_value = depth_frame[v, u] + cfg.wrist_depth_bias
            t_cw_opt_depth = t_cw_opt * (depth_value / t_cw_opt[2])
            R_cw_opt_depth = optimize_R_cw_fixed_t(joints_cam, joints_2d, K, R_cw_opt, t_cw_opt_depth)
            t_cg_opt_depth, R_cg_opt_depth = gripper_pose_in_camera(joints_cam, R_cw_opt_depth, t_cw_opt_depth)

            for key, value in [
                ("joint_c_3d_trajectory", joints_cam), ("joint_pixel_2d_trajectory", joints_2d),
                ("R_cw_opt_trajectory", R_cw_opt), ("t_cw_opt_trajectory", t_cw_opt),
                ("R_cw_opt_depth_trajectory", R_cw_opt_depth), ("t_cw_opt_depth_trajectory", t_cw_opt_depth),
                ("R_cg_opt_trajectory", R_cg_opt), ("t_cg_opt_trajectory", t_cg_opt),
                ("R_cg_opt_depth_trajectory", R_cg_opt_depth), ("t_cg_opt_depth_trajectory", t_cg_opt_depth),
                ("hand_detected_frame_indices", frame_idx), ("bbox_trajectory", boxes.copy()),
                ("hand_side_trajectory", right.copy()),
            ]:
                traj[key].append(value)

            if writer is not None:
                verts = out["pred_vertices"][i].detach().cpu().numpy()
                verts[:, 0] = (2 * batch["right"][i].cpu().numpy() - 1) * verts[:, 0]
                cam_view = self.renderer.render_rgba_multiple(
                    [verts], cam_t=[pred_cam_t_full[i]], render_res=batch["img_size"][i].float(),
                    is_right=[batch["right"][i].cpu().numpy()],
                    mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1), focal_length=scaled_f,
                )
                overlay = img_rgb.astype(np.float32) / 255.0
                overlay = overlay * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
                frame_out = np.ascontiguousarray((overlay * 255).astype(np.uint8))
                x1, y1, x2, y2 = boxes[0].astype(int)
                cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_axes(frame_out, K, t_cg_opt_depth, R_cg_opt)
                writer.write(frame_out[:, :, ::-1])

        if writer is not None:
            writer.release()
        if len(traj["t_cw_opt_trajectory"]) == 0:
            print(f"No hand detected in {episode_dir}")
            return

        # drop frames whose wrist translation jumps
        spike_indices = detect_spikes_in_trajectory(
            traj["t_cw_opt_trajectory"], threshold_factor=cfg.spike_threshold_factor, window_size=cfg.spike_window_size
        )
        if len(spike_indices) > 0:
            print(f"Removing {len(spike_indices)} spike frames: {spike_indices}")
            keep = np.ones(len(traj["t_cw_opt_trajectory"]), dtype=bool)
            keep[spike_indices] = False
            traj = {k: [x for x, valid in zip(v, keep) if valid] for k, v in traj.items()}

        data_dict = {k: np.stack(v) if k != "hand_detected_frame_indices" else np.array(v) for k, v in traj.items()}
        data_dict["spike_indices"] = np.array(spike_indices)
        with open(episode_dir / "data_dict.pkl", "wb") as f:
            pickle.dump(data_dict, f)


@hydra.main(version_base=None, config_path="../../config/preprocessing", config_name="hand_pose")
def main(cfg: DictConfig):
    data_dirs = resolve_data_dirs(cfg.data_dirs)
    # HaMeR and ViTPose locate their weights (_DATA/) and configs relative to the HaMeR repository.
    os.chdir(HAMER_ROOT)
    estimator = HandPoseEstimator(cfg)
    for data_dir in data_dirs:
        for episode_idx in episode_range(data_dir, cfg.episode_start, cfg.episode_end):
            estimator.process_episode(Path(data_dir) / f"episode_{episode_idx}")


if __name__ == "__main__":
    main()
