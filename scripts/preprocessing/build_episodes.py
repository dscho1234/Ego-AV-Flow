"""Segment the task objects and the hand, and write the episodes in the training format.

For every episode, the objects and the hand are segmented in one frame, either from clicks
(`manual_segmentation=true`, SAM point prompts) or from GroundingDINO boxes (SAM box prompts), and
propagated through the episode with Cutie. The clicks are saved to episode_<i>/segmentation_clicks.json
and reused when the script is run again (`reuse_clicks=true`), so that no window is opened.
The episode is clipped to start at the object frame. The
hand trajectory from `estimate_hand_pose.py` is interpolated to every frame and stored as the 10-D
action / proprioception (camera-frame gripper position, 6D rotation, binary gripper state).

With `update_gripper_only=true`, only the gripper state of existing episodes is recomputed (e.g.
with a different `grasp.max_thumb_index_dist`), without segmenting again.
"""
import json
import os
import pickle

import cv2
import hydra
import numpy as np
import torch
import zarr
from omegaconf import DictConfig
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from egoavflow.preprocessing.common import (
    GROUNDED_SAM_ROOT,
    episode_range,
    load_rgbd_pngs,
    resolve_data_dirs,
    save_array,
    write_episode,
)
from egoavflow.utils import compute_ortho6d_from_rotation_matrix

THUMB_TIP, INDEX_TIP = 4, 8
CLICKS_FILE = "segmentation_clicks.json"
PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 128, 0), (128, 0, 128), (0, 128, 128),
]


def load_segmentation_models(cfg):
    from cutie.utils.get_default_model import get_default_model
    from groundingdino.util.inference import Model
    from hydra.core.global_hydra import GlobalHydra
    from segment_anything import SamPredictor, sam_model_registry

    grounding_dino = Model(
        model_config_path=os.path.join(GROUNDED_SAM_ROOT, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
        model_checkpoint_path=cfg.grounding_dino_checkpoint,
    )
    sam = sam_model_registry["vit_h"](checkpoint=cfg.sam_checkpoint)
    sam.to(device=torch.device("cuda"))
    sam_predictor = SamPredictor(sam)

    # Cutie composes its own Hydra config.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    cutie = get_default_model()
    return grounding_dino, sam_predictor, cutie


@torch.inference_mode()
def propagate_masks(images, cutie, first_frame_masks):
    """Track each first-frame mask through `images` with Cutie. Returns [T, num_masks, H, W] (uint8)."""
    from cutie.inference.inference_core import InferenceCore

    T, H, W = images.shape[:3]
    mask_sequences = np.zeros((T, len(first_frame_masks), H, W), dtype=np.uint8)
    for obj_idx, first_mask in enumerate(first_frame_masks):
        processor = InferenceCore(cutie, cfg=cutie.cfg)
        processor.max_internal_size = -1
        for frame_idx, frame in enumerate(images):
            # Frames are passed to Cutie with swapped color channels, as in the setup used for the paper.
            image = to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).cuda().float()
            if frame_idx == 0:
                output_prob = processor.step(image, torch.from_numpy(first_mask).cuda(), objects=[1])
            else:
                output_prob = processor.step(image)
            mask_sequences[frame_idx, obj_idx] = processor.output_prob_to_mask(output_prob).cpu().numpy().astype(np.uint8)
    return mask_sequences


def masks_from_points(image, points, sam_predictor):
    """One SAM mask per clicked (x, y) point."""
    sam_predictor.set_image(image)
    masks_list = []
    for px, py in points:
        masks, scores, _ = sam_predictor.predict(
            point_coords=np.array([[px, py]]), point_labels=np.array([1]), multimask_output=True
        )
        masks_list.append(masks[np.argmax(scores)])
    return masks_list


def segment_by_clicks(images, names, sam_predictor, start_frame):
    """Interactively pick a frame and click one point per name (with a preview of the SAM masks).

    Returns (frame index, list of [x, y] points), or (None, None) if cancelled.
    """
    T = images.shape[0]
    window_name = f"Click: {', '.join(names)}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    state = {"points": [], "masks": None}
    frame_idx = min(start_frame, T - 1)

    def reset():
        state["points"] = []
        state["masks"] = None

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(state["points"]) < len(names):
            state["points"].append((x, y))
            print(f"Clicked {names[len(state['points']) - 1]}: ({x}, {y})")

    cv2.setMouseCallback(window_name, on_mouse)
    print(f"\nClick one point for each of {names}.")
    print("Left/Right or a/d: change frame | c: confirm | r: retry | q: skip episode")

    while True:
        display = cv2.cvtColor(images[frame_idx], cv2.COLOR_RGB2BGR)
        if state["masks"] is not None:
            overlay = display.copy()
            for i, mask in enumerate(state["masks"]):
                m = mask > 0.5
                overlay[m] = (0.6 * overlay[m] + 0.4 * np.array(PALETTE[i % len(PALETTE)])).astype(np.uint8)
            display = overlay
        for i, (px, py) in enumerate(state["points"]):
            color = PALETTE[i % len(PALETTE)]
            cv2.circle(display, (px, py), 10, color, -1)
            cv2.circle(display, (px, py), 15, color, 2)
            cv2.putText(display, names[i], (px + 20, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        for i, text in enumerate([
            f"Frame: {frame_idx + 1}/{T}",
            f"Clicked: {len(state['points'])}/{len(names)} ({', '.join(names)})",
            "Left/Right or a/d: frame | c: confirm | r: retry | q: skip",
        ]):
            cv2.putText(display, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(window_name, display)

        full_key = cv2.waitKey(30)
        if full_key != -1:
            key = full_key & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                return None, None
            if key == ord("c"):
                if state["masks"] is not None:
                    cv2.destroyAllWindows()
                    return frame_idx, [[int(px), int(py)] for px, py in state["points"]]
                print("Click all points first.")
            elif key == ord("r"):
                reset()
                continue
            elif key == ord("a") or full_key in (81, 65361):
                frame_idx = max(0, frame_idx - 1)
                reset()
            elif key == ord("d") or full_key in (83, 65363):
                frame_idx = min(T - 1, frame_idx + 1)
                reset()

        if len(state["points"]) == len(names) and state["masks"] is None:
            state["masks"] = masks_from_points(images[frame_idx], state["points"], sam_predictor)
            print("Masks generated. Press 'c' to confirm or 'r' to retry.")


def collect_manual_masks(cfg, episode_dir, sam_predictor):
    """Object and hand masks from clicks, which are saved to / reused from `CLICKS_FILE`.

    Returns {"object": (frame, masks), "hand": (frame, masks)}, or None if the clicking was cancelled.
    """
    images, _ = load_rgbd_pngs(episode_dir)
    clicks_path = os.path.join(episode_dir, CLICKS_FILE)
    if cfg.reuse_clicks and os.path.exists(clicks_path):
        with open(clicks_path) as f:
            clicks = json.load(f)
        print(f"Using the clicks saved in {clicks_path}")
    else:
        clicks = {}
        for key, names in [("object", list(cfg.object)), ("hand", ["hand"])]:
            print(f"\n=== {episode_dir}: {key} ===")
            frame, points = segment_by_clicks(images, names, sam_predictor, cfg.manual_start_frame)
            if frame is None:
                return None
            clicks[key] = {"frame": int(frame), "points": points}
        with open(clicks_path, "w") as f:
            json.dump(clicks, f, indent=2)
    if len(clicks["object"]["points"]) != len(cfg.object) or len(clicks["hand"]["points"]) != 1:
        raise ValueError(f"{clicks_path}: expected one click per object {list(cfg.object)} and one for the hand")
    return {
        key: (clicks[key]["frame"], masks_from_points(images[clicks[key]["frame"]], clicks[key]["points"], sam_predictor))
        for key in ["object", "hand"]
    }


def masks_from_boxes(image, boxes, sam_predictor):
    sam_predictor.set_image(image)
    masks_list = []
    for box in boxes:
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        masks_list.append(masks[np.argmax(scores)])
    return masks_list


def gripper_state(fingertips_3d_seq, max_thumb_index_dist):
    """1 (closed) where the thumb tip and the index tip are closer than `max_thumb_index_dist` (m)."""
    dist = np.linalg.norm(fingertips_3d_seq[:, THUMB_TIP] - fingertips_3d_seq[:, INDEX_TIP], axis=-1)
    return (dist <= max_thumb_index_dist).astype(np.int64)


def interpolate_trajectory_data(data, detected_indices, full_length):
    """Linearly interpolate [T_detected, ...] values at `detected_indices` to [full_length, ...]
    (constant extrapolation at both ends)."""
    if len(detected_indices) == 0:
        raise ValueError("no detected frames")
    interpolated = np.zeros((full_length,) + data.shape[1:])
    for i, frame_idx in enumerate(detected_indices):
        if frame_idx < full_length:
            interpolated[frame_idx] = data[i]
    detected_sorted = np.array(sorted(detected_indices))
    for frame_idx in range(full_length):
        if frame_idx in detected_indices:
            continue
        prev_indices = detected_sorted[detected_sorted < frame_idx]
        next_indices = detected_sorted[detected_sorted > frame_idx]
        if len(prev_indices) > 0 and len(next_indices) > 0:
            prev_idx, next_idx = prev_indices[-1], next_indices[0]
            prev_value = data[np.where(detected_indices == prev_idx)[0][0]]
            next_value = data[np.where(detected_indices == next_idx)[0][0]]
            alpha = (frame_idx - prev_idx) / (next_idx - prev_idx)
            interpolated[frame_idx] = (1 - alpha) * prev_value + alpha * next_value
        elif len(prev_indices) > 0:
            interpolated[frame_idx] = data[np.where(detected_indices == prev_indices[-1])[0][0]]
        elif len(next_indices) > 0:
            interpolated[frame_idx] = data[np.where(detected_indices == next_indices[0])[0][0]]
    return interpolated.copy()


def interpolate_rotation_matrix(R_matrices, detected_indices, full_length):
    """SLERP-interpolate [T_detected, 3, 3] rotations to [full_length, 3, 3]."""
    if len(detected_indices) == 0:
        raise ValueError("no detected frames")
    interpolated = np.zeros((full_length, 3, 3))
    rotations = R.from_matrix(R_matrices)
    for i, frame_idx in enumerate(detected_indices):
        if frame_idx < full_length:
            interpolated[frame_idx] = R_matrices[i]
    detected_sorted = np.array(sorted(detected_indices))
    for frame_idx in range(full_length):
        if frame_idx in detected_indices:
            continue
        prev_indices = detected_sorted[detected_sorted < frame_idx]
        next_indices = detected_sorted[detected_sorted > frame_idx]
        if len(prev_indices) > 0 and len(next_indices) > 0:
            prev_idx, next_idx = prev_indices[-1], next_indices[0]
            prev_data_idx = np.where(detected_indices == prev_idx)[0][0]
            next_data_idx = np.where(detected_indices == next_idx)[0][0]
            slerp = Slerp([prev_idx, next_idx], R.concatenate([rotations[prev_data_idx], rotations[next_data_idx]]))
            interpolated[frame_idx] = slerp([frame_idx])[0].as_matrix()
        elif len(prev_indices) > 0:
            interpolated[frame_idx] = R_matrices[np.where(detected_indices == prev_indices[-1])[0][0]]
        elif len(next_indices) > 0:
            interpolated[frame_idx] = R_matrices[np.where(detected_indices == next_indices[0])[0][0]]
    return interpolated.copy()


def fill_hand_pose_before_detection(hand_data, hand_first_frame, start_frame,
                                    max_deviation_position=None, max_deviation_rotation_deg=None,
                                    extrapolation_window=10):
    """Extrapolate the hand trajectory backwards from `hand_first_frame` to `start_frame`.

    The velocity is estimated from the first `extrapolation_window` frames after `hand_first_frame`.
    Positions [T, 3] and rotations [T, 3, 3] are clamped to `max_deviation_position` (m) and
    `max_deviation_rotation_deg` at `start_frame`; other arrays are extrapolated linearly.
    """
    if start_frame >= hand_first_frame:
        return hand_data
    filled = hand_data.copy()
    end_frame = min(hand_first_frame + extrapolation_window, hand_data.shape[0])
    num_after = end_frame - hand_first_frame
    if num_after < 2:
        filled[start_frame:hand_first_frame] = hand_data[hand_first_frame]
        return filled

    if hand_data.ndim == 3 and hand_data.shape[1:] == (3, 3):
        rotations_after = R.from_matrix(hand_data[hand_first_frame:end_frame])
        base = rotations_after[0]
        relative = [rotations_after[i] * base.inv() for i in range(1, num_after)]
        avg_angular_velocity = relative[-1].as_rotvec() / (num_after - 1)
        first_rotation = R.from_matrix(hand_data[hand_first_frame])
        for frame_idx in range(start_frame, hand_first_frame):
            delta = R.from_rotvec(-avg_angular_velocity * (hand_first_frame - frame_idx))
            filled[frame_idx] = (delta * first_rotation).as_matrix()
        if max_deviation_rotation_deg is not None:
            rel_rot = R.from_matrix(filled[start_frame]) * first_rotation.inv()
            if np.degrees(np.linalg.norm(rel_rot.as_rotvec())) > max_deviation_rotation_deg:
                rotvec = rel_rot.as_rotvec()
                if np.linalg.norm(rotvec) > 1e-8:
                    clamped = R.from_rotvec(rotvec / np.linalg.norm(rotvec) * np.radians(max_deviation_rotation_deg))
                    constrained_first = clamped * first_rotation
                else:
                    constrained_first = first_rotation
                slerp = Slerp([start_frame, hand_first_frame], R.concatenate([constrained_first, first_rotation]))
                for frame_idx in range(start_frame, hand_first_frame):
                    filled[frame_idx] = slerp([frame_idx])[0].as_matrix()
        return filled

    avg_velocity = np.mean(np.diff(hand_data[hand_first_frame:end_frame], axis=0), axis=0)
    first_value = hand_data[hand_first_frame]
    for frame_idx in range(start_frame, hand_first_frame):
        filled[frame_idx] = first_value - avg_velocity * (hand_first_frame - frame_idx)
    if max_deviation_position is not None and hand_data.ndim == 2 and hand_data.shape[1] == 3:
        distance = np.linalg.norm(filled[start_frame] - first_value)
        if distance > max_deviation_position:
            direction = (filled[start_frame] - first_value) / (distance + 1e-8)
            constrained_first = first_value + direction * max_deviation_position
            for frame_idx in range(start_frame, hand_first_frame):
                alpha = (frame_idx - start_frame) / (hand_first_frame - start_frame)
                filled[frame_idx] = (1 - alpha) * constrained_first + alpha * first_value
    return filled


def temporal_smoothing(data, sigma):
    smoothed = np.zeros_like(data)
    for i in range(data.shape[1]):
        smoothed[:, i] = gaussian_filter1d(data[:, i], sigma=sigma)
    return smoothed


def align_quaternion_signs(quaternion):
    for i in range(1, len(quaternion)):
        if np.dot(quaternion[i - 1], quaternion[i]) < 0:
            quaternion[i] = -quaternion[i]
    return quaternion


def overlay_masks(image, masks, color_offset=0):
    overlay = image.copy()
    for i, mask in enumerate(masks):
        color = PALETTE[(color_offset + i) % len(PALETTE)]
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array(color)).astype(np.uint8)
    return cv2.addWeighted(image, 0.5, overlay, 0.5, 0)


def save_gripper_video(images, gripper, out_path, fps=20):
    T, H, W, _ = images.shape
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for t in range(T):
        frame = images[t][..., ::-1].copy()
        text, color = ("CLOSE", (0, 0, 255)) if gripper[t] > 0.5 else ("OPEN", (0, 255, 0))
        cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
        cv2.circle(frame, (W - 50, 50), 30, color, -1)
        writer.write(frame)
    writer.release()


def save_mask_video(images, mask_sequences, out_path, fps=20):
    T, H, W, _ = images.shape
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for t in range(T):
        writer.write(overlay_masks(images[t][..., ::-1].copy(), mask_sequences[t]))
    writer.release()


def detect_first_frame(grounding_dino, images, frame_indices, classes, threshold):
    """First frame in `frame_indices` where GroundingDINO finds every class; returns (frame, boxes)."""
    for frame_idx in frame_indices:
        detections = grounding_dino.predict_with_classes(
            image=images[frame_idx], classes=classes, box_threshold=threshold, text_threshold=threshold
        )
        if all(c in set(detections.class_id) for c in range(len(classes))):
            return frame_idx, detections.xyxy
    return None, None


def load_hand_trajectory(episode_dir):
    with open(os.path.join(episode_dir, "data_dict.pkl"), "rb") as f:
        return pickle.load(f)


def build_episode(cfg, data_dir, root, episode_idx, models, manual_masks):
    grounding_dino, sam_predictor, cutie = models
    episode_dir = os.path.join(data_dir, f"episode_{episode_idx}")
    images, depths = load_rgbd_pngs(episode_dir)
    T_full = images.shape[0]
    hand = load_hand_trajectory(episode_dir)

    t_cg = hand["t_cg_opt_depth_trajectory" if cfg.use_depth_rescaled_hand_pose else "t_cg_opt_trajectory"].astype(np.float32)
    R_cg = hand["R_cg_opt_trajectory"].astype(np.float32)
    hand_frames = hand["hand_detected_frame_indices"]
    R_cg_full = interpolate_rotation_matrix(R_cg, hand_frames, T_full)
    t_cg_full = interpolate_trajectory_data(t_cg, hand_frames, T_full)
    joints_3d_full = interpolate_trajectory_data(hand["joint_c_3d_trajectory"].astype(np.float32), hand_frames, T_full)
    hand_first_frame = hand_frames[0]

    if manual_masks is not None:
        object_frame, object_masks = manual_masks["object"]
        hand_frame, hand_masks = manual_masks["hand"]
        hand_first_frame = hand_frame
    else:
        object_frame, object_boxes = detect_first_frame(
            grounding_dino, images, range(T_full), list(cfg.object), cfg.object_box_threshold
        )
        if object_frame is None:
            print(f"Episode {episode_idx}: objects not detected in any frame, skipping")
            return False
        hand_frame, hand_boxes = detect_first_frame(
            grounding_dino, images, hand_frames, ["hand."], cfg.hand_box_threshold
        )
        if hand_frame is None:
            print(f"Episode {episode_idx}: hand not detected in any frame, skipping")
            return False
        object_masks = masks_from_boxes(images[object_frame].copy(), object_boxes, sam_predictor)
        hand_masks = masks_from_boxes(images[hand_frame].copy(), hand_boxes, sam_predictor)

    # The episode starts at the object frame, which is also the first frame of the flow tracking.
    start = object_frame
    print(f"Episode {episode_idx}: hand first detected at {hand_frames[0]}, object frame {object_frame}, hand frame {hand_frame}")

    if object_frame < hand_first_frame:
        R_cg_full = fill_hand_pose_before_detection(R_cg_full, hand_first_frame, object_frame, max_deviation_rotation_deg=30.0)
        t_cg_full = fill_hand_pose_before_detection(t_cg_full, hand_first_frame, object_frame, max_deviation_position=0.15)
        joints_3d_full = fill_hand_pose_before_detection(joints_3d_full, hand_first_frame, object_frame)

    if cfg.temporal_smoothing_sigma > 0:
        quat = align_quaternion_signs(R.from_matrix(R_cg_full).as_quat())
        quat = R.from_quat(temporal_smoothing(quat, sigma=cfg.temporal_smoothing_sigma)).as_quat()
        R_cg_full = R.from_quat(quat).as_matrix()
        t_cg_full = temporal_smoothing(t_cg_full, sigma=cfg.temporal_smoothing_sigma)

    object_mask_sequence = propagate_masks(images[start:], cutie, object_masks)
    T, H, W = object_mask_sequence.shape[0], images.shape[1], images.shape[2]
    if hand_frame > start:
        hand_mask_sequence = np.zeros((T, len(hand_masks), H, W), dtype=np.uint8)
        hand_mask_sequence[hand_frame - start:] = propagate_masks(images[hand_frame:], cutie, hand_masks)
    else:
        hand_mask_sequence = propagate_masks(images[hand_frame:], cutie, hand_masks)[start - hand_frame:]

    object_overlay = overlay_masks(images[object_frame].copy(), object_masks)
    hand_overlay = overlay_masks(images[hand_frame].copy(), hand_masks, color_offset=len(cfg.object))
    if len(hand_masks) != 1:
        print(f"Episode {episode_idx}: expected one hand, got {len(hand_masks)}, skipping")
        cv2.imwrite(os.path.join(episode_dir, "hand_mask_debug.png"), hand_overlay[..., ::-1])
        return False

    # object masks in the order of `cfg.object`, followed by the hand mask
    masks = np.concatenate([object_mask_sequence[:, : len(cfg.object)], hand_mask_sequence], axis=1).astype(np.uint8)

    cv2.imwrite(os.path.join(episode_dir, f"objects_detected_frame_{object_frame}.jpg"), object_overlay[..., ::-1])
    cv2.imwrite(os.path.join(episode_dir, f"hand_detected_frame_{hand_frame}.jpg"), hand_overlay[..., ::-1])
    debug_dir = os.path.join(data_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    if object_overlay.shape[0] == hand_overlay.shape[0]:
        cv2.imwrite(os.path.join(debug_dir, f"detection_e_{episode_idx}.png"), np.hstack([object_overlay, hand_overlay])[..., ::-1])

    # grasping is determined from the first object only
    gripper = gripper_state(joints_3d_full[start:], cfg.grasp.max_thumb_index_dist)
    R_cg_ep = R_cg_full[start:].copy()
    t_cg_ep = t_cg_full[start:].copy()
    assert R_cg_ep.shape[0] == t_cg_ep.shape[0] == gripper.shape[0] == masks.shape[0] == images[start:].shape[0]

    proprioception_se3 = np.concatenate([R_cg_ep, t_cg_ep[:, :, None]], axis=2)
    proprioception_se3 = np.concatenate(
        [proprioception_se3, np.tile(np.array([0, 0, 0, 1.0])[None, None], (proprioception_se3.shape[0], 1, 1))], axis=1
    )
    proprioception = np.concatenate([t_cg_ep, compute_ortho6d_from_rotation_matrix(R_cg_ep), gripper[:, None]], axis=1)

    if cfg.save_video:
        save_gripper_video(images[start:], gripper, os.path.join(episode_dir, "gripper_action.mp4"))
        save_mask_video(images[start:], masks, os.path.join(episode_dir, "mask_overlay.mp4"))

    write_episode(
        root,
        data_dir,
        episode_idx,
        rgb=images[start:],
        depth=depths[start:],
        info=[{"task_description": cfg.task}],
        save_video=cfg.save_video,
        action=proprioception.copy(),
        proprioception=proprioception,
        action_se3=proprioception_se3.copy(),
        proprioception_se3=proprioception_se3,
        hand_detected_frame_indices=hand_frames,
        all_detected_frame_index=np.array(start),
        sam_mask_sequence_multi_obj=masks,
    )
    print(f"Episode {episode_idx}: {len(proprioception)} frames")
    return True


def update_gripper(cfg, data_dir, root, episode_idx):
    """Recompute the gripper state of an existing episode with `cfg.grasp.max_thumb_index_dist`."""
    episode_dir = os.path.join(data_dir, f"episode_{episode_idx}")
    group = root[f"episode_{episode_idx}"]
    action = group["action"][:].copy()
    proprioception = group["proprioception"][:].copy()
    start = int(group["all_detected_frame_index"][()])
    hand = load_hand_trajectory(episode_dir)
    hand_frames = hand["hand_detected_frame_indices"]
    T_full = len([f for f in os.listdir(os.path.join(episode_dir, "color")) if f.endswith(".png")])

    joints_3d_full = interpolate_trajectory_data(hand["joint_c_3d_trajectory"].astype(np.float32), hand_frames, T_full)
    if start < hand_frames[0]:
        joints_3d_full = fill_hand_pose_before_detection(joints_3d_full, hand_frames[0], start)
    gripper = gripper_state(joints_3d_full[start:], cfg.grasp.max_thumb_index_dist)
    assert len(gripper) == len(action) == group["camera_0/rgb"].shape[0]

    action[:, -1] = gripper
    proprioception[:, -1] = gripper
    save_array(group, "action", action)
    save_array(group, "proprioception", proprioception)
    if cfg.save_video:
        save_gripper_video(group["camera_0/rgb"][:], gripper, os.path.join(episode_dir, "gripper_action_updated.mp4"))
    print(f"Episode {episode_idx}: gripper state updated")


@hydra.main(version_base=None, config_path="../../config/preprocessing", config_name="build_episodes")
def main(cfg: DictConfig):
    data_dirs = resolve_data_dirs(cfg.data_dirs)

    if cfg.update_gripper_only:
        for data_dir in data_dirs:
            root = zarr.open(data_dir, mode="a")
            for episode_idx in episode_range(data_dir, cfg.episode_start, cfg.episode_end):
                if f"episode_{episode_idx}" in root and "action" in root[f"episode_{episode_idx}"]:
                    update_gripper(cfg, data_dir, root, episode_idx)
        return

    models = load_segmentation_models(cfg)
    for data_dir in data_dirs:
        root = zarr.open(data_dir, mode="a")
        episodes = [
            i for i in episode_range(data_dir, cfg.episode_start, cfg.episode_end)
            if os.path.exists(os.path.join(data_dir, f"episode_{i}", "data_dict.pkl"))
        ]

        # Collect the clicks of all episodes first, so the rest runs unattended.
        manual_masks = {}
        if cfg.manual_segmentation:
            for episode_idx in episodes:
                masks = collect_manual_masks(cfg, os.path.join(data_dir, f"episode_{episode_idx}"), models[1])
                if masks is not None:
                    manual_masks[episode_idx] = masks

        succeeded, failed = [], []
        for episode_idx in tqdm(episodes, desc=os.path.basename(data_dir)):
            if cfg.manual_segmentation and episode_idx not in manual_masks:
                failed.append(episode_idx)
                continue
            try:
                ok = build_episode(cfg, data_dir, root, episode_idx, models, manual_masks.get(episode_idx))
            except Exception:
                import traceback

                traceback.print_exc()
                ok = False
            (succeeded if ok else failed).append(episode_idx)
        print(f"{data_dir}: {len(succeeded)} episodes written, failed: {failed}")


if __name__ == "__main__":
    main()
