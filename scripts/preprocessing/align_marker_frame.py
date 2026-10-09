"""Express the camera trajectory in the frame of a ChArUco board placed in the scene.

The board is detected in every frame of the recording, and a GTSAM pose graph combines the
DROID-SLAM odometry (relative poses between consecutive frames) with the detections of the
reference marker (absolute poses) into the camera trajectory in the marker frame.

Writes (from the episode start):
    T_mc_opt_droid                   [T, 4, 4]  marker <- camera, optimized (used for training)
    T_mc_from_first_detection_droid  [T, 4, 4]  marker <- camera, anchored at the first detection
    T_wm_opt_droid                   [1, 4, 4]  SLAM world <- marker
"""
import os

import cv2
import gtsam
import hydra
import matplotlib
import numpy as np
import zarr
from omegaconf import DictConfig

from egoavflow.preprocessing.common import episode_range, load_rgbd_pngs, resolve_data_dirs, save_array

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------- SE(3) helpers ----------
def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def mat4_to_pose3(T):
    return gtsam.Pose3(gtsam.Rot3(T[:3, :3]), gtsam.Point3(*T[:3, 3]))


def pose3_to_mat4(P):
    T = np.eye(4)
    T[:3, :3] = np.array(P.rotation().matrix())
    T[:3, 3] = np.array([P.x(), P.y(), P.z()])
    return T


def inv_T(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def ensure_SE3(T):
    U, _, Vt = np.linalg.svd(T[:3, :3])
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    out = T.copy()
    out[:3, :3] = Rn
    return out


def so3_log(R):
    theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-12:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
    return theta * w


def se3_exp(xi):
    w, v = xi[:3], xi[3:]
    theta = np.linalg.norm(w)
    T = np.eye(4)
    if theta < 1e-12:
        T[:3, 3] = v
        return T
    K = skew(w / theta)
    T[:3, :3] = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    J = np.eye(3) + (1 - np.cos(theta)) / (theta ** 2) * skew(w) + ((theta - np.sin(theta)) / (theta ** 3)) * (skew(w) @ skew(w))
    T[:3, 3] = J @ v
    return T


def se3_mean(T_list, iters=15):
    """Left-invariant mean (small-error approximation of the SE(3) log)."""
    Mu = T_list[0].copy()
    for _ in range(iters):
        xi_sum = np.zeros(6)
        for T in T_list:
            E = inv_T(Mu) @ T
            xi_sum += np.hstack([so3_log(E[:3, :3]), E[:3, 3]])
        xi = xi_sum / len(T_list)
        if np.linalg.norm(xi) < 1e-10:
            break
        Mu = ensure_SE3(Mu @ se3_exp(xi))
    return Mu


# ---------- ChArUco detection ----------
def create_charuco_board(cfg):
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.board.aruco_dict))
    return cv2.aruco.CharucoBoard(
        (cfg.board.squares_x, cfg.board.squares_y), cfg.board.square_size, cfg.board.marker_size, aruco_dict
    )


def make_detector_params():
    p = cv2.aruco.DetectorParameters()
    try:
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    except AttributeError:
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 30
    p.cornerRefinementMinAccuracy = 0.01
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 23
    p.adaptiveThreshWinSizeStep = 10
    return p


def get_board_corners_3d(board):
    pts = board.getChessboardCorners() if hasattr(board, "getChessboardCorners") else board.chessboardCorners
    return np.asarray(pts, dtype=np.float32).reshape(-1, 3)


def detect_marker_poses(gray, board, detector, K, dist_coeffs):
    """Board pose from the ChArUco corners (PnP) and the resulting pose of every detected marker.

    Returns {marker_id: T_cm (camera <- marker)} and the detected marker corners / ids.
    """
    out = detector.detectBoard(gray)
    charuco_corners, charuco_ids = out[0], out[1]
    marker_corners, marker_ids = (out[2], out[3]) if len(out) >= 4 else (None, None)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 6 or marker_ids is None or len(marker_ids) == 0:
        return {}, marker_corners, marker_ids

    ids = charuco_ids.reshape(-1).astype(np.int32)
    board_corners = get_board_corners_3d(board)
    valid = (ids >= 0) & (ids < board_corners.shape[0])
    objp = board_corners[ids[valid], :].astype(np.float32)
    imgp = charuco_corners.reshape(-1, 2).astype(np.float32)[valid]
    if len(objp) < 6:
        return {}, marker_corners, marker_ids
    success, rvec_board, tvec_board = cv2.solvePnP(objp, imgp, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return {}, marker_corners, marker_ids

    # every marker lies in the board plane: same rotation as the board, translated to its center
    R_board, _ = cv2.Rodrigues(rvec_board)
    board_obj_points = board.getObjPoints()
    board_ids = board.getIds().flatten()
    poses = {}
    for marker_id in marker_ids.flatten():
        marker_idx = np.where(board_ids == int(marker_id))[0]
        if len(marker_idx) == 0:
            continue
        center = np.mean(board_obj_points[marker_idx[0]], axis=0).reshape(3, 1)
        R_marker, _ = cv2.Rodrigues(rvec_board.copy())
        T_cm = np.eye(4)
        T_cm[:3, :3] = R_marker
        T_cm[:3, 3] = (R_board @ center + tvec_board).flatten()
        poses[int(marker_id)] = ensure_SE3(T_cm)
    return poses, marker_corners, marker_ids


# ---------- Pose graph ----------
def optimize_with_gtsam(T_wc, marker_obs, odom_sigma_rot, odom_sigma_pos, meas_sigma_rot, meas_sigma_pos):
    """Marker <- camera trajectory from SLAM odometry (between factors) and marker detections (priors).

    T_wc: [N, 4, 4] SLAM world <- camera; marker_obs: {frame index: T_cm (camera <- marker)}.
    Returns T_mc_opt [N, 4, 4] and the estimated world <- marker pose.
    """
    N = T_wc.shape[0]
    T_wc = np.stack([ensure_SE3(T) for T in T_wc])
    obs = {i: ensure_SE3(T_cm) for i, T_cm in marker_obs.items()}

    # initialization: T_mc0 = T_mw * T_wc with T_mw averaged over the detections
    T_mw_init = se3_mean([ensure_SE3(inv_T(T_cm) @ inv_T(T_wc[i])) for i, T_cm in obs.items()])

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([odom_sigma_rot] * 3 + [odom_sigma_pos] * 3, dtype=float))
    meas_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([meas_sigma_rot] * 3 + [meas_sigma_pos] * 3, dtype=float))
    for i in range(N):
        initial.insert(i, mat4_to_pose3(ensure_SE3(T_mw_init @ T_wc[i])))
    for i in range(N - 1):
        graph.add(gtsam.BetweenFactorPose3(i, i + 1, mat4_to_pose3(ensure_SE3(inv_T(T_wc[i]) @ T_wc[i + 1])), odom_noise))
    for i, T_cm in obs.items():
        graph.add(gtsam.PriorFactorPose3(i, mat4_to_pose3(inv_T(T_cm)), meas_noise))

    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosity("SILENT")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    print(f"Pose graph: final error {optimizer.error():.4f} after {optimizer.iterations()} iterations")

    T_mc_opt = np.stack([pose3_to_mat4(result.atPose3(i)) for i in range(N)])
    T_wm_est = se3_mean([ensure_SE3(T_wc[i] @ inv_T(T_mc_opt[i])) for i in range(N)])
    return T_mc_opt, T_wm_est


def save_trajectory_comparison(T_wc, T_wc_corrected, detection_frames, output_path):
    p_orig, p_corr = T_wc[:, :3, 3], T_wc_corrected[:, :3, 3]
    frames = np.arange(len(p_orig))
    fig = plt.figure(figsize=(24, 12))
    ax = fig.add_subplot(221, projection="3d")
    ax.plot(*p_orig.T, "b-", label="DROID-SLAM")
    ax.plot(*p_corr.T, "r-", label="pose graph")
    ax.set_title("Camera trajectory (SLAM world frame)")
    ax.legend()
    ax = fig.add_subplot(222)
    ax.plot(np.linalg.norm(p_corr - p_orig, axis=1), "g-")
    ax.set_title("Position correction (m)")
    ax.set_xlabel("frame")
    ax = fig.add_subplot(223)
    for k, c in enumerate("bgr"):
        ax.plot(frames, p_orig[:, k], c + "-", alpha=0.7)
        ax.plot(frames, p_corr[:, k], c + "--", alpha=0.7)
    ax.set_title("Position (solid: SLAM, dashed: pose graph)")
    ax = fig.add_subplot(224)
    ax.plot(detection_frames, np.ones(len(detection_frames)), "g|", markersize=20)
    ax.set_xlim(0, len(frames))
    ax.set_title("Frames with a reference marker detection")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def process_episode(cfg, data_dir, root, episode_idx, board, detector):
    key = f"episode_{episode_idx}"
    episode = root[key]
    episode_dir = os.path.join(data_dir, key)
    K = np.array(cfg.camera.intrinsics, dtype=np.float32)
    dist_coeffs = np.zeros(5)

    T_wc = episode["extrinsics_droid_all"][:]
    rgb_frames, _ = load_rgbd_pngs(episode_dir)
    assert len(T_wc) == len(rgb_frames), f"{len(T_wc)} poses vs {len(rgb_frames)} frames"

    detections = {}
    marker_counts = {}
    vis_dir = os.path.join(episode_dir, "marker_detections")
    if cfg.save_marker_images:
        os.makedirs(vis_dir, exist_ok=True)
    for frame_idx, rgb in enumerate(rgb_frames):
        gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        poses, corners, ids = detect_marker_poses(gray, board, detector, K, dist_coeffs)
        if len(poses) == 0:
            continue
        detections[frame_idx] = poses
        for marker_id in poses:
            marker_counts[marker_id] = marker_counts.get(marker_id, 0) + 1
        if cfg.save_marker_images:
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            for marker_id, T_cm in poses.items():
                rvec, _ = cv2.Rodrigues(T_cm[:3, :3])
                length = cfg.board.marker_size * (0.8 if marker_id == cfg.board.reference_marker_id else 0.5)
                cv2.drawFrameAxes(vis, K, dist_coeffs, rvec, T_cm[:3, 3], length)
            cv2.imwrite(os.path.join(vis_dir, f"frame_{frame_idx:05d}.png"), vis)

    if len(detections) == 0:
        print(f"{key}: no marker detected, skipping")
        return False

    reference_id = cfg.board.reference_marker_id
    if reference_id not in marker_counts:
        reference_id = max(marker_counts, key=marker_counts.get)
        print(f"{key}: reference marker {cfg.board.reference_marker_id} not detected, using marker {reference_id}")
    marker_obs = {i: poses[reference_id] for i, poses in detections.items() if reference_id in poses}
    print(f"{key}: reference marker {reference_id} detected in {len(marker_obs)}/{len(T_wc)} frames")

    T_mc_opt, T_wm_est = optimize_with_gtsam(
        T_wc,
        marker_obs,
        odom_sigma_rot=np.deg2rad(cfg.pose_graph.odom_sigma_rot_deg),
        odom_sigma_pos=cfg.pose_graph.odom_sigma_pos,
        meas_sigma_rot=np.deg2rad(cfg.pose_graph.marker_sigma_rot_deg),
        meas_sigma_pos=cfg.pose_graph.marker_sigma_pos,
    )
    save_trajectory_comparison(
        T_wc, np.stack([T_wm_est @ T for T in T_mc_opt]), sorted(marker_obs),
        os.path.join(episode_dir, "marker_frame_trajectory.png"),
    )

    # trajectory anchored at the first detection of the reference marker (no optimization)
    first_idx = min(marker_obs)
    T_wm_first = T_wc[first_idx] @ marker_obs[first_idx]
    T_mc_first = np.stack([inv_T(T_wm_first) @ T for T in T_wc])

    start = int(episode["all_detected_frame_index"][()])
    save_array(episode, "T_mc_opt_droid", T_mc_opt[start:])
    save_array(episode, "T_mc_from_first_detection_droid", T_mc_first[start:])
    save_array(episode, "T_wm_opt_droid", T_wm_est[None])
    return True


@hydra.main(version_base=None, config_path="../../config/preprocessing", config_name="marker_frame")
def main(cfg: DictConfig):
    board = create_charuco_board(cfg)
    detector = cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), make_detector_params())
    for data_dir in resolve_data_dirs(cfg.data_dirs):
        root = zarr.open(data_dir, mode="a")
        failed = []
        for episode_idx in episode_range(data_dir, cfg.episode_start, cfg.episode_end):
            key = f"episode_{episode_idx}"
            if key not in root or "extrinsics_droid_all" not in root[key]:
                print(f"{key}: no camera trajectory (run estimate_camera_trajectory.py first), skipping")
                failed.append(episode_idx)
                continue
            try:
                if not process_episode(cfg, data_dir, root, episode_idx, board, detector):
                    failed.append(episode_idx)
            except Exception:
                import traceback

                traceback.print_exc()
                failed.append(episode_idx)
        print(f"{data_dir}: failed episodes {failed}")


if __name__ == "__main__":
    main()
