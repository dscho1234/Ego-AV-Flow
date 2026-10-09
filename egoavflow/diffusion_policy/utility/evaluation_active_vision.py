import os
import random
import cv2
import numpy as np
import torch
from tqdm import tqdm

from egoavflow.diffusion_policy.dataloader.diffusion_bc_dataset import (
    normalize_data,
    unnormalize_data,
)
from egoavflow.common.utility.projection import project_points_to_image, get_pixel_queries, fill_depth_zeros
from scipy.spatial.transform import Rotation as R
from itertools import accumulate
import time
import pickle
from egoavflow.common.utility.zarr import parallel_reading
from egoavflow.svdd import run_svdd, convert_to_relative, convert_se3_to_9d, compute_visibility, create_pose_from_9d, retarget_se3_trajectory
from egoavflow.visibility import Z1RobotVisualizer, CustomVisualizer, get_scene_mesh, rgbd_to_point_cloud, VOXEL_SIZE
from egoavflow.widowx_visualizer import WidowXVisualizer
import open3d as o3d
from nvblox_torch.mapper import Mapper
from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams, MeshIntegratorParams
from egoavflow.utils import compute_ortho6d_from_rotation_matrix, compute_rotation_matrix_from_ortho6d
import matplotlib.pyplot as plt

B = 1


def _stack_or_empty(values, trailing_shape=None, dtype=np.float32):
    if len(values) == 0:
        if trailing_shape is None:
            return np.empty((0,), dtype=dtype)
        return np.empty((0, *trailing_shape), dtype=dtype)
    return np.stack(values, axis=0)


def _pad_future_slice(sequence, start_idx, horizon, episode_length):
    end_idx = start_idx + horizon
    if end_idx <= episode_length:
        return sequence[start_idx:end_idx].copy()
    available = sequence[start_idx:episode_length].copy()
    remaining = horizon - len(available)
    if remaining <= 0:
        return available[:horizon].copy()
    last_frame = sequence[episode_length - 1:episode_length].copy()
    padding = np.tile(last_frame, (remaining, *([1] * (sequence.ndim - 1))))
    return np.concatenate([available, padding], axis=0)


def _normalize_flow_sequence(flow_unnorm, eval_dataset, flow_norm_type):
    flow_norm = flow_unnorm.copy()
    if flow_norm_type == 'min_max':
        flow_norm = (flow_norm - eval_dataset.pt_min[None, None]) / (eval_dataset.pt_max[None, None] - eval_dataset.pt_min[None, None])
        original_shape = flow_norm.shape
        flow_norm = eval_dataset.flow_transforms(flow_norm.reshape(-1, original_shape[-1]).transpose(1, 0))
        flow_norm = flow_norm.transpose(1, 0).reshape(original_shape)
    elif flow_norm_type == 'mean_std':
        flow_norm = (flow_norm - eval_dataset.pt_mean[None, None]) / (eval_dataset.pt_std[None, None] + 1e-10)
    else:
        raise NotImplementedError(f"Unsupported flow_norm_type: {flow_norm_type}")
    return flow_norm


def _compute_action_mse(action, action_gt):
    return {
        'mse_all': float(np.mean((action - action_gt) ** 2)),
        'mse_pos': float(np.mean((action[..., :3] - action_gt[..., :3]) ** 2)),
        'mse_rot6d': float(np.mean((action[..., 3:9] - action_gt[..., 3:9]) ** 2)),
        'mse_gripper': float(np.mean((action[..., -1:] - action_gt[..., -1:]) ** 2)),
    }


def use_pixel_flow_for_robot_policy(cfgs):
    return bool(getattr(cfgs.training, "use_pixel_flow_for_robot_policy", False))


def get_robot_policy_input_flow_dim(cfgs):
    return int(getattr(cfgs.training, "robot_policy_input_flow_dim", cfgs.training.input_flow_dim))


def build_flow_tokens(flow_history, input_flow_dim):
    B, H, N, D = flow_history.shape
    assert input_flow_dim <= D, f"input_flow_dim ({input_flow_dim}) must be <= flow dim ({D})"
    return flow_history[:, :, :, :input_flow_dim].permute(0, 2, 1, 3).reshape(B, N, -1)


def inv_T(T: np.ndarray) -> np.ndarray:
    """Inverse of SE(3) transformation matrix"""
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3,3]  = -R.T@t
    return Ti


def apply_fake_depth_to_mask(depths, mask, fake_value):
    """
    Apply fake depth (0) to depth values where mask=1
    
    Args:
        depths: numpy array of shape [t, h, w] with depth values
        mask: numpy array of shape [t, h, w] with binary values (0 or 1)
    
    Returns:
        Modified depths with 0 values in masked regions
    """
    # Create a copy to avoid modifying original data
    modified_depths = depths.copy()
    
    # Apply 0 depth to masked regions
    modified_depths = np.where(mask == 1, fake_value, modified_depths)
    
    return modified_depths


def apply_mask_depth_filter_to_tracked_3d(tracked_3d, sam_mask_sequence_multi_obj, depth, intrinsics, extrinsics, mask_point_tracking_threshold=0.1, num_points_per_mask=64):
    """
    Apply mask-based depth filtering to tracked_3d's visibility.
    Similar to realworld_3d_point_tracking_depth.py's filtering logic.
    
    Args:
        tracked_3d: [N, 4] array - unnormalized, marker coordinate (x, y, z, visibility)
        sam_mask_sequence_multi_obj: [num_obj, H, W] array - current step's mask
        depth: [H, W] array - current step's depth
        intrinsics: [3, 3] array - camera intrinsics
        extrinsics: [4, 4] array - current step's extrinsics (T_mc: marker to camera)
        mask_point_tracking_threshold: float - threshold for filtering (in meters)
        num_points_per_mask: int - number of points to sample per mask
    
    Returns:
        tracked_3d: [N, 4] array with updated visibility
    """
    if tracked_3d.shape[1] != 4:
        return tracked_3d  # No visibility dimension, return as is
    
    # Process each object's mask to generate point cloud and compute mean_z
    num_obj = sam_mask_sequence_multi_obj.shape[0]
    if num_obj > 1:
        num_obj -= 1  # remove hand
    
    all_valid_mask_points = []
    
    for obj_idx in range(num_obj):
        mask = sam_mask_sequence_multi_obj[obj_idx]  # [H, W]
        
        # Unproject mask pixels to 3D
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        
        # Get depth values at mask pixels
        depths_at_mask = depth[ys, xs]
        valid_mask = depths_at_mask > 0
        if not np.any(valid_mask):
            continue
        
        xs_valid = xs[valid_mask]
        ys_valid = ys[valid_mask]
        
        # Randomly sample num_points_per_mask points (with replacement if needed)
        num_valid = len(xs_valid)
        if num_valid > 0:
            indices = np.random.choice(num_valid, num_points_per_mask, replace=(num_valid < num_points_per_mask))
            sampled_xs = xs_valid[indices]
            sampled_ys = ys_valid[indices]
        else:
            continue
        
        # Unproject to 3D using identity extrinsics to get camera coordinate
        xy_coords = np.stack([sampled_xs, sampled_ys], axis=1).astype(np.float32)  # [num_points_per_mask, 2]
        depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)  # [1, H, W]
        intrinsics_tensor = torch.from_numpy(intrinsics).float().unsqueeze(0)  # [1, 3, 3]
        identity_extrinsics = np.eye(4, dtype=np.float32)  # Identity matrix for camera coordinate
        extrinsics_tensor = torch.from_numpy(identity_extrinsics).float().unsqueeze(0)  # [1, 4, 4]
        xy_coords_tensor = torch.from_numpy(xy_coords).float()  # [N, 2]
        
        queries = get_pixel_queries(
            xy_coords_tensor,
            depth_tensor,
            intrinsics_tensor,
            extrinsics_tensor
        )  # [N, 4] -> (0, X, Y, Z) in camera coordinate
        
        point_cloud = queries[:, 1:4].detach().cpu().numpy()  # [num_points_per_mask, 3] in camera coordinate
        
        # Remove zero points (invalid points)
        valid_mask_points = point_cloud[np.any(point_cloud != 0, axis=1)]  # [num_valid, 3]
        if len(valid_mask_points) > 0:
            all_valid_mask_points.append(valid_mask_points)
    
    # Compute mean_z from all valid mask points (in camera coordinate)
    if len(all_valid_mask_points) > 0:
        all_valid_mask_points = np.concatenate(all_valid_mask_points, axis=0)  # [total_valid, 3]
        mean_z = np.mean(all_valid_mask_points[:, 2])  # scalar (camera coordinate z)
        
        # Get z values from tracked_3d (unnormalized, marker coordinate)
        # Need to convert marker coordinate to camera coordinate
        tracked_3d_xyz_marker = tracked_3d[:, :3]  # [N, 3] - marker coordinate, unnormalized
        
        # Convert marker coordinate to camera coordinate
        # extrinsics is T_mc (marker to camera), so T_mc = extrinsics
        T_cm = inv_T(extrinsics)  # [4, 4] - camera to marker (inverse of marker to camera)
        
        # Convert tracked_3d_xyz_marker from marker coordinate to camera coordinate
        tracked_3d_xyz_marker_homo = np.hstack([tracked_3d_xyz_marker, np.ones((tracked_3d_xyz_marker.shape[0], 1))])  # [N, 4]
        tracked_3d_xyz_camera_homo = (T_cm @ tracked_3d_xyz_marker_homo.T).T  # [N, 4] - camera coordinate
        tracked_3d_xyz_camera = tracked_3d_xyz_camera_homo[:, :3]  # [N, 3] - camera coordinate
        
        # Get z values in camera coordinate
        z_values_camera = tracked_3d_xyz_camera[:, 2]  # [N] - camera coordinate z
        
        # Compute distance from mean z for all points (in camera coordinate)
        dist_from_mean_z = np.abs(z_values_camera - mean_z)  # [N]
        
        # Set visibility to 0 if too far from mean z
        too_far_mask = dist_from_mean_z > mask_point_tracking_threshold
        tracked_3d[too_far_mask, -1] = 0.0  # Set visibility to 0 for points too far
    
    return tracked_3d

def get_flow_output(flow_model, point_flow, proprioception, stats, cfgs, pred_horizon, input_horizon, current_point_flow, visibility, eval_dataset, viewpoint, future_action_tokens_tensor=None):
    """
    Get future point flow prediction from flow_model.
    
    Args:
        flow_model: CustomRDTRunner model for predicting future point flow
        point_flow: [B, H, N, D] - point flow history
        proprioception: [B, H, dim] - proprioception history
        stats: statistics for unnormalization
        cfgs: config object
        pred_horizon: prediction horizon
        input_horizon: input horizon (may include history if use_ptp)
        current_point_flow: [B, N, D] - current point flow
        visibility: [B, N] - visibility mask
        eval_dataset: evaluation dataset
        viewpoint: [B, H, 9] - viewpoint history
        future_action_tokens_tensor: [B(1), pred_horizon, action_dim] or None - normalized future action tokens tensor (already normalized and converted to tensor)
    
    Returns:
        flow_pred_unnorm: [pred_horizon, N, D] - unnormalized predicted flow
    """
    # Convert point_flow to flow_tokens: [B, H, N, D] -> [B, N, H*D] for point-wise positional embedding
    B, H, N, D = point_flow.shape
    assert B == 1
    flow_tokens = build_flow_tokens(point_flow, get_robot_policy_input_flow_dim(cfgs))
    
    # Create flow_mask based on visibility
    if D == 4 or use_pixel_flow_for_robot_policy(cfgs):
        if cfgs.training.use_masking:
            flow_mask = visibility.bool()  # [B, N] - True if visible
            if not torch.any(flow_mask, dim=1).all():
                flow_mask[:, torch.randint(0, N, (1,), device=point_flow.device)] = True
            assert torch.any(flow_mask, dim=1).all(), f"at least one should be visible(true) along N axis, flow_mask : {flow_mask}"
        else:
            flow_mask = None
    else:
        flow_mask = None
    proprio_mask = None
    
    # Proprioception tokens: [B, H, dim] -> already in correct shape
    proprio_tokens = proprioception  # [B, H, dim]
    
    # Prepare fix_mask and prior for flow_model
    flow_fix_mask = None
    flow_prior = None
    if cfgs.use_fix_mask:
        flow_output_dim = flow_model.action_dim  # flow_model's action_dim is flow_output_dim
        flow_fix_mask = torch.zeros(B, input_horizon, flow_output_dim, device=point_flow.device)
        flow_prior = torch.zeros(B, input_horizon, flow_output_dim, device=point_flow.device)
        if not cfgs.use_relative:
            flow_fix_mask[:, 0, :] = 1.0  # fix the first flow
            if cfgs.training.use_ptp:
                flow_prior[:, :cfgs.dataset.obs_horizon, :] = current_point_flow.reshape(B, -1)  # [B, N*D] normalized
            else:
                flow_prior[:, 0, :] = current_point_flow.reshape(B, -1)  # [B, N*D] normalized
    

    viewpoint_tokens = viewpoint
    viewpoint_mask = None
    
    # future_action_tokens_tensor is already normalized and converted to tensor outside this function
    # future_action_mask is always None in evaluation
    future_action_mask = None
    with torch.no_grad():
        flow_pred_output = flow_model.predict_action(
            flow_tokens=flow_tokens,
            proprio_tokens=proprio_tokens,
            fix_mask=flow_fix_mask,
            prior=flow_prior,
            flow_mask=flow_mask,
            proprio_mask=proprio_mask,
            viewpoint_tokens=viewpoint_tokens,  
            viewpoint_mask=viewpoint_mask,
            future_action_tokens=future_action_tokens_tensor,
            future_action_mask=future_action_mask,
            return_attn=False
        )
    
    # flow_pred_output: [B, horizon, flow_output_dim] where flow_output_dim = N * D
    # Reshape to [B, horizon, N, D]
    B_flow, H_flow, flow_output_dim = flow_pred_output.shape
    num_flow_points = cfgs.num_flow_points
    flow_dim = eval_dataset.flow_dim
    assert flow_output_dim == num_flow_points * flow_dim, f"flow_output_dim ({flow_output_dim}) != num_flow_points ({num_flow_points}) * flow_dim ({flow_dim})"
    flow_pred = flow_pred_output.reshape(B_flow, H_flow, num_flow_points, flow_dim)  # [B, horizon, N, D]
    
    # Unnormalize predicted flow (marker coordinate, normalized -> unnormalized)
    flow_pred_np = flow_pred[0].detach().cpu().numpy()  # [horizon, N, D]
    if cfgs.training.use_ptp:
        flow_pred_np = flow_pred_np[cfgs.dataset.obs_horizon-1:].copy()
    flow_pred_np = flow_pred_np[:pred_horizon].copy()  # [pred_horizon, N, D]
    
    # Unnormalize: use unnormalize_data for delta_point_tracking when use_relative, otherwise use pt_min/pt_max or pt_mean/pt_std
    if cfgs.use_relative:
        # For delta_point_tracking, use unnormalize_data with stats["delta_point_tracking_sequence"]
        flow_pred_np = unnormalize_data(flow_pred_np, eval_dataset.stats["delta_point_tracking_sequence"], type=eval_dataset.flow_norm_type)  # [pred_horizon, N, D]
    else:
        # For absolute point tracking, use pt_min/pt_max or pt_mean/pt_std
        pt_min = eval_dataset.pt_min
        pt_max = eval_dataset.pt_max
        pt_mean = eval_dataset.pt_mean
        pt_std = eval_dataset.pt_std
        if eval_dataset.flow_norm_type == 'min_max':
            flow_pred_np = eval_dataset.unnormalize_transform(flow_pred_np)  # [pred_horizon, N, D]
            flow_pred_np = flow_pred_np * (pt_max[None, None, :] - pt_min[None, None, :]) + pt_min[None, None, :]  # [pred_horizon, N, D]
        elif eval_dataset.flow_norm_type == 'mean_std':
            flow_pred_np = flow_pred_np * (pt_std[None, None, :] + 1e-10) + pt_mean[None, None, :]  # [pred_horizon, N, D]
    
    if flow_pred_np.shape[-1] == 4:
        flow_pred_np[..., -1] = np.clip(flow_pred_np[..., -1], 0.0, 1.0) # visibility clip to [0, 1]
        # 1 if greater than 0.5 else 0
        flow_pred_np[..., -1] = np.where(flow_pred_np[..., -1] > 0.5, 1.0, 0.0)
    else:
        assert flow_pred_np.shape[-1] == 3, f"flow_pred_np.shape: {flow_pred_np.shape}"
    
    # Handle use_relative: predicted flow is relative change, need to accumulate from current point
    if cfgs.use_relative:
        # flow_pred_np is relative change (delta) [pred_horizon, N, D], already unnormalized
        # Get current tracking point (normalized tensor -> unnormalized numpy)
        current_point_flow_np = current_point_flow[0].detach().cpu().numpy()  # [N, D] - current tracking point (normalized)
        # Unnormalize current_point_flow
        if eval_dataset.flow_norm_type == 'min_max':
            current_point_flow_np = eval_dataset.unnormalize_transform(current_point_flow_np[None])[0]
            pt_min = eval_dataset.pt_min
            pt_max = eval_dataset.pt_max
            current_point_flow_np = current_point_flow_np * (pt_max[None, :] - pt_min[None, :]) + pt_min[None, :]
        elif eval_dataset.flow_norm_type == 'mean_std':
            pt_mean = eval_dataset.pt_mean
            pt_std = eval_dataset.pt_std
            current_point_flow_np = current_point_flow_np * (pt_std[None, :] + 1e-10) + pt_mean[None, :]
        
        # Accumulate relative changes using cumsum (same as action processing)
        # Note: visibility (if D=4) is not included in delta calculation, so only x, y, z are accumulated
        if flow_pred_np.shape[-1] == 4:
            # Separate xyz and visibility
            flow_pred_xyz = flow_pred_np[..., :3]  # [pred_horizon, N, 3]
            flow_pred_vis = flow_pred_np[..., -1:]  # [pred_horizon, N, 1]
            current_point_flow_xyz = current_point_flow_np[..., :3]  # [N, 3]
            current_point_flow_vis = current_point_flow_np[..., -1:]  # [N, 1]
            
            # Accumulate only xyz (visibility is not delta, use predicted visibility directly)
            flow_pred_accumulated_xyz = current_point_flow_xyz[None, :, :] + np.cumsum(flow_pred_xyz, axis=0)  # [pred_horizon, N, 3]
            # Use predicted visibility directly (not accumulated)
            flow_pred_accumulated = np.concatenate([flow_pred_accumulated_xyz, flow_pred_vis], axis=-1)  # [pred_horizon, N, 4]
        else:
            # D=3: no visibility, accumulate all dimensions
            flow_pred_accumulated = current_point_flow_np[None, :, :] + np.cumsum(flow_pred_np, axis=0)  # [pred_horizon, N, 3]
        
        flow_pred_unnorm = flow_pred_accumulated  # [pred_horizon, N, D], marker coordinate, unnormalized, absolute
    else:
        # Absolute: predicted flow is already absolute position
        flow_pred_unnorm = flow_pred_np  # [pred_horizon, N, D], marker coordinate, unnormalized
    
    if flow_pred_np.shape[-1] == 4:
        assert flow_pred_unnorm[..., -1].min() in [0, 1] and flow_pred_unnorm[..., -1].max() in [0, 1], f"flow_pred_unnorm[..., -1].min(): {flow_pred_unnorm[..., -1].min()}, flow_pred_unnorm[..., -1].max(): {flow_pred_unnorm[..., -1].max()}"
    
    return flow_pred_unnorm


def retargeting(robot_viz, action, T_B_M, current_proprioception_unnorm=None):
    """
    Retarget robot action trajectory using IK and retargeting if needed.
    
    Args:
        robot_viz: Robot visualizer (WidowXVisualizer) with methods:
            - compute_forward_kinematics(joint_angles, gripper_angle)
            - solve_ik_null_space(target_T, initial_guess, ...)
            - joint_limits: joint limits
        action: [H, 10] array of robot actions (position: 3, ortho6d: 6, gripper: 1) in marker coordinate
        T_B_M: [4, 4] transformation from marker to base coordinate
        current_proprioception_unnorm: [10] current proprioception (position: 3, ortho6d: 6, gripper: 1) in marker coordinate, optional
    
    Returns:
        action_retargeted: [H, 10] array of retargeted robot actions in marker coordinate
    """
    retarget_start = time.time()
    H, action_dim = action.shape
    assert action_dim == 10, f"Expected action_dim=10 (pos:3, ortho6d:6, gripper:1), got {action_dim}"
    
    # Extract position, rotation, and gripper from action
    action_pos = action[:, :3]  # [H, 3] marker coordinate
    action_ortho6d = action[:, 3:9]  # [H, 6] marker coordinate
    action_gripper = action[:, 9:10]  # [H, 1]
    
    # Convert action to SE(3) in marker coordinate
    action_se3_marker = create_pose_from_9d(np.concatenate([action_pos, action_ortho6d], axis=-1))  # [H, 4, 4] marker coordinate
    
    # Convert to base coordinate
    action_se3_B_G = np.einsum('ij,hjk->hik', T_B_M, action_se3_marker)  # [H, 4, 4] base coordinate

    T_ge = np.eye(4)
    T_ge[:3, 3] = -robot_viz.ee_bias.copy()

    action_se3_B_E = np.einsum('hij,jk->hik', action_se3_B_G, T_ge)  # [H, 4, 4]


    # Get initial joint guess from current end-effector pose if available
    if current_proprioception_unnorm is not None:
        current_pos = current_proprioception_unnorm[:3]  # [3] marker coordinate
        current_ortho6d = current_proprioception_unnorm[3:9]  # [6] marker coordinate
        current_se3_marker = create_pose_from_9d(np.concatenate([current_pos[None], current_ortho6d[None]], axis=-1))[0]  # [4, 4] marker coordinate
        current_se3_B_G = T_B_M @ current_se3_marker  # [4, 4] base coordinate
        current_se3_B_E = current_se3_B_G @ T_ge  # [4, 4] base coordinate
        
        # Use IK to get initial joint guess
        initial_guess = robot_viz.joint_angles.copy() if hasattr(robot_viz, 'joint_angles') else np.zeros(6)
        success, q0, _, _, _ = robot_viz.solve_ik_null_space(
            current_se3_B_E,
            initial_guess=initial_guess,
            max_iterations=50,
            tolerance=1e-2,
            tolerance_null=1e-3
        )
        
    else:
        # Use current joint angles if available
        q0 = robot_viz.joint_angles.copy() if hasattr(robot_viz, 'joint_angles') else np.zeros(6)
    
    # Perform retargeting
    joint_limits = robot_viz.joint_limits
    q_min = np.array([limit[0] for limit in joint_limits])
    q_max = np.array([limit[1] for limit in joint_limits])
    
    q_traj, T_traj, retarget_info = retarget_se3_trajectory(
        robot_viz,
        action_se3_B_E,
        q0,
        w_pos=1.0,
        w_rot=0.5,
        lambda_smooth=1e-3,
        bounds=(q_min, q_max),
        max_nfev=100,
    )
    
    # Convert retargeted trajectory back to marker coordinate
    T_M_B = np.linalg.inv(T_B_M)
    T_B_E_traj = T_traj.copy()
    T_B_G_traj = np.einsum('hij,jk->hik', T_B_E_traj, np.linalg.inv(T_ge))  # [H, 4, 4] base coordinate
    action_se3_retargeted_marker = np.einsum('ij,hjk->hik', T_M_B, T_B_G_traj)  # [H, 4, 4] marker coordinate
    
    # Convert SE(3) back to position + ortho6d
    action_retargeted_9d = convert_se3_to_9d(action_se3_retargeted_marker[None])[0]  # [H, 9] marker coordinate
    action_retargeted_pos = action_retargeted_9d[:, :3]  # [H, 3]
    action_retargeted_ortho6d = action_retargeted_9d[:, 3:9]  # [H, 6]
    
    # Keep original gripper action
    action_retargeted = np.concatenate([action_retargeted_pos, action_retargeted_ortho6d, action_gripper], axis=-1)  # [H, 10]
    
    print(f"Robot env retargeting completed. Time taken: {time.time() - retarget_start:.2f}s. Final cost: {retarget_info['costs'][-1]:.6f}")
    
    return action_retargeted


def get_policy_output(model, point_flow, proprioception, stats, cfgs, action_dim, pred_horizon, input_horizon, current_proprioception, current_point_flow, visibility, eval_dataset, return_attn=False):
    # future flow conditioned diffusion policy to sample actions using CustomRDTRunner
    
    
    # Convert point_flow to flow_tokens: [B, H, N, D] -> [B, N, H*D] for point-wise positional embedding
    B, H, N, D = point_flow.shape
    assert B == 1
    flow_tokens = build_flow_tokens(point_flow, get_robot_policy_input_flow_dim(cfgs))
    
    # Create flow_mask based on visibility (last dimension of point_flow)
    # If D >= 4, the last dimension is visibility (0 > visible, <= 0 invisible)
    # flow_mask: (B, N) where True means visible (masking=true), False means invisible (masking=false)
    if D == 4 or use_pixel_flow_for_robot_policy(cfgs):
        if cfgs.training.use_masking: # flow: 3dim, but masking is applied
            flow_mask = visibility.bool()  # [B, N] - True if visible in the most recent timestep
            if not torch.any(flow_mask, dim=1).all():
                flow_mask[:, torch.randint(0, N, (1,), device=point_flow.device)] = True
                
            assert torch.any(flow_mask, dim=1).all(), f"at least one should be visible(true) along N axis, flow_mask : {flow_mask}"
        else:
            flow_mask = None
    else:
        # If D < 4, no visibility dimension, assume all points are visible
        flow_mask = None
    proprio_mask = None

    # Proprioception tokens: [B, H, dim] -> already in correct shape
    proprio_tokens = proprioception  # [B, H, dim]
    
    # Prepare fix_mask and prior if enabled (similar to evaluation_active_vision.py)
    fix_mask = None
    prior = None
    if cfgs.use_fix_mask:
        if model.predict_flow:
            flow_output_dim = model.flow_output_dim
            fix_mask = torch.zeros(B, input_horizon, action_dim + flow_output_dim, device=point_flow.device)
            prior = torch.zeros(B, input_horizon, action_dim + flow_output_dim, device=point_flow.device)
        else:
            fix_mask = torch.zeros(B, input_horizon, action_dim, device=point_flow.device)
            prior = torch.zeros(B, input_horizon, action_dim, device=point_flow.device)
        
        if not cfgs.use_relative:
            fix_mask[:, 0, :] = 1.0  # fix the first action
            if model.predict_flow: # current_point_flow: [B(1), N, 3 or 4] -> [B, N*D]
                if cfgs.training.use_ptp:
                    prior[:, :cfgs.dataset.obs_horizon, :] = torch.cat([current_proprioception, current_point_flow.reshape(B, -1)], dim=-1)  # [B, D+N*D] normalized
                else:
                    prior[:, 0, :] = torch.cat([current_proprioception, current_point_flow.reshape(B, -1)], dim=-1)  # [B, D+N*D] normalized
            else:
                if cfgs.training.use_ptp:
                    prior[:, :cfgs.dataset.obs_horizon, :] = current_proprioception  # [B, D] normalized
                else:
                    prior[:, 0, :] = current_proprioception  # [B, D] normalized
    
    start = time.time()
    with torch.no_grad():
        # Use CustomRDTRunner's predict_action method with fix_mask and prior
        action_pred_output = model.predict_action(
            flow_tokens=flow_tokens,
            proprio_tokens=proprio_tokens,
            fix_mask=fix_mask,
            prior=prior,
            flow_mask=flow_mask,
            proprio_mask=proprio_mask,
            return_attn=return_attn
        )
        
        # Handle output based on predict_flow or predict_separate_flow flag
        flow_pred_unnorm = None
        if return_attn:
            if model.predict_separate_flow or model.predict_flow:
                action_pred, flow_pred, attentions = action_pred_output  # Extract action, flow, and attention
                # flow_pred: [B, horizon, flow_output_dim] where flow_output_dim = N * D
                # Reshape to [B, horizon, N, D]
                B_flow, H_flow, flow_output_dim = flow_pred.shape
                
                # Get N and D from flow_output_dim
                num_flow_points = cfgs.num_flow_points
                flow_dim = eval_dataset.flow_dim
                assert flow_output_dim == num_flow_points * flow_dim, f"flow_output_dim ({flow_output_dim}) != num_flow_points ({num_flow_points}) * flow_dim ({flow_dim})"
                flow_pred = flow_pred.reshape(B_flow, H_flow, num_flow_points, flow_dim)  # [B, horizon, N, D]
                # Unnormalize predicted flow (marker coordinate, normalized -> unnormalized)
                flow_pred_np = flow_pred[0].detach().cpu().numpy()  # [horizon, N, D]
                if cfgs.training.use_ptp:
                    flow_pred_np = flow_pred_np[cfgs.dataset.obs_horizon-1:].copy()
                flow_pred_np = flow_pred_np[:pred_horizon].copy()  # [pred_horizon, N, D]
                
                
                # Unnormalize: use unnormalize_data for delta_point_tracking when use_relative, otherwise use pt_min/pt_max or pt_mean/pt_std
            
                if cfgs.use_relative:
                    # For delta_point_tracking, use unnormalize_data with stats["delta_point_tracking_sequence"]
                    flow_pred_np = unnormalize_data(flow_pred_np, eval_dataset.stats["delta_point_tracking_sequence"], type=eval_dataset.flow_norm_type)  # [pred_horizon, N, D]
                else:
                    # For absolute point tracking, use pt_min/pt_max or pt_mean/pt_std
                    pt_min = eval_dataset.pt_min
                    pt_max = eval_dataset.pt_max
                    pt_mean = eval_dataset.pt_mean
                    pt_std = eval_dataset.pt_std
                    if eval_dataset.flow_norm_type == 'min_max':
                        flow_pred_np = eval_dataset.unnormalize_transform(flow_pred_np)  # [pred_horizon, N, D]
                        flow_pred_np = flow_pred_np * (pt_max[None, None, :] - pt_min[None, None, :]) + pt_min[None, None, :]  # [pred_horizon, N, D]
                    elif eval_dataset.flow_norm_type == 'mean_std':
                        flow_pred_np = flow_pred_np * (pt_std[None, None, :] + 1e-10) + pt_mean[None, None, :]  # [pred_horizon, N, D]
                
                if flow_pred_np.shape[-1] == 4:
                    flow_pred_np[..., -1] = np.clip(flow_pred_np[..., -1], 0.0, 1.0) # visibility clip to [0, 1]
                    # 1 if greater than 0.5 else 0
                    flow_pred_np[..., -1] = np.where(flow_pred_np[..., -1] > 0.5, 1.0, 0.0)
                else:
                    assert flow_pred_np.shape[-1] == 3, f"flow_pred_np.shape: {flow_pred_np.shape}"

                # Handle use_relative: predicted flow is relative change, need to accumulate from current point
                if cfgs.use_relative:
                    # flow_pred_np is relative change (delta) [pred_horizon, N, D], already unnormalized
                    # Get current tracking point (normalized tensor -> unnormalized numpy)
                    current_point_flow_np = current_point_flow[0].detach().cpu().numpy()  # [N, D] - current tracking point (normalized)
                    # Unnormalize current_point_flow
                    if eval_dataset.flow_norm_type == 'min_max':
                        current_point_flow_np = eval_dataset.unnormalize_transform(current_point_flow_np[None])[0]
                        pt_min = eval_dataset.pt_min
                        pt_max = eval_dataset.pt_max
                        current_point_flow_np = current_point_flow_np * (pt_max[None, :] - pt_min[None, :]) + pt_min[None, :]
                    elif eval_dataset.flow_norm_type == 'mean_std':
                        pt_mean = eval_dataset.pt_mean
                        pt_std = eval_dataset.pt_std
                        current_point_flow_np = current_point_flow_np * (pt_std[None, :] + 1e-10) + pt_mean[None, :]
                
                    # Accumulate relative changes using cumsum (same as action processing)
                    # Note: visibility (if D=4) is not included in delta calculation, so only x, y, z are accumulated
                    if flow_pred_np.shape[-1] == 4:
                        # Separate xyz and visibility
                        flow_pred_xyz = flow_pred_np[..., :3]  # [pred_horizon, N, 3]
                        flow_pred_vis = flow_pred_np[..., -1:]  # [pred_horizon, N, 1]
                        current_point_flow_xyz = current_point_flow_np[..., :3]  # [N, 3]
                        current_point_flow_vis = current_point_flow_np[..., -1:]  # [N, 1]
                        
                        # Accumulate only xyz (visibility is not delta, use predicted visibility directly)
                        flow_pred_accumulated_xyz = current_point_flow_xyz[None, :, :] + np.cumsum(flow_pred_xyz, axis=0)  # [pred_horizon, N, 3]
                        # Use predicted visibility directly (not accumulated)
                        flow_pred_accumulated = np.concatenate([flow_pred_accumulated_xyz, flow_pred_vis], axis=-1)  # [pred_horizon, N, 4]
                    else:
                        # D=3: no visibility, accumulate all dimensions
                        flow_pred_accumulated = current_point_flow_np[None, :, :] + np.cumsum(flow_pred_np, axis=0)  # [pred_horizon, N, 3]
                    
                    flow_pred_unnorm = flow_pred_accumulated  # [pred_horizon, N, D], marker coordinate, unnormalized, absolute
                else:
                    # Absolute: predicted flow is already absolute position
                    flow_pred_unnorm = flow_pred_np  # [pred_horizon, N, D], marker coordinate, unnormalized
                
                assert flow_pred_unnorm[..., -1].min() in [0,1] and flow_pred_unnorm[..., -1].max() in [0,1], f"flow_pred_unnorm[..., -1].min(): {flow_pred_unnorm[..., -1].min()}, flow_pred_unnorm[..., -1].max(): {flow_pred_unnorm[..., -1].max()}"
            else:
                action_pred, _, attentions = action_pred_output
        else: # not return attention
            if model.predict_separate_flow or model.predict_flow:
                action_pred, flow_pred = action_pred_output  # Extract action and flow
                # flow_pred: [B, horizon, flow_output_dim] where flow_output_dim = N * D
                # Reshape to [B, horizon, N, D]
                B_flow, H_flow, flow_output_dim = flow_pred.shape
                
                # Get N and D from flow_output_dim
                num_flow_points = cfgs.num_flow_points
                flow_dim = eval_dataset.flow_dim
                assert flow_output_dim == num_flow_points * flow_dim, f"flow_output_dim ({flow_output_dim}) != num_flow_points ({num_flow_points}) * flow_dim ({flow_dim})"
                flow_pred = flow_pred.reshape(B_flow, H_flow, num_flow_points, flow_dim)  # [B, horizon, N, D]
                # Unnormalize predicted flow (marker coordinate, normalized -> unnormalized)
                flow_pred_np = flow_pred[0].detach().cpu().numpy()  # [horizon, N, D]
                if cfgs.training.use_ptp:
                    flow_pred_np = flow_pred_np[cfgs.dataset.obs_horizon-1:].copy()
                flow_pred_np = flow_pred_np[:pred_horizon].copy()  # [pred_horizon, N, D]

                
                # Unnormalize: use unnormalize_data for delta_point_tracking when use_relative, otherwise use pt_min/pt_max or pt_mean/pt_std
                if cfgs.use_relative:
                    # For delta_point_tracking, use unnormalize_data with stats["delta_point_tracking_sequence"]
                    flow_pred_np = unnormalize_data(flow_pred_np, eval_dataset.stats["delta_point_tracking_sequence"], type=eval_dataset.flow_norm_type)  # [pred_horizon, N, D]
                else:
                    # For absolute point tracking, use pt_min/pt_max or pt_mean/pt_std
                    pt_min = eval_dataset.pt_min
                    pt_max = eval_dataset.pt_max
                    pt_mean = eval_dataset.pt_mean
                    pt_std = eval_dataset.pt_std
                    if eval_dataset.flow_norm_type == 'min_max':
                        flow_pred_np = eval_dataset.unnormalize_transform(flow_pred_np)  # [pred_horizon, N, D]
                        flow_pred_np = flow_pred_np * (pt_max[None, None, :] - pt_min[None, None, :]) + pt_min[None, None, :]  # [pred_horizon, N, D]
                    elif eval_dataset.flow_norm_type == 'mean_std':
                        flow_pred_np = flow_pred_np * (pt_std[None, None, :] + 1e-10) + pt_mean[None, None, :]  # [pred_horizon, N, D]
                
                if flow_pred_np.shape[-1] == 4:
                    flow_pred_np[..., -1] = np.clip(flow_pred_np[..., -1], 0.0, 1.0) # visibility clip to [0, 1]
                    # 1 if greater than 0.5 else 0
                    flow_pred_np[..., -1] = np.where(flow_pred_np[..., -1] > 0.5, 1.0, 0.0)
                else:
                    assert flow_pred_np.shape[-1] == 3, f"flow_pred_np.shape: {flow_pred_np.shape}"


                # Handle use_relative: predicted flow is relative change, need to accumulate from current point
                if cfgs.use_relative:
                    # flow_pred_np is relative change (delta) [pred_horizon, N, D], already unnormalized
                    # Get current tracking point (normalized tensor -> unnormalized numpy)
                    current_point_flow_np = current_point_flow[0].detach().cpu().numpy()  # [N, D] - current tracking point (normalized)
                    # Unnormalize current_point_flow
                    if eval_dataset.flow_norm_type == 'min_max':
                        current_point_flow_np = eval_dataset.unnormalize_transform(current_point_flow_np[None])[0]
                        pt_min = eval_dataset.pt_min
                        pt_max = eval_dataset.pt_max
                        current_point_flow_np = current_point_flow_np * (pt_max[None, :] - pt_min[None, :]) + pt_min[None, :]
                    elif eval_dataset.flow_norm_type == 'mean_std':
                        pt_mean = eval_dataset.pt_mean
                        pt_std = eval_dataset.pt_std
                        current_point_flow_np = current_point_flow_np * (pt_std[None, :] + 1e-10) + pt_mean[None, :]
                
                    # Accumulate relative changes using cumsum (same as action processing)
                    # Note: visibility (if D=4) is not included in delta calculation, so only x, y, z are accumulated
                    if flow_pred_np.shape[-1] == 4:
                        # Separate xyz and visibility
                        flow_pred_xyz = flow_pred_np[..., :3]  # [pred_horizon, N, 3]
                        flow_pred_vis = flow_pred_np[..., -1:]  # [pred_horizon, N, 1]
                        current_point_flow_xyz = current_point_flow_np[..., :3]  # [N, 3]
                        current_point_flow_vis = current_point_flow_np[..., -1:]  # [N, 1]
                        
                        # Accumulate only xyz (visibility is not delta, use predicted visibility directly)
                        flow_pred_accumulated_xyz = current_point_flow_xyz[None, :, :] + np.cumsum(flow_pred_xyz, axis=0)  # [pred_horizon, N, 3]
                        # Use predicted visibility directly (not accumulated)
                        flow_pred_accumulated = np.concatenate([flow_pred_accumulated_xyz, flow_pred_vis], axis=-1)  # [pred_horizon, N, 4]
                    else:
                        # D=3: no visibility, accumulate all dimensions
                        flow_pred_accumulated = current_point_flow_np[None, :, :] + np.cumsum(flow_pred_np, axis=0)  # [pred_horizon, N, 3]
                    
                    flow_pred_unnorm = flow_pred_accumulated  # [pred_horizon, N, D], marker coordinate, unnormalized, absolute
                else:
                    # Absolute: predicted flow is already absolute position
                    flow_pred_unnorm = flow_pred_np  # [pred_horizon, N, D], marker coordinate, unnormalized
                
                assert flow_pred_unnorm[..., -1].min() in [0,1] and flow_pred_unnorm[..., -1].max() in [0,1], f"flow_pred_unnorm[..., -1].min(): {flow_pred_unnorm[..., -1].min()}, flow_pred_unnorm[..., -1].max(): {flow_pred_unnorm[..., -1].max()}"
            else:
                action_pred = action_pred_output
        # action_pred: [B, horizon, action_dim]
    
    print(f'In evaluation, action generation diffusion time taken: {time.time() - start}')
    
    # Convert to numpy
    naction = action_pred[0].detach().cpu().numpy()  # [horizon, dim]
    if cfgs.training.use_ptp:
        naction = naction[cfgs.dataset.obs_horizon-1:].copy()

    if cfgs.use_relative:
        action_pred = unnormalize_data(naction.copy(), stats=stats["delta_action"], type=cfgs.dataset.action_norm_type) # 10D for absolute, 7D for relative
        action_gripper = action_pred[..., -1:].copy()
        
        current_proprioception_unnorm = unnormalize_data(current_proprioception.cpu().numpy(), stats=stats["proprioception"], type=cfgs.dataset.norm_type)[0] # [10]
        current_proprioception_unnorm_pos = current_proprioception_unnorm[:3].copy() # [3]
        current_proprioception_unnorm_ortho6d = current_proprioception_unnorm[3:9].copy()
        current_proprioception_unnorm_rot = compute_rotation_matrix_from_ortho6d(current_proprioception_unnorm_ortho6d[None])[0]  # [3, 3] # marker coordinate
        relative_pos = action_pred[..., :3].copy() # [h, 3], {t+1}-{t}, {t+2}-{t+1}, ...
        relative_rotvec = action_pred[..., 3:6].copy() # [h, 3] # axis angle
        relative_rot = R.from_rotvec(relative_rotvec).as_matrix() # [h, 3, 3]
        
        action_pred_pos = current_proprioception_unnorm_pos[None] + np.cumsum(relative_pos, axis=0) # [h, 3], {t+1}, {t+2}, {t+3}, ...
        # cumulative product of rotation matrices (itertools.accumulate)
        rot_objects = [R.from_matrix(relative_rot[i]) for i in range(len(relative_rot))]
        cumulative_rots = list(accumulate(rot_objects, lambda acc, r: acc * r, initial=R.identity()))[1:]
        cumulative_rot_matrices = np.array([r.as_matrix() for r in cumulative_rots])
        action_pred_rot = np.einsum('ij,tjk->tik', current_proprioception_unnorm_rot, cumulative_rot_matrices) # [h, 3, 3]
        action_pred_ortho6d = compute_ortho6d_from_rotation_matrix(action_pred_rot) # [h, 6]
        action_pred = np.concatenate([action_pred_pos, action_pred_ortho6d, action_gripper], axis=-1) # [h, 10]
    else:
        action_pred = unnormalize_data(naction.copy(), stats=stats["action"], type=cfgs.dataset.action_norm_type) # 10D for absolute, 7D for relative

    action_start = 0 # obs_horizon - 1
    action_end = action_start + pred_horizon
    action = action_pred[action_start:action_end, :]
    
    # convert policy's action 
    # (0: open, 1: close) to z1's gripper action (0: close, -1: open)
    # (0: open, 1: close) to widowx's gripper action: (0.001: close, 0.039: open)
    gripper_action = action[..., -1].copy()
    gripper_action = np.clip(gripper_action, 0.0, 1.0)
    
    gripper_action = np.where(gripper_action < 0.5, 0.0, 1.0)

    action[..., -1] = gripper_action

    if return_attn:
        return action, flow_pred_unnorm, attentions
    return action, flow_pred_unnorm


def evaluate_active_vision_wo_env(
    model,
    view_model,
    view_num_inference_steps,
    stats,
    num_samples,
    eval_dataset,
    data_buffers,
    result_save_path,
    seed=42,
    downsample_ratio=None,
    use_masking = False,
    cfgs = None,
    use_fake_depth = False,
    slam_fake_value = 0,  # depth value assigned to masked (dynamic) pixels
    use_depth_estimate = False,
    policy_type="diffusion",
    tracker_type = "cotracker3",
    visualize_raw_view_policy_output = True,
    droid = False,
    T_B_M = None,
    T_B_M_view = None,
    T_E_C_view = None,
    ee_bias = None,
    use_mask_depth_filter=False, 
    mask_point_tracking_threshold=0.1, 
    num_points_per_mask = 64,
    reward_type_list = ['visibility+close_to_query_points', 'visibility+close_to_nominal', 'visibility'],  # List of reward types to evaluate: ['visibility', 'visibility+close_to_nominal', 'visibility+close_to_query_points']
    flow_model = None,  # flow_model for predicting future point flow when use_flow_model=True
    visibility_reward_type = 'wo_consider_tracking_lost',
    use_retargeting = True,
    use_visibility_aware_retargeting = False,
    use_future_action_for_flow_model = True,  # If True, use model's predicted action as future_action condition for flow_model
    query_point_indices = None,  # List of query point indices to use (None means use all query points)
    z1_urdf_path = None,  # view (camera) arm description
    z1_mesh_base_path = None,
    widowx_urdf_path = None,  # manipulation arm description
    widowx_mesh_base_path = None,
    cotracker_checkpoint = None,  # CoTracker3 online checkpoint (scaled_online.pth)
    max_steps = None,  # evaluate only the first steps of every episode (None: whole episode)
):
    assert downsample_ratio is not None, "downsample_ratio must be provided"
    
    # SVDD settings of the offline evaluation (override the training config)
    cfgs.svdd.batch_size = 3
    cfgs.svdd.close_to_nominal_weight = 0.1
    cfgs.svdd.close_to_query_points_weight = 0.1
    cfgs.svdd.duplicate = 10

    use_flow_model = cfgs.training.use_flow_model if hasattr(cfgs.training, 'use_flow_model') else False

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model.eval()
    
    eval_dataset.stats = stats
    data_path_offset = eval_dataset.buffer.data_path_offset
    print(">> data_path_offset", data_path_offset)
    action_dim = eval_dataset.action_dim
    pred_horizon = eval_dataset.pred_horizon
    input_horizon = pred_horizon if not cfgs.training.use_ptp else pred_horizon + cfgs.dataset.obs_horizon - 1
    flow_norm_type = eval_dataset.flow_norm_type
    pixel_tracking_data = eval_dataset.pixel_tracking_data
    task_description = eval_dataset.text
    
    extrinsics_data = eval_dataset.extrinsics # [num_episode, T, 4, 4]
    use_marker_coordinate = eval_dataset.use_marker_coordinate
    if use_marker_coordinate:
        T_mc_opt = eval_dataset.T_mc_opt

    backward_episode_len = getattr(eval_dataset.buffer, "backward_episode_len", 10000000)
    
    episode_ends = eval_dataset.episode_ends

    if eval_dataset.use_dift:
        dift_points = eval_dataset.dift_points

    debug_episode = [0]

    
    T_B_M = np.array(T_B_M, dtype=np.float32)
    T_B_M_view = np.array(T_B_M_view, dtype=np.float32)
    T_E_C_view = np.array(T_E_C_view, dtype=np.float32)
    ee_bias = np.array(ee_bias, dtype=np.float32)


    if "cotracker3" in tracker_type:
        from cotracker.predictor import CoTrackerOnlinePredictor
        tracker_model = CoTrackerOnlinePredictor(checkpoint=cotracker_checkpoint).to('cuda')
        tracker_model.requires_grad_(False)

    if result_save_path is not None:
        os.makedirs(result_save_path, exist_ok=True)
    print("Evaluation store at:", result_save_path)

    for db, data_buffer in enumerate(data_buffers):
        buffer_result_save_path = os.path.join(result_save_path, f"buffer_{db}")
        os.makedirs(buffer_result_save_path, exist_ok=True)
        buffer_offset = data_path_offset[db]
        
        for i in tqdm(range(num_samples)):

            # Iterate over reward types for the same episode
            for reward_type_idx, reward_type in enumerate(reward_type_list):
                print(f"\n=== Processing Episode {i} with reward_type: {reward_type} ===")

                # Initialize visualization data list for this reward_type
                visualization_data_list = []

                # Initialize episode-level lists for visibility rewards
                episode_visibility_rewards = []
                episode_visibility_rewards_among_tracking = []
                episode_visibility_rewards_wo_tracking_lost = []
                episode_active_visibility_rewards = []
                episode_active_visibility_rewards_among_tracking = []
                episode_active_visibility_rewards_wo_tracking_lost = []
                episode_active_visibility_rewards_retargeted = []
                episode_active_visibility_rewards_among_tracking_retargeted = []
                episode_active_visibility_rewards_wo_tracking_lost_retargeted = []

                track_2d_list = []
                episode_T_mc_list = []
                episode_track_2d_list = []
                episode_track_3d_list = []
                episode_index = buffer_offset + i
                episode_start_index = (
                    episode_ends[episode_index - 1] if episode_index > 0 else 0
                )
                episode_length = episode_ends[episode_index] - episode_start_index
                # --- Load episode data ---
                episode_point_tracking = eval_dataset.episodic_point_tracking_data[episode_index].copy() # 3D (T(padding),N,3 or 4)
                episode_pixel_tracking = pixel_tracking_data[episode_index].copy() # 2D (T, N, 3)
                episode_extrinsics = extrinsics_data[episode_index].copy() # [T(padding), 4, 4]
                episode_text = task_description[episode_index] # Use string directly, not numpy array
                initial_pixel_tracking = pixel_tracking_data[episode_index].copy()[0]  # (N,3)
                if use_marker_coordinate:
                    episode_T_mc_opt = T_mc_opt[episode_index].copy() # [T(padding), 4, 4]
                else:
                    episode_T_mc_opt = None
                
                print(f'episode_index : {episode_index}, episode text : {episode_text}, length : {episode_length}')
                
                if eval_dataset.use_dift:
                    # Get DIFT points for current episode
                    episode_dift_points = dift_points[episode_index].copy()  # [N, 2]
                    num_dift_points = len(episode_dift_points)
                    
                else:
                    episode_dift_points = None
                    
                    
                # random sample pixels and normalize roughly to [0, 1]
                episode_pixel_tracking, pixel_tracking_random_sample_indices = eval_dataset.process_pixel_tracking_data(
                    episode_pixel_tracking,
                    # eval_dataset.num_points if not eval_dataset.use_dift else num_dift_points,
                    eval_dataset.point_tracking_img_size, # point_tracking_img_size is [724(w), 543(h)] (TAPIP3D) 
                )
                # --- Unnormalize 3D points ---
                pt_min = eval_dataset.pt_min
                pt_max = eval_dataset.pt_max
                pt_mean = eval_dataset.pt_mean
                pt_std = eval_dataset.pt_std
                print(f'pt_min: {pt_min}, pt_max: {pt_max}, pt_mean: {pt_mean}, pt_std: {pt_std}')
                
                # initial flow (2D)
                initial_pixel_flow = episode_pixel_tracking[0].copy() # normalized [N, 3 (x,y, vis)]
                initial_pixel_flow[:, 0] *= eval_dataset.point_tracking_img_size[0]
                initial_pixel_flow[:, 1] *= eval_dataset.point_tracking_img_size[1]
                initial_pixel_flow = initial_pixel_flow.astype(np.int32)
                initial_pixel_flow[:, :2] = np.clip(
                    initial_pixel_flow[:, :2], a_min=np.zeros(2), a_max=np.array(eval_dataset.point_tracking_img_size) - 1
                ) # unnormalized (N,2)

                # --- Load RGB, depth, intrinsics, extrinsics ---
                rgb = parallel_reading(
                    group=data_buffer[f"episode_{i}/camera_0"],
                    array_name="rgb",
                )[-backward_episode_len:][::downsample_ratio]
                if use_depth_estimate:
                    depth = data_buffer[f"episode_{i}/depth_estimate"][-backward_episode_len:][::downsample_ratio]
                else:
                    depth = parallel_reading(
                        group=data_buffer[f"episode_{i}/camera_0"],
                        array_name="depth",
                    )[-backward_episode_len:][::downsample_ratio].astype(np.float32) / 1000.0 # uint16 -> meter 
                    
                    depth = fill_depth_zeros(depth)
                

                # remove dynamic objects for mesh / slam
                mask = data_buffer[f"episode_{i}/sam_mask_sequence_multi_obj"][-backward_episode_len:][:] # [t, num_obj(obj, hand), h, w]
                mask = mask[::downsample_ratio].copy()
                assert rgb.shape[0] == mask.shape[0], f"rgb shape: {rgb.shape}, mask shape: {mask.shape}"
                print(f"mask.shape: {mask.shape}")
                
                # Combine all objects into a single mask [t, h, w]
                # Any pixel that is 1 in any object becomes 1 in the combined mask
                combined_mask = np.any(mask == 1, axis=1)  # [t, h, w]
                print(f"combined_mask.shape: {combined_mask.shape}")

                if use_fake_depth:
                    # additional_depth_preprocess
                    # 1. dilate the mask (margin) - per frame
                    dilated_mask = np.zeros_like(combined_mask, dtype=bool)
                    for t in range(combined_mask.shape[0]):
                        dilated_mask[t] = cv2.dilate(combined_mask[t].astype(np.uint8), np.ones((3,3), np.uint8), iterations=2).astype(bool)
                    combined_mask = dilated_mask
                    
                    print("Applying fake depth (0) to masked regions...")
                    
                    depth_with_fake_depth = apply_fake_depth_to_mask(depth, combined_mask, slam_fake_value)

                    # 2. remove small islands of valid depth (connected components) - per frame
                    valid = (depth_with_fake_depth > 0)  # check > 0 instead of np.nan
                    min_area = 20  # adjust to the image resolution
                    
                    for t in range(valid.shape[0]):
                        cnt, labels, stats_cv, _ = cv2.connectedComponentsWithStats(valid[t].astype(np.uint8), 8)
                        for c in range(1, cnt):
                            if stats_cv[c, cv2.CC_STAT_AREA] < min_area:
                                depth_with_fake_depth[t][labels == c] = slam_fake_value


                intrinsics = data_buffer[f"episode_{i}/intrinsics"][:].copy() # [3, 3]
                
                proprioceptions = data_buffer[f"episode_{i}/proprioception"][-backward_episode_len:][::downsample_ratio].copy() # [T, dim] # unnormalized
                
                # NOTE: later for comparison with g.t.
                actions_gt = data_buffer[f"episode_{i}/action"][-backward_episode_len:][::downsample_ratio].copy() # [T, dim] # unnormalized
                
                if use_marker_coordinate:
                    # convert to marker coordinate
                    if droid:
                        T_mc_transformation = data_buffer[f"episode_{i}/T_mc_opt_droid"][-backward_episode_len:][::downsample_ratio].copy() # [T, 4, 4]
                    else:
                        T_mc_transformation = data_buffer[f"episode_{i}/T_mc_opt"][-backward_episode_len:][::downsample_ratio].copy() # [T, 4, 4]
                    assert len(proprioceptions) == len(T_mc_transformation), f"proprioceptions.shape: {proprioceptions.shape}, T_mc_transformation.shape: {T_mc_transformation.shape}"
                    
                    assert (episode_T_mc_opt[:T_mc_transformation.shape[0]].astype(np.float32) == T_mc_transformation.astype(np.float32)).all() 
                    
                    # Extract xyz positions and euler angles from proprioception
                    xyz_positions = proprioceptions[:, :3]  # (T, 3)
                    ortho6d = proprioceptions[:, 3:9]  # (T, 6)
                    
                    # Transform proprioception positions and euler angles to marker coordinates
                    proprioception_marker_xyz = eval_dataset.buffer.transform_points_to_marker_coords(xyz_positions, T_mc_transformation)
                    proprioception_marker_ortho6d = eval_dataset.buffer.transform_ortho6d_to_marker_coords(ortho6d, T_mc_transformation)
                    
                    # Combine transformed xyz and euler angles with rest of data
                    proprioceptions = np.concatenate([
                        proprioception_marker_xyz,  # (T, 3)
                        proprioception_marker_ortho6d,  # (T, 6)
                        proprioceptions[:, -1:]  # (T, 1) - gripper action
                    ], axis=1)

                    # Extract xyz positions and euler angles from action
                    action_xyz = actions_gt[:, :3]  # (T, 3)
                    action_ortho6d = actions_gt[:, 3:9]  # (T, 6)
                    
                    # Transform action positions and euler angles to marker coordinates
                    action_marker_xyz = eval_dataset.buffer.transform_points_to_marker_coords(action_xyz, T_mc_transformation)
                    action_marker_ortho6d = eval_dataset.buffer.transform_ortho6d_to_marker_coords(action_ortho6d, T_mc_transformation)
                    
                    # Combine transformed xyz and euler angles with rest of data
                    actions_gt = np.concatenate([
                        action_marker_xyz,  # (T, 3)
                        action_marker_ortho6d,  # (T, 6)
                        actions_gt[:, -1:]  # (T, 1) - gripper action
                    ], axis=1)

                if not (rgb[0].shape[1] == eval_dataset.pixel_tracking_img_size[0] and rgb[0].shape[0] == eval_dataset.pixel_tracking_img_size[1]):
                    intrinsics[0, :] *= (eval_dataset.pixel_tracking_img_size[0] - 1) / (rgb[0].shape[1] - 1)
                    intrinsics[1, :] *= (eval_dataset.pixel_tracking_img_size[1] - 1) / (rgb[0].shape[0] - 1)

                # use logged 2d pixel points since in real-world evaluation, we should project 2d pixel into 3D, rather than backprojecting 3D to 2D
                query_points_2d = initial_pixel_flow[..., :2] # [N, 2]
                
                # NOTE: the order of the last axis should be [0, y(height), x(width) ]
                query_points = np.concatenate([np.zeros((query_points_2d.shape[0], 1)), query_points_2d[..., ::-1]], axis=-1) # (N, 3)
                
                # frame sequence for pixel tracking (resized to pixel_tracking_img_size)
                pixel_tracking_frames = [cv2.resize(img, tuple(eval_dataset.pixel_tracking_img_size), interpolation=cv2.INTER_LINEAR) for img in rgb]
                pixel_tracking_depths = [cv2.resize(img, tuple(eval_dataset.pixel_tracking_img_size), interpolation=cv2.INTER_LINEAR) for img in depth]

                if use_fake_depth:
                    pixel_tracking_depths_fake_depth = [cv2.resize(img, tuple(eval_dataset.pixel_tracking_img_size), interpolation=cv2.INTER_LINEAR) for img in depth_with_fake_depth]
                else:
                    pixel_tracking_depths_fake_depth = pixel_tracking_depths.copy()
                

                frames = []

                
                # 5. initialize query_features, causal_state (frames/intrinsics for pixel tracking)
                point_tracking_frame = pixel_tracking_frames[0].copy()
                
                if "cotracker3" in tracker_type:
                    window_frames = [pixel_tracking_frames[0] for _ in range(tracker_model.model.window_len)]
                    video_chunk = torch.tensor(
                        np.stack(window_frames), dtype=torch.float32, device='cuda'
                    ).permute(0, 3, 1, 2)[None]
                    queries = np.zeros_like(query_points[None]) # [B(1), N, 3]
                    queries[..., 1] = query_points[..., 2] # x
                    queries[..., 2] = query_points[..., 1] # y
                    queries = torch.from_numpy(queries).float().cuda() # [B(1), N, 3]

                    # this should be called only once for the first frame
                    tracker_model(
                        video_chunk=video_chunk[0, 0].unsqueeze(0).unsqueeze(0),
                        is_first_step=True,
                        add_support_grid=True,
                        queries=queries,
                    )
                    
                    pred_tracks, pred_visibility = tracker_model(video_chunk, one_frame=True)
                    pred_tracks = pred_tracks[0, -1, :queries.shape[1], :] # [B(1), T(1), N, 3] -> [N, 2]
                    pred_visibility = pred_visibility[0, -1, :queries.shape[1]].unsqueeze(-1) # [B(1), T(1), N] -> [N, 1]
                    current_flow_2d = np.concatenate([pred_tracks.detach().cpu().numpy(), pred_visibility.detach().cpu().numpy()], axis=-1)[:, None] # [N, T(1), 3]
                
                current_flow_2d[..., :2] = np.clip(
                    current_flow_2d[..., :2], a_min=np.zeros(2), a_max=np.array(eval_dataset.pixel_tracking_img_size) - 1
                )
                current_flow_2d_norm = current_flow_2d.copy()
                current_flow_2d_norm[..., 0] = current_flow_2d_norm[..., 0] / eval_dataset.pixel_tracking_img_size[0]
                current_flow_2d_norm[..., 1] = current_flow_2d_norm[..., 1] / eval_dataset.pixel_tracking_img_size[1]
                

                track_2d_list.append(current_flow_2d.copy().astype(np.int32))

                from collections import deque
                point_flow_queue = deque(maxlen=eval_dataset.obs_horizon)
                pixel_flow_queue = deque(maxlen=eval_dataset.obs_horizon)
                proprioception_queue = deque(maxlen=eval_dataset.obs_horizon)
                current_extrinsics_queue = deque(maxlen=eval_dataset.obs_horizon)
                

                # Initialize episode-level mesh and robot visualizer for active vision
                print("Initializing episode mesh and robot visualizer...")
            
                
                relative_poses = convert_to_relative(T_mc_transformation.copy()[None])[0] # [T, 4, 4]
                
                
                # nvblox Mapper settings (as in the nvblox sun3d.py example)
                projective_integrator_params = ProjectiveIntegratorParams()
                projective_integrator_params.projective_integrator_max_integration_distance_m = 3.0 # default:5.0

                # NOTE: raise the min weight to remove tiny mesh fragments
                mesh_param = MeshIntegratorParams()
                mesh_param.mesh_integrator_min_weight = 1.0 # default 1e-4
                

                mapper_params = MapperParams()
                mapper_params.set_projective_integrator_params(projective_integrator_params)
                mapper_params.set_mesh_integrator_params(mesh_param)


                # create the Mapper
                episode_mapper = Mapper(
                    voxel_sizes_m=VOXEL_SIZE,
                    mapper_parameters=mapper_params,
                )
                
                
                visualizer = CustomVisualizer(
                    video_save=True,
                    result_save_path=buffer_result_save_path,
                    follow_camera_view=True,
                    active_camera=True,
                    vis_window=False

                )

                if not visualizer.follow_camera_view:
                    first_camera_pose = np.eye(4)
                    visualizer.set_initial_camera_pose(first_camera_pose)
                    print("Set initial camera pose from first frame (static mode)")
                elif visualizer.follow_camera_view:
                    print("Camera following mode enabled - camera will follow each frame's pose")


                # Initialize robot visualizer
                episode_view_robot_viz = Z1RobotVisualizer(z1_urdf_path, z1_mesh_base_path, T_E_C=T_E_C_view)
                episode_robot_viz = WidowXVisualizer(widowx_urdf_path, widowx_mesh_base_path, ee_bias = ee_bias)

                assert use_marker_coordinate
                # fixed valeus during episode
                T_mw = T_mc_transformation[0].copy()
                
                # varying valeus during episode
                T_mc = T_mc_transformation[0]
                T_W_C = np.linalg.inv(T_mw) @ T_mc
                T_W_M = np.linalg.inv(T_mw)

                
                actions_with_masking = []
                actions_with_gt_future_flow = []
                actions_retargeted_list = []
                actions_gt_list = []
                gt_future_flow_list = []
                pred_future_flow_list = []
                action_mse_with_masking_list = []
                action_mse_with_gt_future_flow_list = []
                gt_future_flow_action_errors = []
                proprioceptions_list = []
                view_actions_list = []
                view_actions_svdd_list = []
                view_actions_svdd_retargeted_list = []
                viewpoint_gt_list = []


                scene_mesh, pose_tensor = get_scene_mesh(pixel_tracking_frames[0], pixel_tracking_depths_fake_depth[0], intrinsics, relative_poses[0], VOXEL_SIZE, episode_mapper)
                        

                print("Episode mesh and robot visualizer initialized successfully")

                # Initialize lists to collect visibility rewards throughout the episode
                episode_visibility_rewards = []
                episode_active_visibility_rewards = []  # Will store list of rewards for each batch index at each timestep
                episode_visibility_rewards_among_tracking = []
                episode_active_visibility_rewards_among_tracking = []  # Will store list of rewards for each batch index at each timestep
                episode_visibility_rewards_wo_tracking_lost = []
                episode_active_visibility_rewards_wo_tracking_lost = []  # Will store list of rewards for each batch index at each timestep
                
                # Initialize lists to collect detailed reward components from SVDD
                episode_svdd_total_rewards = []
                episode_svdd_visibility_rewards = []
                episode_svdd_close_to_nominal_rewards = []
                episode_svdd_close_to_query_points_rewards = []
                episode_svdd_smoothness_rewards = []
                episode_svdd_best_indices = []  # Store best index for each step
                
                # Initialize lists to collect retargeted reward components from SVDD
                episode_svdd_total_rewards_retargeted = []
                episode_svdd_visibility_rewards_retargeted = []
                episode_svdd_close_to_nominal_rewards_retargeted = []
                episode_svdd_close_to_query_points_rewards_retargeted = []
                episode_svdd_smoothness_rewards_retargeted = []
                n_steps = len(rgb) if max_steps is None else min(max_steps, len(rgb))
                for step in range(n_steps):
                    # use the stored (marker-coordinate) extrinsics
                    current_extrinsics = episode_T_mc_opt[step]
                    tracked_3d = get_pixel_queries(torch.from_numpy(current_flow_2d[:,0,:2]).float(), 
                                                torch.from_numpy(pixel_tracking_depths[step]).float(), 
                                                torch.from_numpy(intrinsics).float(), 
                                                torch.from_numpy(current_extrinsics).float(),
                                                ).detach().cpu().numpy()
                
                
                    if step == 0:
                        for _ in range(eval_dataset.obs_horizon):
                            current_extrinsics_queue.append(current_extrinsics[None].copy()) 
                    else:
                        current_extrinsics_queue.append(current_extrinsics[None].copy()) # [1, 4, 4]
                    

                    tracked_3d = tracked_3d[..., -3:] # (N,4(0,x,y,z)) -> (N,3(x,y,z)) 
                    # visibility from the 2D tracker
                    visible = current_flow_2d[:, 0, -1] # [N] 
                    mask_indices = torch.from_numpy(np.where(visible == 0)[0][None]).to('cuda') # [B(1), N]

                    if eval_dataset.flow_dim == 4: 
                        # current_flow_2d [N, T(1), 3]
                        assert current_flow_2d.shape[1] == 1, f"current_flow_2d.shape: {current_flow_2d.shape}"
                        tracked_3d = np.concatenate([tracked_3d, visible[..., None]], axis=-1) # [N, 4 (x,y,z, visible)]
                    
                    # Apply mask-based depth filtering to tracked_3d's visibility before normalization
                    if use_mask_depth_filter and mask is not None and depth is not None:
                        print('use mask depth filter')
                        # Get mask and depth for the current step
                        sam_mask_step = mask[step]  # [num_obj, H, W]
                        depth_step = depth[step]  # [H, W]
                        
                        # Apply mask-based depth filtering to tracked_3d
                        tracked_3d = apply_mask_depth_filter_to_tracked_3d(
                            tracked_3d,
                            sam_mask_step,
                            depth_step,
                            intrinsics,
                            current_extrinsics,
                            mask_point_tracking_threshold=mask_point_tracking_threshold,
                            num_points_per_mask=num_points_per_mask
                        )
                        
                        # Update visible and mask_indices based on modified tracked_3d
                        visible = tracked_3d[:, -1]  # [N] - updated visibility
                        mask_indices = torch.from_numpy(np.where(visible == 0)[0][None]).to('cuda')  # [B(1), N]
                    
                    visibility = torch.from_numpy(visible[None, :]).float().to('cuda') # [B(1), N]

                    tracked_3d_unnormalized = tracked_3d.copy()
                    episode_T_mc_list.append(current_extrinsics.copy())
                    episode_track_2d_list.append(current_flow_2d[:, 0].copy())
                    episode_track_3d_list.append(tracked_3d_unnormalized.copy())

                    
                    if flow_norm_type == 'min_max':
                        # normalize [0, 1]
                        tracked_3d = (tracked_3d - pt_min[None]) / (pt_max[None] - pt_min[None])
                        # normalize [-1, 1] via flow_transforms
                        assert len(tracked_3d.shape) == 2, f"tracked_3d.shape: {tracked_3d.shape}"
                        tracked_3d = eval_dataset.flow_transforms(tracked_3d.transpose(1,0)) # [3 or 4, N]
                        tracked_3d = tracked_3d.transpose(1,0) # [N, 3 or 4]
                    elif flow_norm_type == 'mean_std':
                        # normalize [-1, 1]
                        tracked_3d = (tracked_3d - pt_mean[None]) / (pt_std[None] + 1e-10)
                        assert len(tracked_3d.shape) == 2, f"tracked_3d.shape: {tracked_3d.shape}"
                    
                    
                    current_flow_tensor = torch.from_numpy(tracked_3d).float().unsqueeze(0).cuda() # (B(1),N,3 or 4)
                    current_pixel_flows_tensor = torch.from_numpy(current_flow_2d_norm[:, 0]).float().unsqueeze(0).cuda() # (B(1),N,3)

                    # Handle padding when step+1+pred_horizon exceeds the available data length
                    start_idx = step + 1
                    end_idx = step + 1 + eval_dataset.pred_horizon
                    assert start_idx <= episode_length, f"start_idx: {start_idx}, episode_length: {episode_length}"
                    
                    # Use the actual episode length for bounds checking
                    if end_idx <= episode_length:
                        # Normal case: we have enough data
                        if not use_marker_coordinate:
                            future_extrinsics = episode_extrinsics[start_idx:end_idx] # [pred_horizon, 4, 4]
                        else:
                            assert len(episode_extrinsics) == len(episode_T_mc_opt), f"episode_extrinsics.shape: {episode_extrinsics.shape}, episode_T_mc_opt.shape: {episode_T_mc_opt.shape}"
                            future_extrinsics = episode_T_mc_opt[start_idx:end_idx] # [pred_horizon, 4, 4]


                    else:
                        # Need to pad with the last available index
                        if not use_marker_coordinate:
                            available_future_extrinsics = episode_extrinsics[start_idx:episode_length] # [pred_horizon, 4, 4]
                        else:
                            available_future_extrinsics = episode_T_mc_opt[start_idx:episode_length] # [pred_horizon, 4, 4]
                            
                        remaining_length_extrinsics = eval_dataset.pred_horizon - len(available_future_extrinsics)
                        if remaining_length_extrinsics > 0:
                            # Pad with the last available frame 
                            if not use_marker_coordinate:
                                last_frame = episode_extrinsics[episode_length-1:episode_length] # [1, 4, 4]
                            else:
                                last_frame = episode_T_mc_opt[episode_length-1:episode_length] # [1, 4, 4]
                            padding = np.tile(last_frame, (remaining_length_extrinsics, 1, 1)) # [remaining_length_extrinsics, 4, 4]
                            future_extrinsics = np.concatenate([available_future_extrinsics, padding], axis=0) # [pred_horizon, 4, 4]
                        else:
                            raise NotImplementedError('this code should not be used')

                    # Compute ground truth viewpoint from future_extrinsics
                    viewpoint_gt = convert_se3_to_9d(future_extrinsics[None])[0]  # [pred_horizon, 9], marker coordinate

                    # 3. prepare model inputs (obs_horizon, flow, etc. are fixed)
                    proprioceptions_norm = normalize_data(proprioceptions[step:step+1], stats=stats["proprioception"], type=cfgs.dataset.norm_type)
                    proprioceptions_tensor = torch.from_numpy(proprioceptions_norm).float().cuda() # [T(1), dim]
                    
                    
                    text = None # [episode_text]
                    
                    
                    if step == 0 :
                        for _ in range(eval_dataset.obs_horizon):
                            point_flow_queue.append(current_flow_tensor[:, None].detach().cpu().numpy())
                            pixel_flow_queue.append(current_pixel_flows_tensor[:, None].detach().cpu().numpy())
                            proprioception_queue.append(proprioceptions_tensor[None].detach().cpu().numpy())
                    else:
                        point_flow_queue.append(current_flow_tensor[:, None].detach().cpu().numpy())
                        pixel_flow_queue.append(current_pixel_flows_tensor[:, None].detach().cpu().numpy())
                        proprioception_queue.append(proprioceptions_tensor[None].detach().cpu().numpy())
                    
                    point_flow = torch.from_numpy(np.concatenate(point_flow_queue, axis=1)).float().cuda() # [B(1), obs_horizon, N, 3 or 4]
                    pixel_flow = torch.from_numpy(np.concatenate(pixel_flow_queue, axis=1)).float().cuda() # [B(1), obs_horizon, N, 3]
                    proprioception = torch.from_numpy(np.concatenate(proprioception_queue, axis=1)).float().cuda() # [B(1), obs_horizon, dim]
                    
                    current_point_flow = point_flow[:, -1] # [B(1), N, 3 or 4]
                    current_pixel_flow = pixel_flow[:, -1] # [B(1), N, 3]
                    condition_flow = pixel_flow if use_pixel_flow_for_robot_policy(cfgs) else point_flow
                    if cfgs.training.use_ptp and not cfgs.use_relative:
                        current_proprioception = proprioception
                    else:
                        current_proprioception = proprioception[:, -1] # [B(1), dim]
                    

                    if policy_type == "diffusion":
                        # Get action prediction from model
                        if use_pixel_flow_for_robot_policy(cfgs):
                            robot_policy_flow = pixel_flow
                            current_robot_policy_flow = current_pixel_flow
                        else:
                            robot_policy_flow = point_flow
                            current_robot_policy_flow = current_point_flow
                        action, flow_pred_unnorm = get_policy_output(model, robot_policy_flow, proprioception, stats, cfgs, action_dim, pred_horizon, input_horizon, current_proprioception, current_robot_policy_flow, visibility, eval_dataset)
                        
                        
                        actions_with_masking.append(action.copy()) # list of [h, dim]
                        
                        # Apply retargeting to action
                        # Get current proprioception (unnormalized) for initial guess
                        current_proprioception_unnorm = None
                        if step < len(proprioceptions):
                            current_proprioception_unnorm = unnormalize_data(
                                proprioceptions[step:step+1], 
                                stats=stats["proprioception"], 
                                type=cfgs.dataset.norm_type
                            )[0]  # [10] marker coordinate
                        
                        action_retargeted = retargeting(episode_robot_viz, action, T_B_M, current_proprioception_unnorm)
                        actions_retargeted_list.append(action_retargeted.copy()) # list of [h, dim]
                        
                        start_idx = step
                        end_idx = start_idx + action.shape[0]
                        
                        # Handle actions_gt_list with padding if needed
                        if end_idx <= len(actions_gt):
                            # Normal case: we have enough data
                            actions_gt_slice = actions_gt[start_idx:end_idx].copy()
                        else:
                            # Need to pad with the last available index
                            available_actions_gt = actions_gt[start_idx:len(actions_gt)].copy()
                            remaining_length = action.shape[0] - len(available_actions_gt)
                            if remaining_length > 0:
                                # Pad with the last available frame
                                last_frame = actions_gt[len(actions_gt)-1:len(actions_gt)].copy() # [1, dim]
                                padding = np.tile(last_frame, (remaining_length, 1)) # [remaining_length, dim]
                                actions_gt_slice = np.concatenate([available_actions_gt, padding], axis=0) # [action.shape[0], dim]
                            else:
                                actions_gt_slice = available_actions_gt
                        
                        # Handle proprioceptions_list with padding if needed
                        if end_idx <= len(proprioceptions):
                            # Normal case: we have enough data
                            proprioceptions_slice = proprioceptions[start_idx:end_idx].copy()
                        else:
                            # Need to pad with the last available index
                            available_proprioceptions = proprioceptions[start_idx:len(proprioceptions)].copy()
                            remaining_length = action.shape[0] - len(available_proprioceptions)
                            if remaining_length > 0:
                                # Pad with the last available frame
                                last_frame = proprioceptions[len(proprioceptions)-1:len(proprioceptions)].copy() # [1, dim]
                                padding = np.tile(last_frame, (remaining_length, 1)) # [remaining_length, dim]
                                proprioceptions_slice = np.concatenate([available_proprioceptions, padding], axis=0) # [action.shape[0], dim]
                            else:
                                proprioceptions_slice = available_proprioceptions

                        actions_gt_list.append(actions_gt_slice) # list of [h, dim]
                        proprioceptions_list.append(proprioceptions_slice) # list of [h, dim]
                        action_mse_with_masking_list.append(_compute_action_mse(action, actions_gt_slice))
                        pred_future_flow_list.append(flow_pred_unnorm.copy() if flow_pred_unnorm is not None else None)

                        action_with_gt_future_flow = None
                        gt_future_flow_unnorm = None
                        if not use_pixel_flow_for_robot_policy(cfgs):
                            try:
                                gt_future_flow_unnorm = _pad_future_slice(
                                    episode_point_tracking,
                                    step + 1,
                                    pred_horizon,
                                    episode_length,
                                )
                                if gt_future_flow_unnorm.shape[1:] != tracked_3d_unnormalized.shape:
                                    raise ValueError(
                                        f"GT future flow shape {gt_future_flow_unnorm.shape[1:]} does not match current tracked flow {tracked_3d_unnormalized.shape}"
                                    )
                                gt_future_flow_norm = _normalize_flow_sequence(
                                    gt_future_flow_unnorm,
                                    eval_dataset,
                                    flow_norm_type,
                                )
                                gt_future_flow_tensor = torch.from_numpy(gt_future_flow_norm[None]).float().cuda()
                                proprioception_for_gt_flow = proprioception[:, -1:, :].repeat(1, gt_future_flow_tensor.shape[1], 1)
                                action_with_gt_future_flow, _ = get_policy_output(
                                    model,
                                    gt_future_flow_tensor,
                                    proprioception_for_gt_flow,
                                    stats,
                                    cfgs,
                                    action_dim,
                                    pred_horizon,
                                    input_horizon,
                                    current_proprioception,
                                    current_point_flow,
                                    visibility,
                                    eval_dataset,
                                )
                                actions_with_gt_future_flow.append(action_with_gt_future_flow.copy())
                                action_mse_with_gt_future_flow_list.append(
                                    _compute_action_mse(action_with_gt_future_flow, actions_gt_slice)
                                )
                            except Exception as exc:
                                print(f"Warning: failed to compute action with GT future flow at episode {i}, step {step}: {exc}")
                                actions_with_gt_future_flow.append(None)
                                action_mse_with_gt_future_flow_list.append(None)
                                gt_future_flow_action_errors.append({'step': step, 'error': str(exc)})
                        else:
                            actions_with_gt_future_flow.append(None)
                            action_mse_with_gt_future_flow_list.append(None)
                            gt_future_flow_action_errors.append({
                                'step': step,
                                'error': 'Skipped because robot policy is configured to use pixel flow.',
                            })
                        gt_future_flow_list.append(gt_future_flow_unnorm.copy() if gt_future_flow_unnorm is not None else None)


                    if use_marker_coordinate:
                        if cfgs.training.use_ptp and not cfgs.use_relative:
                            current_viewpoint = np.concatenate(current_extrinsics_queue, axis=0) # [h, 4, 4]
                            current_viewpoint = convert_se3_to_9d(current_viewpoint[None])[0] # [h, 9] # marker coordinate
                            current_viewpoint_norm = normalize_data(current_viewpoint, stats=stats["view_actions"], type=cfgs.dataset.action_norm_type) # [h, 9]
                        else:
                            current_viewpoint = convert_se3_to_9d(current_extrinsics[None, None])[0][0] # marker coordinate
                            current_viewpoint_norm = normalize_data(current_viewpoint[None], stats=stats["view_actions"], type=cfgs.dataset.action_norm_type)[0] # [9]
                    
                    
                    if visualize_raw_view_policy_output:
                        # raw view policy output (before SVDD)
                        # Use CustomRDTRunner's predict_action method for view model
                        # Convert point_flow to flow_tokens: [B, H, N, D] -> [B, N, H*D] for point-wise positional embedding
                        B_view, H_view, N_view, D_view = condition_flow.shape
                        
                        # Prepare tokens for view_model
                        # If view_obs_horizon == 1, use only the latest timestep (index -1)
                        if cfgs.dataset.view_obs_horizon == 1:
                            # Extract only the last timestep for view_model
                            # Flow tokens: extract last timestep before reshaping
                            view_flow_tokens = build_flow_tokens(condition_flow[:, -1:, :, :], get_robot_policy_input_flow_dim(cfgs))
                            
                            # Proprioception tokens: extract last timestep
                            view_proprio_tokens = proprioception[:, -1:, :]  # [B, 1, dim]
                            
                            # Viewpoint tokens: extract last timestep from queue
                            viewpoint_se3 = np.concatenate(list(current_extrinsics_queue), axis=0)  # [H, 4, 4]
                            viewpoint_se3 = viewpoint_se3[-1:, :, :]  # [1, 4, 4] - only last timestep
                            viewpoint_9d = convert_se3_to_9d(viewpoint_se3[None])[0]  # [1, 9]
                            viewpoint = normalize_data(
                                viewpoint_9d, 
                                stats=stats["view_actions"], 
                                type=cfgs.dataset.action_norm_type
                            )  # [1, 9]
                            viewpoint = torch.from_numpy(viewpoint).float().to(point_flow.device).unsqueeze(0)  # [B(1), 1, 9]
                            view_viewpoint_tokens = viewpoint
                             
                            
                            # All masks should be None when input is a single timestep
                            view_flow_mask = None
                            
                        else:
                            # Use all history for view_model
                            view_flow_tokens = build_flow_tokens(condition_flow, get_robot_policy_input_flow_dim(cfgs))
                            
                            # Create flow_mask based on visibility (last dimension of point_flow)
                            # If D_view >= 4, the last dimension is visibility (0 > visible, <= 0 invisible)
                            # flow_mask: (B_view, N_view) where True means visible (masking=true), False means invisible (masking=false)
                            if D_view == 4 or use_pixel_flow_for_robot_policy(cfgs):
                                assert B_view == 1, f"B_view: {B_view}"
                                if cfgs.training.use_masking: # flow: 3dim, but masking is applied
                                    view_flow_mask = visibility.bool()  # [B_view, N_view] - True if visible in the most recent timestep
                                    if not torch.any(view_flow_mask, dim=1).all():
                                        view_flow_mask[:, torch.randint(0, N_view, (1,), device=view_flow_tokens.device)] = True
                                        
                                    assert torch.any(view_flow_mask, dim=1).all(), f"at least one should be visible(true) along N axis, view_flow_mask : {view_flow_mask}"
                                else:
                                    view_flow_mask = None
                            else:
                                # If D_view < 4, no visibility dimension, assume all points are visible
                                view_flow_mask = None
                            
                            # Proprioception tokens: [B, H, dim] -> already in correct shape
                            view_proprio_tokens = proprioception  # [B, H, dim]

                            # Viewpoint tokens: prepare viewpoint_history from current_extrinsics_queue
                            # Use all history from queue
                            viewpoint_se3 = np.concatenate(list(current_extrinsics_queue), axis=0)  # [H, 4, 4]
                            
                            
                            # Convert to 9d representation
                            viewpoint_9d = convert_se3_to_9d(viewpoint_se3[None])[0]  # [H, 9] or [1, 9]
                            
                            # Normalize viewpoint history
                            viewpoint = normalize_data(
                                viewpoint_9d, 
                                stats=stats["view_actions"], 
                                type=cfgs.dataset.action_norm_type
                            )  # [H, 9] or [1, 9]
                            
                            # Convert to tensor and add batch dimension
                            viewpoint = torch.from_numpy(viewpoint).float().to(point_flow.device).unsqueeze(0)  # [B(1), H, 9]
                            view_viewpoint_tokens = viewpoint
                             
                        
                        # If use_flow_model=True, get flow prediction separately from flow_model
                        if use_flow_model and flow_model is not None:
                            # Prepare future_action_tokens_tensor if use_future_action_for_flow_model is True
                            future_action_tokens_tensor = None
                            if use_future_action_for_flow_model:
                                # Use model's predicted action as future_action condition
                                # action: [pred_horizon, action_dim] - unnormalized
                                # Normalize and convert to tensor
                                future_action_tokens_normalized = normalize_data(
                                    action[None],  # [1, pred_horizon, action_dim]
                                    stats=stats["action"],
                                    type=cfgs.dataset.action_norm_type
                                )  # [1, pred_horizon, action_dim] - normalized
                                future_action_tokens_tensor = torch.from_numpy(future_action_tokens_normalized[0]).float().to(point_flow.device).unsqueeze(0)  # [B(1), pred_horizon, action_dim]
                            
                            flow_pred_unnorm = get_flow_output(
                                flow_model, condition_flow, proprioception, stats, cfgs, 
                                pred_horizon, input_horizon, current_point_flow, visibility, eval_dataset, viewpoint,
                                future_action_tokens_tensor=future_action_tokens_tensor
                            )


                        # Add condition flag to tokens based on use_condition
                        # For view_model.predict_action, always use condition=True
                        use_condition_for_predict = True
                        # view_flow_tokens: [B, N, D] -> [B, N, D+1]
                        if view_flow_tokens is not None:
                            if use_condition_for_predict:
                                condition_flag = torch.ones(B_view, N_view, 1, dtype=view_flow_tokens.dtype, device=view_flow_tokens.device)
                            else:
                                # Unconditional: use zero tokens and append 0 flag
                                flow_token_dim = view_flow_tokens.shape[-1]
                                view_flow_tokens = torch.zeros(B_view, N_view, flow_token_dim, dtype=view_flow_tokens.dtype, device=view_flow_tokens.device)
                                condition_flag = torch.zeros(B_view, N_view, 1, dtype=view_flow_tokens.dtype, device=view_flow_tokens.device)
                            view_flow_tokens = torch.cat([view_flow_tokens, condition_flag], dim=-1)
                        
                        # view_proprio_tokens: [B, H, D] -> [B, H, D+1]
                        if view_proprio_tokens is not None:
                            H_proprio = view_proprio_tokens.shape[1]
                            if use_condition_for_predict:
                                condition_flag = torch.ones(B_view, H_proprio, 1, dtype=view_proprio_tokens.dtype, device=view_proprio_tokens.device)
                            else:
                                # Unconditional: use zero tokens and append 0 flag
                                proprio_token_dim = view_proprio_tokens.shape[-1]
                                view_proprio_tokens = torch.zeros(B_view, H_proprio, proprio_token_dim, dtype=view_proprio_tokens.dtype, device=view_proprio_tokens.device)
                                condition_flag = torch.zeros(B_view, H_proprio, 1, dtype=view_proprio_tokens.dtype, device=view_proprio_tokens.device)
                            view_proprio_tokens = torch.cat([view_proprio_tokens, condition_flag], dim=-1)
                        
                        # view_viewpoint_tokens: [B, H, 9] -> [B, H, 10]
                        if view_viewpoint_tokens is not None:
                            H_viewpoint = view_viewpoint_tokens.shape[1]
                            if use_condition_for_predict:
                                condition_flag = torch.ones(B_view, H_viewpoint, 1, dtype=view_viewpoint_tokens.dtype, device=view_viewpoint_tokens.device)
                            else:
                                # Unconditional: use zero tokens and append 0 flag
                                viewpoint_token_dim = view_viewpoint_tokens.shape[-1]
                                view_viewpoint_tokens = torch.zeros(B_view, H_viewpoint, viewpoint_token_dim, dtype=view_viewpoint_tokens.dtype, device=view_viewpoint_tokens.device)
                                condition_flag = torch.zeros(B_view, H_viewpoint, 1, dtype=view_viewpoint_tokens.dtype, device=view_viewpoint_tokens.device)
                            view_viewpoint_tokens = torch.cat([view_viewpoint_tokens, condition_flag], dim=-1)
                            
                            
                        # Apply fix_mask if enabled (after prediction)
                        
                        if cfgs.use_fix_mask:
                            view_fix_mask = torch.zeros(B_view, input_horizon, view_model.action_dim, device=point_flow.device)
                            view_prior = torch.zeros(B_view, input_horizon, view_model.action_dim, device=point_flow.device)
                            if not cfgs.use_relative:
                                if cfgs.training.use_ptp:
                                    view_fix_mask[:, :cfgs.dataset.obs_horizon, :] = 1.0 # [t-2, t-1, t] for h =3
                                    view_prior[:, :cfgs.dataset.obs_horizon, :] = torch.from_numpy(current_viewpoint_norm[None]).float().to(point_flow.device)  # [dim]
                                else:
                                    view_fix_mask[:, 0, :] = 1.0  # fix the first action
                                    # Use current_viewpoint_norm as prior
                                    view_prior[:, 0, :] = torch.from_numpy(current_viewpoint_norm[None]).float().to(point_flow.device)  # [dim]
                                
                        
                        start = time.time()
                        
                        with torch.no_grad():
                            # Use CustomRDTRunner's predict_action method
                            view_action_pred_output = view_model.predict_action(
                                flow_tokens=view_flow_tokens,
                                proprio_tokens=view_proprio_tokens,
                                viewpoint_tokens=view_viewpoint_tokens,
                                fix_mask=view_fix_mask,
                                prior=view_prior,
                                flow_mask=view_flow_mask,
                                proprio_mask=None,
                                viewpoint_mask=None,
                                
                            )
                            
                            # Handle output based on predict_flow flag (view_model doesn't use predict_separate_flow)
                            if view_model.predict_flow or view_model.predict_separate_flow:
                                view_action_pred, _ = view_action_pred_output  # Extract action, ignore flow prediction
                            else:
                                view_action_pred = view_action_pred_output
                            # view_action_pred: [B, horizon, action_dim]
                        
                        print(f'In evaluation, view action generation diffusion time taken: {time.time() - start}')
                        
                        # Convert to numpy
                        view_action = view_action_pred[0].detach().cpu().numpy()  # [horizon, dim]
                        if cfgs.training.use_ptp:
                            view_action = view_action[cfgs.dataset.obs_horizon-1:].copy()
                        
                        view_action_start = 0 # obs_horizon - 1
                        view_action_end = view_action_start + pred_horizon # action_horizon
                        view_action = view_action[view_action_start:view_action_end, :] # [h, 9]
                        
                        
                        if cfgs.use_relative:
                            view_action = unnormalize_data(view_action.copy(), stats=stats["delta_view_action_for_uncondition"], type=cfgs.dataset.action_norm_type) # [h, 6]
                            current_viewpoint_pos = current_viewpoint[:3].copy() # unnormalized, marker coordinate
                            current_viewpoint_ortho6d = current_viewpoint[3:9].copy()
                            current_viewpoint_rot = compute_rotation_matrix_from_ortho6d(current_viewpoint_ortho6d[None])[0] # marker coordinate
                            
                            relative_pos = view_action[..., :3].copy() # [h, 3]
                            relative_rotvec = view_action[..., 3:6].copy() # [h, 3] # axis angle
                            relative_rot = R.from_rotvec(relative_rotvec).as_matrix() # [h, 3, 3]
                            
                            view_action_pos = current_viewpoint_pos[None] + np.cumsum(relative_pos, axis=0) # [h, 3], {t+1}, {t+2}, {t+3}, ...
                            # cumulative product of rotation matrices (itertools.accumulate)
                            rot_objects = [R.from_matrix(relative_rot[idx]) for idx in range(len(relative_rot))]
                            cumulative_rots = list(accumulate(rot_objects, lambda acc, r: acc * r, initial=R.identity()))[1:]
                            cumulative_rot_matrices = np.array([r.as_matrix() for r in cumulative_rots])
                            view_action_rot = np.einsum('ij,tjk->tik', current_viewpoint_rot, cumulative_rot_matrices) # [h, 3, 3]
                            view_action_ortho6d = compute_ortho6d_from_rotation_matrix(view_action_rot) # [h, 6]
                            view_action = np.concatenate([view_action_pos, view_action_ortho6d], axis=-1) # [h, 9]
                        else:
                            view_action = unnormalize_data(view_action.copy(), stats=stats["view_actions"], type=cfgs.dataset.action_norm_type) # [h, 9]

                        
                    else:
                        view_action = current_viewpoint[None] # [h(1), 9]
                    
                    # Store view_action and viewpoint_gt for plotting
                    view_actions_list.append(view_action.copy())  # list of [h, 9]
                    viewpoint_gt_list.append(viewpoint_gt.copy())  # list of [h, 9]
                    # retargeted_active_camera_pose_list will be stored later when it's computed
                        
                    view_action_se3 = create_pose_from_9d(view_action.copy()) # [h, 4, 4], marker coordinate
                    # convert to world coordinate
                    # Broadcasting: [4, 4] @ [H, 4, 4] -> [H, 4, 4]
                    view_action_se3_w = np.einsum('ij,hjk->hik', T_W_M, view_action_se3)
                    
                    current_camera_pose_list = [v for v in view_action_se3_w]


                    # Prepare query points based on predict_separate_flow or predict_flow flag (from model, not view_model)
                    if flow_pred_unnorm is not None:
                        # Use predicted flow as query points: [pred_horizon, N, D] (marker coordinate, unnormalized)
                        query_tracking_point_in_marker_coordinate = flow_pred_unnorm  # [pred_horizon, N, D]
                        
                        # Convert to world coordinate for visualization
                        # query_tracking_point_in_marker_coordinate: [pred_horizon, N, D]
                        query_points_xyz_marker = query_tracking_point_in_marker_coordinate[..., :3]  # [pred_horizon, N, 3]
                        query_points_homo_marker = np.concatenate([query_points_xyz_marker, np.ones((*query_points_xyz_marker.shape[:2], 1))], axis=-1)  # [pred_horizon, N, 4]
                        # Broadcasting: [4, 4] @ [pred_horizon, N, 4, 1] -> [pred_horizon, N, 4]
                        query_points_homo_world = np.einsum('ij,hnj->hni', T_W_M, query_points_homo_marker)  # [pred_horizon, N, 4]
                        query_points_xyz_world = query_points_homo_world[..., :3]  # [pred_horizon, N, 3]
                        
                        if eval_dataset.flow_dim == 4:
                            # Include visibility if available
                            query_points_visibility = query_tracking_point_in_marker_coordinate[..., -1:]  # [pred_horizon, N, 1]
                            query_tracking_point_in_world_coordinate = np.concatenate([query_points_xyz_world, query_points_visibility], axis=-1)  # [pred_horizon, N, 4]
                        else:
                            query_tracking_point_in_world_coordinate = query_points_xyz_world  # [pred_horizon, N, 3]
                    else:
                        # Use current tracked points (original behavior)
                        # Convert tracked_3d_unnormalized [N, 3] to homogeneous coordinates [N, 4]
                        tracked_3d_homo = np.hstack([tracked_3d_unnormalized[..., :3], np.ones((tracked_3d_unnormalized.shape[0], 1))])
                        if use_marker_coordinate:
                            # Include visibility if available (tracked_3d_unnormalized can be [N, 3] or [N, 4])
                            if eval_dataset.flow_dim == 4:
                                # [N, 4] - includes visibility from cotracker
                                query_tracking_point_in_marker_coordinate = tracked_3d_unnormalized.copy()
                            else:
                                raise NotImplementedError
                            query_tracking_point_in_world_coordinate = (T_W_M @ tracked_3d_homo.T).T[:, :3] # [N, 3]
                            if eval_dataset.flow_dim == 4:
                                query_tracking_point_in_world_coordinate = np.concatenate([query_tracking_point_in_world_coordinate, tracked_3d_unnormalized[..., -1][..., None]], axis=-1) # [N, 4]
                        else:
                            raise NotImplementedError
                        
                    
                    T_M_C = T_mc_transformation[step].copy()
                    
                    # view policy (SVDD)
                    # Convert view_action to SE(3) for nominal trajectory (marker coordinate)
                    view_action_se3_nominal = create_pose_from_9d(view_action.copy())  # [h, 4, 4], marker coordinate
                    
                    task_kwargs = {'robot_future_action': action.copy(), # [horizon, act_dim] (from time t, unnormalized), marker coordinate
                                    'T_mc_transformation' : T_M_C, # [horizon, 4, 4] (from time t), used for prior, robot mesh 
                                    'T_mw' : T_mw, 
                                    'T_B_M' : T_B_M,
                                    'T_B_M_view' : T_B_M_view,
                                    'mesh_original': scene_mesh,
                                    'robot_viz': episode_robot_viz,
                                    'view_robot_viz': episode_view_robot_viz,
                                    'query_point': query_tracking_point_in_marker_coordinate,
                                    'query_point_indices': query_point_indices,  # List of query point indices to use (None means use all)
                                    'use_marker_coordinate': use_marker_coordinate,
                                    'selected_horizons': np.round(np.linspace(3, pred_horizon-1, num=3)).astype(int),
                                    'camera_intrinsics': intrinsics,
                                    'image_size': eval_dataset.pixel_tracking_img_size,
                                    'predict_separate_flow': model.predict_separate_flow or model.predict_flow or use_flow_model,  # Flag to indicate if using predicted flow as query points
                                    'reward_type': reward_type,  # Reward type: 'visibility', 'visibility+close_to_nominal', or 'visibility+close_to_query_points'
                                    'visibility_reward_type': visibility_reward_type,  # Visibility reward type: 'w_consider_tracking_lost', 'wo_consider_tracking_lost', or None
                                    'nominal_trajectory': view_action_se3_nominal,  # [h, 4, 4] SE(3) transformation matrices in marker coordinate
                                    'use_visibility_aware_retargeting': use_visibility_aware_retargeting,
                                    }
                    assert (eval_dataset.pt_max == stats["point_tracking_data"]["max"]).all()
                    assert (eval_dataset.pt_min == stats["point_tracking_data"]["min"]).all()
                    assert (eval_dataset.pt_mean == stats["point_tracking_data"]["mean"]).all()
                    assert (eval_dataset.pt_std == stats["point_tracking_data"]["std"]).all()
                    
                    
                    start = time.time()
                    view_condition_flow = condition_flow
                    if cfgs.svdd.batch_size > 1:
                        assert view_condition_flow.shape[0] == 1, f"view_condition_flow.shape[0]: {view_condition_flow.shape[0]}"
                        assert proprioception.shape[0] == 1, f"proprioception.shape[0]: {proprioception.shape[0]}"
                        assert viewpoint.shape[0] == 1, f"viewpoint.shape[0]: {viewpoint.shape[0]}"
                        view_condition_flow = torch.tile(view_condition_flow, (cfgs.svdd.batch_size, 1, 1, 1)) # [B(1), h, N, D] -> [B, h, N, D]
                        proprioception = torch.tile(proprioception, (cfgs.svdd.batch_size, 1, 1)) # [B(1), h, dim] -> [B, h, dim]
                        viewpoint = torch.tile(viewpoint, (cfgs.svdd.batch_size, 1, 1)) # [B(1), h, dim] -> [B, h, dim]
                        visibility = torch.tile(visibility, (cfgs.svdd.batch_size, 1)) # [B(1), N] -> [B, N]
                        
                    
                    # For run_svdd, always use condition=False
                    viewpoint_pred_svdd, visual_info = run_svdd(cfgs, view_model.noise_scheduler_sample, view_num_inference_steps, view_model, view_condition_flow, proprioception, 
                        dataset=eval_dataset, stats=stats, task='active_vision', task_kwargs=task_kwargs, return_task_kwargs=True, visibility=visibility, viewpoint=viewpoint, use_condition=False, use_retargeting=use_retargeting)
                    
                    # choose best reward sample based on 'total_reward' from final_reward_dict
                    # final_reward_dict contains rewards for final x0_svdd with shape [B]
                    best_idx = None
                    final_reward_dict = visual_info['final_reward_dict']
                    total_reward = final_reward_dict['total_reward'].detach().cpu().numpy()
                    best_idx = int(np.argmax(total_reward))
                
                    # choose best reward sample for retargeted trajectory based on 'total_reward' from final_reward_dict_retargeted
                    best_idx_retargeted = None
                    if visual_info is not None and 'final_reward_dict_retargeted' in visual_info and visual_info['final_reward_dict_retargeted'] is not None:
                        final_reward_dict_retargeted = visual_info['final_reward_dict_retargeted']
                        total_reward_retargeted = final_reward_dict_retargeted['total_reward'].detach().cpu().numpy()
                        best_idx_retargeted = int(np.argmax(total_reward_retargeted))
                    
                    
                    # Store best_idx for plotting
                    episode_svdd_best_indices.append(best_idx)
                    
                    # Store original batch size before selecting best_idx
                    B_original = viewpoint_pred_svdd.shape[0]
                    
                    # Unnormalize all batch indices for visibility computation
                    viewpoint_pred_svdd_all = []  # Will store unnormalized viewpoints for all batch indices
                    for b_idx in range(B_original):
                        viewpoint_pred_svdd_b = viewpoint_pred_svdd[b_idx]  # [horizon, dim]
                        
                        if cfgs.use_relative:
                            viewpoint_pred_svdd_b = unnormalize_data(viewpoint_pred_svdd_b.copy(), stats=stats["delta_view_action_for_uncondition"], type=cfgs.dataset.action_norm_type)  # [horizon, 6]
                            # convert to absolute
                            relative_pos = viewpoint_pred_svdd_b[..., :3].copy() # [horizon, 3]
                            relative_rotvec = viewpoint_pred_svdd_b[..., 3:6].copy() # [horizon, 3] # axis angle
                            relative_rot = R.from_rotvec(relative_rotvec).as_matrix() # [horizon, 3, 3]
                            
                            current_viewpoint_pos = current_viewpoint[:3].copy() # [3]
                            current_viewpoint_ortho6d = current_viewpoint[3:9].copy() # [6]
                            current_viewpoint_rot = compute_rotation_matrix_from_ortho6d(current_viewpoint_ortho6d[None])[0] # [3, 3]
                            
                            absolute_pos = current_viewpoint_pos[None] + np.cumsum(relative_pos, axis=0) # [horizon, 3], {t+1}, {t+2}, {t+3}, ...
                            # cumulative product of rotation matrices (itertools.accumulate)
                            rot_objects = [R.from_matrix(relative_rot[idx]) for idx in range(len(relative_rot))]
                            cumulative_rots = list(accumulate(rot_objects, lambda acc, r: acc * r, initial=R.identity()))[1:]
                            cumulative_rot_matrices = np.array([r.as_matrix() for r in cumulative_rots])
                            absolute_rot = np.einsum('ij,tjk->tik', current_viewpoint_rot, cumulative_rot_matrices) # [horizon, 3, 3]
                            absolute_ortho6d = compute_ortho6d_from_rotation_matrix(absolute_rot) # [horizon, 6]
                            viewpoint_pred_svdd_b = np.concatenate([absolute_pos, absolute_ortho6d], axis=-1) # [horizon, 9]
                        else:
                            viewpoint_pred_svdd_b = unnormalize_data(viewpoint_pred_svdd_b.copy(), stats=stats["view_actions"], type=cfgs.dataset.action_norm_type)  # [horizon, 9]
                        
                        viewpoint_pred_svdd_all.append(viewpoint_pred_svdd_b)
                    

                    viewpoint_pred_svdd_all_temp = visual_info['x0_svdd'].copy() # [B, H, 9]
                    for b in range(B_original):
                        assert np.linalg.norm(viewpoint_pred_svdd_all[b] - viewpoint_pred_svdd_all_temp[b], axis=-1).max() < 1e-6
                        

                    # Select best_idx for actual use
                    viewpoint_pred_svdd = viewpoint_pred_svdd_all[best_idx]  # [horizon, dim]

                    # Store viewpoint_pred_svdd for plotting
                    view_actions_svdd_list.append(viewpoint_pred_svdd.copy())  # list of [h, 9]
                    # Note: view_actions_svdd_retargeted_list will be populated later using retargeted_active_camera_pose_list
                    
                    # Collect detailed reward components from final_reward_dict if available
                    # visual_info is the task_kwargs returned from run_svdd
                    if visual_info is not None and 'final_reward_dict' in visual_info and visual_info['final_reward_dict'] is not None:
                        # Get final_reward_dict (for final x0_svdd with shape [B])
                        reward_dict = visual_info['final_reward_dict']
                        
                        # Extract reward components (convert tensors to numpy/scalar)
                        # Store all values instead of averaging when size != 1
                        if reward_dict.get('total_reward') is not None:
                            total_reward = reward_dict['total_reward'].detach().cpu().numpy()
                            if isinstance(total_reward, np.ndarray) and total_reward.size == 1:
                                total_reward = float(total_reward.item())
                            elif isinstance(total_reward, np.ndarray):
                                # Store all values as a list instead of taking mean
                                total_reward = total_reward.flatten().tolist()
                            episode_svdd_total_rewards.append(total_reward)
                        
                        if reward_dict.get('visibility_reward') is not None:
                            vis_reward = reward_dict['visibility_reward'].detach().cpu().numpy()
                            if isinstance(vis_reward, np.ndarray) and vis_reward.size == 1:
                                vis_reward = float(vis_reward.item())
                            elif isinstance(vis_reward, np.ndarray):
                                # Store all values as a list instead of taking mean
                                vis_reward = vis_reward.flatten().tolist()
                            episode_svdd_visibility_rewards.append(vis_reward)
                        
                        if reward_dict.get('close_to_nominal_reward') is not None:
                            close_reward = reward_dict['close_to_nominal_reward'].detach().cpu().numpy()
                            if isinstance(close_reward, np.ndarray) and close_reward.size == 1:
                                close_reward = float(close_reward.item())
                            elif isinstance(close_reward, np.ndarray):
                                # Store all values as a list instead of taking mean
                                close_reward = close_reward.flatten().tolist()
                            episode_svdd_close_to_nominal_rewards.append(close_reward)
                        else:
                            episode_svdd_close_to_nominal_rewards.append(None)
                        
                        if reward_dict.get('close_to_query_points_reward') is not None:
                            close_query_reward = reward_dict['close_to_query_points_reward'].detach().cpu().numpy()
                            if isinstance(close_query_reward, np.ndarray) and close_query_reward.size == 1:
                                close_query_reward = float(close_query_reward.item())
                            elif isinstance(close_query_reward, np.ndarray):
                                # Store all values as a list instead of taking mean
                                close_query_reward = close_query_reward.flatten().tolist()
                            episode_svdd_close_to_query_points_rewards.append(close_query_reward)
                        else:
                            episode_svdd_close_to_query_points_rewards.append(None)
                        
                        if reward_dict.get('smoothness_reward') is not None:
                            smoothness_reward = reward_dict['smoothness_reward']
                            # smoothness is a numpy array (not tensor), so handle accordingly
                            if isinstance(smoothness_reward, np.ndarray):
                                if smoothness_reward.size == 1:
                                    smoothness_reward = float(smoothness_reward.item())
                                else:
                                    # Store all values as a list instead of taking mean
                                    smoothness_reward = smoothness_reward.flatten().tolist()
                            elif hasattr(smoothness_reward, 'detach'):
                                # If it's a tensor, convert to numpy first
                                smoothness_reward = smoothness_reward.detach().cpu().numpy()
                                if smoothness_reward.size == 1:
                                    smoothness_reward = float(smoothness_reward.item())
                                else:
                                    smoothness_reward = smoothness_reward.flatten().tolist()
                            else:
                                smoothness_reward = float(smoothness_reward)
                            episode_svdd_smoothness_rewards.append(smoothness_reward)
                        else:
                            episode_svdd_smoothness_rewards.append(None)
                    else:
                        # If no final_reward_dict available, append None
                        episode_svdd_total_rewards.append(None)
                        episode_svdd_visibility_rewards.append(None)
                        episode_svdd_close_to_nominal_rewards.append(None)
                        episode_svdd_close_to_query_points_rewards.append(None)
                        episode_svdd_smoothness_rewards.append(None)
                    
                    # Collect detailed reward components from final_reward_dict_retargeted if available
                    if visual_info is not None and 'final_reward_dict_retargeted' in visual_info and visual_info['final_reward_dict_retargeted'] is not None:
                        # Get final_reward_dict_retargeted (for retargeted x0_svdd with shape [B])
                        reward_dict_retargeted = visual_info['final_reward_dict_retargeted']
                        
                        # Extract reward components (convert tensors to numpy/scalar)
                        # Store all values instead of averaging when size != 1
                        if reward_dict_retargeted.get('total_reward') is not None:
                            total_reward_retargeted = reward_dict_retargeted['total_reward'].detach().cpu().numpy()
                            if isinstance(total_reward_retargeted, np.ndarray) and total_reward_retargeted.size == 1:
                                total_reward_retargeted = float(total_reward_retargeted.item())
                            elif isinstance(total_reward_retargeted, np.ndarray):
                                # Store all values as a list instead of taking mean
                                total_reward_retargeted = total_reward_retargeted.flatten().tolist()
                            episode_svdd_total_rewards_retargeted.append(total_reward_retargeted)
                        else:
                            episode_svdd_total_rewards_retargeted.append(None)
                        
                        if reward_dict_retargeted.get('visibility_reward') is not None:
                            vis_reward_retargeted = reward_dict_retargeted['visibility_reward'].detach().cpu().numpy()
                            if isinstance(vis_reward_retargeted, np.ndarray) and vis_reward_retargeted.size == 1:
                                vis_reward_retargeted = float(vis_reward_retargeted.item())
                            elif isinstance(vis_reward_retargeted, np.ndarray):
                                # Store all values as a list instead of taking mean
                                vis_reward_retargeted = vis_reward_retargeted.flatten().tolist()
                            episode_svdd_visibility_rewards_retargeted.append(vis_reward_retargeted)
                        else:
                            episode_svdd_visibility_rewards_retargeted.append(None)
                        
                        if reward_dict_retargeted.get('close_to_nominal_reward') is not None:
                            close_reward_retargeted = reward_dict_retargeted['close_to_nominal_reward'].detach().cpu().numpy()
                            if isinstance(close_reward_retargeted, np.ndarray) and close_reward_retargeted.size == 1:
                                close_reward_retargeted = float(close_reward_retargeted.item())
                            elif isinstance(close_reward_retargeted, np.ndarray):
                                # Store all values as a list instead of taking mean
                                close_reward_retargeted = close_reward_retargeted.flatten().tolist()
                            episode_svdd_close_to_nominal_rewards_retargeted.append(close_reward_retargeted)
                        else:
                            episode_svdd_close_to_nominal_rewards_retargeted.append(None)
                        
                        if reward_dict_retargeted.get('close_to_query_points_reward') is not None:
                            close_query_reward_retargeted = reward_dict_retargeted['close_to_query_points_reward'].detach().cpu().numpy()
                            if isinstance(close_query_reward_retargeted, np.ndarray) and close_query_reward_retargeted.size == 1:
                                close_query_reward_retargeted = float(close_query_reward_retargeted.item())
                            elif isinstance(close_query_reward_retargeted, np.ndarray):
                                # Store all values as a list instead of taking mean
                                close_query_reward_retargeted = close_query_reward_retargeted.flatten().tolist()
                            episode_svdd_close_to_query_points_rewards_retargeted.append(close_query_reward_retargeted)
                        else:
                            episode_svdd_close_to_query_points_rewards_retargeted.append(None)
                        
                        if reward_dict_retargeted.get('smoothness_reward') is not None:
                            smoothness_reward_retargeted = reward_dict_retargeted['smoothness_reward']
                            # smoothness is a numpy array (not tensor), so handle accordingly
                            if isinstance(smoothness_reward_retargeted, np.ndarray):
                                if smoothness_reward_retargeted.size == 1:
                                    smoothness_reward_retargeted = float(smoothness_reward_retargeted.item())
                                else:
                                    # Store all values as a list instead of taking mean
                                    smoothness_reward_retargeted = smoothness_reward_retargeted.flatten().tolist()
                            elif hasattr(smoothness_reward_retargeted, 'detach'):
                                # If it's a tensor, convert to numpy first
                                smoothness_reward_retargeted = smoothness_reward_retargeted.detach().cpu().numpy()
                                if smoothness_reward_retargeted.size == 1:
                                    smoothness_reward_retargeted = float(smoothness_reward_retargeted.item())
                                else:
                                    smoothness_reward_retargeted = smoothness_reward_retargeted.flatten().tolist()
                            else:
                                smoothness_reward_retargeted = float(smoothness_reward_retargeted)
                            episode_svdd_smoothness_rewards_retargeted.append(smoothness_reward_retargeted)
                        else:
                            episode_svdd_smoothness_rewards_retargeted.append(None)
                    else:
                        # If no final_reward_dict_retargeted available, append None
                        episode_svdd_total_rewards_retargeted.append(None)
                        episode_svdd_visibility_rewards_retargeted.append(None)
                        episode_svdd_close_to_nominal_rewards_retargeted.append(None)
                        episode_svdd_close_to_query_points_rewards_retargeted.append(None)
                        episode_svdd_smoothness_rewards_retargeted.append(None)

                    if step % 1 == 0 :
                        print(f'In evaluation, svdd time taken: {time.time() - start}')
                    
                    if i in debug_episode: # first episode
                        # Compute visibility for all batch indices (needed for plotting)
                        # Create task_kwargs for visibility computation
                        visibility_task_kwargs = task_kwargs.copy()
                        if 'precomputed_scenes' in visual_info:
                            visibility_task_kwargs.update({
                                'precomputed_scenes': visual_info['precomputed_scenes'],
                            })
                        # Use final_reward_dict from run_svdd - all rewards and visibility_results are already computed
                        reward_dict = visual_info['final_reward_dict']
                        
                        # Extract visibility rewards for all batch indices
                        if reward_dict.get('visibility_reward') is not None:
                            visibility_rewards_tensor = reward_dict['visibility_reward'].detach().cpu().numpy()  # [B]
                            active_visibility_rewards_all = visibility_rewards_tensor.flatten().tolist()  # [B] -> list
                        else:
                            active_visibility_rewards_all = [None] * B_original
                        
                        if reward_dict.get('visibility_reward_among_tracking_points') is not None:
                            visibility_rewards_among_tracking_tensor = reward_dict['visibility_reward_among_tracking_points'].detach().cpu().numpy()  # [B]
                            active_visibility_rewards_among_tracking_all = visibility_rewards_among_tracking_tensor.flatten().tolist()  # [B] -> list
                        else:
                            active_visibility_rewards_among_tracking_all = [None] * B_original
                        
                        if reward_dict.get('visibility_reward_wo_tracking_lost') is not None:
                            visibility_rewards_wo_tracking_lost_tensor = reward_dict['visibility_reward_wo_tracking_lost'].detach().cpu().numpy()  # [B]
                            active_visibility_rewards_wo_tracking_lost_all = visibility_rewards_wo_tracking_lost_tensor.flatten().tolist()  # [B] -> list
                        else:
                            active_visibility_rewards_wo_tracking_lost_all = [None] * B_original
                        
                        # Extract visibility_results for all batch indices
                        if reward_dict.get('visibility_results') is not None:
                            visibility_results_all_batch = reward_dict['visibility_results']  # [B, h, N]
                            active_visibility_results_all = [visibility_results_all_batch[b_idx] for b_idx in range(B_original)]
                        else:
                            active_visibility_results_all = [None] * B_original
                        
                        if reward_dict.get('visibility_results_among_tracking') is not None:
                            visibility_results_among_tracking_all_batch = reward_dict['visibility_results_among_tracking']  # [B, h, N]
                            active_visibility_results_among_tracking_all = [visibility_results_among_tracking_all_batch[b_idx] for b_idx in range(B_original)]
                        else:
                            active_visibility_results_among_tracking_all = [None] * B_original
                        
                        if reward_dict.get('visibility_results_wo_tracking_lost') is not None:
                            visibility_results_wo_tracking_lost_all_batch = reward_dict['visibility_results_wo_tracking_lost']  # [B, h, N]
                            active_visibility_results_wo_tracking_lost_all = [visibility_results_wo_tracking_lost_all_batch[b_idx] for b_idx in range(B_original)]
                        else:
                            active_visibility_results_wo_tracking_lost_all = [None] * B_original
                        
                        # Extract visibility_results for retargeted from final_reward_dict_retargeted (same as above)
                        if visual_info is not None and 'final_reward_dict_retargeted' in visual_info and visual_info['final_reward_dict_retargeted'] is not None:
                            reward_dict_retargeted = visual_info['final_reward_dict_retargeted']
                            
                            # Extract visibility rewards for retargeted (same as above)
                            if reward_dict_retargeted.get('visibility_reward') is not None:
                                visibility_rewards_retargeted_tensor = reward_dict_retargeted['visibility_reward'].detach().cpu().numpy()  # [B]
                                active_visibility_rewards_retargeted_all = visibility_rewards_retargeted_tensor.flatten().tolist()  # [B] -> list
                            else:
                                active_visibility_rewards_retargeted_all = [None] * B_original
                            
                            if reward_dict_retargeted.get('visibility_reward_among_tracking_points') is not None:
                                visibility_rewards_among_tracking_retargeted_tensor = reward_dict_retargeted['visibility_reward_among_tracking_points'].detach().cpu().numpy()  # [B]
                                active_visibility_rewards_among_tracking_retargeted_all = visibility_rewards_among_tracking_retargeted_tensor.flatten().tolist()  # [B] -> list
                            else:
                                active_visibility_rewards_among_tracking_retargeted_all = [None] * B_original
                            
                            if reward_dict_retargeted.get('visibility_reward_wo_tracking_lost') is not None:
                                visibility_rewards_wo_tracking_lost_retargeted_tensor = reward_dict_retargeted['visibility_reward_wo_tracking_lost'].detach().cpu().numpy()  # [B]
                                active_visibility_rewards_wo_tracking_lost_retargeted_all = visibility_rewards_wo_tracking_lost_retargeted_tensor.flatten().tolist()  # [B] -> list
                            else:
                                active_visibility_rewards_wo_tracking_lost_retargeted_all = [None] * B_original
                            
                            if reward_dict_retargeted.get('visibility_results') is not None:
                                visibility_results_retargeted_all_batch = reward_dict_retargeted['visibility_results']  # [B, h, N]
                                active_visibility_results_retargeted_all = [visibility_results_retargeted_all_batch[b_idx] for b_idx in range(B_original)]
                            else:
                                active_visibility_results_retargeted_all = [None] * B_original
                            
                            if reward_dict_retargeted.get('visibility_results_among_tracking') is not None:
                                visibility_results_among_tracking_retargeted_all_batch = reward_dict_retargeted['visibility_results_among_tracking']  # [B, h, N]
                                active_visibility_results_among_tracking_retargeted_all = [visibility_results_among_tracking_retargeted_all_batch[b_idx] for b_idx in range(B_original)]
                            else:
                                active_visibility_results_among_tracking_retargeted_all = [None] * B_original
                            
                            if reward_dict_retargeted.get('visibility_results_wo_tracking_lost') is not None:
                                visibility_results_wo_tracking_lost_retargeted_all_batch = reward_dict_retargeted['visibility_results_wo_tracking_lost']  # [B, h, N]
                                active_visibility_results_wo_tracking_lost_retargeted_all = [visibility_results_wo_tracking_lost_retargeted_all_batch[b_idx] for b_idx in range(B_original)]
                            else:
                                active_visibility_results_wo_tracking_lost_retargeted_all = [None] * B_original
                        else:
                            active_visibility_rewards_retargeted_all = [None] * B_original
                            active_visibility_rewards_among_tracking_retargeted_all = [None] * B_original
                            active_visibility_rewards_wo_tracking_lost_retargeted_all = [None] * B_original
                            active_visibility_results_retargeted_all = [None] * B_original
                            active_visibility_results_among_tracking_retargeted_all = [None] * B_original
                            active_visibility_results_wo_tracking_lost_retargeted_all = [None] * B_original
                    
                        
                        # Collect visibility rewards for plotting
                        episode_active_visibility_rewards.append(active_visibility_rewards_all)
                        episode_active_visibility_rewards_among_tracking.append(active_visibility_rewards_among_tracking_all)
                        episode_active_visibility_rewards_wo_tracking_lost.append(active_visibility_rewards_wo_tracking_lost_all)
                        
                        # Collect retargeted visibility rewards for plotting
                        episode_active_visibility_rewards_retargeted.append(active_visibility_rewards_retargeted_all)
                        episode_active_visibility_rewards_among_tracking_retargeted.append(active_visibility_rewards_among_tracking_retargeted_all)
                        episode_active_visibility_rewards_wo_tracking_lost_retargeted.append(active_visibility_rewards_wo_tracking_lost_retargeted_all)
                        
                        # Compute visibility for raw view_action (for comparison plot)
                        visibility_rewards, visibility_results, _, visibility_rewards_among_tracking, visibility_results_among_tracking, visibility_rewards_wo_tracking_lost, visibility_results_wo_tracking_lost, visibility_info = compute_visibility(view_action[None], visibility_task_kwargs)
                        visibility_rewards = visibility_rewards[0]  # scalar
                        visibility_results = visibility_results[0]  # [h, N] - only selected_horizons have meaningful values
                        visibility_rewards_among_tracking = visibility_rewards_among_tracking[0]  # scalar
                        visibility_rewards_wo_tracking_lost = visibility_rewards_wo_tracking_lost[0]  # scalar
                        visibility_results_among_tracking = visibility_results_among_tracking[0]  # [h, N] - only selected_horizons have meaningful values
                        visibility_results_wo_tracking_lost = visibility_results_wo_tracking_lost[0]  # [h, N] - only selected_horizons have meaningful values
                        vis_reward_val = float(np.mean(visibility_rewards)) if isinstance(visibility_rewards, np.ndarray) else float(visibility_rewards)
                        vis_reward_among_tracking_val = float(np.mean(visibility_rewards_among_tracking)) if isinstance(visibility_rewards_among_tracking, np.ndarray) else float(visibility_rewards_among_tracking)
                        vis_reward_wo_tracking_lost_val = float(np.mean(visibility_rewards_wo_tracking_lost)) if isinstance(visibility_rewards_wo_tracking_lost, np.ndarray) else float(visibility_rewards_wo_tracking_lost)
                        episode_visibility_rewards.append(vis_reward_val)
                        episode_visibility_rewards_among_tracking.append(vis_reward_among_tracking_val)
                        episode_visibility_rewards_wo_tracking_lost.append(vis_reward_wo_tracking_lost_val)

                        
                        # create a point cloud from the RGB-D frame (camera frame)
                        point_cloud_camera = rgbd_to_point_cloud(pixel_tracking_frames[step], pixel_tracking_depths[step], intrinsics)
                        
                        # transform the point cloud to the world frame
                        if len(point_cloud_camera.points) > 0:
                            # transform points from the camera frame to the world frame
                            points_camera = np.asarray(point_cloud_camera.points)
                            colors = np.asarray(point_cloud_camera.colors)
                            
                            # use the camera pose to transform to the world frame
                            points_camera_homo = np.hstack([points_camera, np.ones((len(points_camera), 1))])  # [N, 4]
                            points_world_homo = (T_W_C @ points_camera_homo.T).T  # [N, 4]
                            points_world = points_world_homo[:, :3]  # [N, 3]
                            
                            # create the point cloud in the world frame
                            point_cloud = o3d.geometry.PointCloud()
                            point_cloud.points = o3d.utility.Vector3dVector(points_world)
                            point_cloud.colors = o3d.utility.Vector3dVector(colors)
                            
                        else:
                            point_cloud = point_cloud_camera

                        
                        viewpoint_pred_svdd_se3 = create_pose_from_9d(viewpoint_pred_svdd) # [h, 9] -> [h, 4, 4], marker coordinate
                        
                        # convert to world coordinate
                        # Broadcasting: [4, 4] @ [H, 4, 4] -> [H, 4, 4]
                        viewpoint_pred_svdd_se3_w = np.einsum('ij,hjk->hik', T_W_M, viewpoint_pred_svdd_se3)
                        
                        active_camera_pose_list = [v for v in viewpoint_pred_svdd_se3_w] # len h list of [4, 4]
                        assert len(active_camera_pose_list) == pred_horizon, "nvblox visualization assume full prediction horizon, not chunk"
                        
                        # Extract retargeted camera poses from x0_svdd_retargeted using best_idx_retargeted
                        if visual_info is not None and 'x0_svdd_retargeted' in visual_info and visual_info['x0_svdd_retargeted'] is not None:
                            x0_svdd_retargeted = visual_info['x0_svdd_retargeted']  # [B, H, 9] marker coordinate, unnormalized
                            # Select best_idx_retargeted batch (computed based on retargeted rewards)
                            x0_svdd_retargeted_best = x0_svdd_retargeted[best_idx_retargeted]  # [H, 9] marker coordinate
                            # Convert 9D to SE(3)
                            viewpoint_pred_svdd_retargeted_se3 = create_pose_from_9d(x0_svdd_retargeted_best)  # [H, 9] -> [H, 4, 4], marker coordinate
                            # Convert to world coordinate
                            viewpoint_pred_svdd_retargeted_se3_w = np.einsum('ij,hjk->hik', T_W_M, viewpoint_pred_svdd_retargeted_se3)  # [H, 4, 4] world coordinate
                            retargeted_active_camera_pose_list = [v for v in viewpoint_pred_svdd_retargeted_se3_w]  # len h list of [4, 4]
                            
                            assert len(retargeted_active_camera_pose_list) == pred_horizon, "nvblox visualization assume full prediction horizon, not chunk"

                            
                            # Convert retargeted_active_camera_pose_list (world coordinate SE(3)) to marker coordinate 9D for plotting
                            T_M_W = np.linalg.inv(T_W_M)
                            viewpoint_pred_svdd_retargeted_se3_marker = np.einsum('ij,hjk->hik', T_M_W, viewpoint_pred_svdd_retargeted_se3_w)  # [H, 4, 4] marker coordinate
                            viewpoint_pred_svdd_retargeted_9d = convert_se3_to_9d(viewpoint_pred_svdd_retargeted_se3_marker[None])[0]  # [H, 9] marker coordinate
                            view_actions_svdd_retargeted_list.append(viewpoint_pred_svdd_retargeted_9d.copy())  # list of [h, 9]
                        else:
                            retargeted_active_camera_pose_list = []  # len h list of [4, 4]
                            view_actions_svdd_retargeted_list.append(None)

                        # Reuse visibility results from best_idx (already computed above)
                        if visibility_reward_type == 'wo_consider_tracking_lost':
                            active_visibility_results_for_dict = active_visibility_results_wo_tracking_lost_all[best_idx]  # [h, N] - only selected_horizons have meaningful values
                            visibility_results_for_dict = visibility_results_wo_tracking_lost
                        
                        elif visibility_reward_type == 'w_consider_tracking_lost':
                            active_visibility_results_for_dict = active_visibility_results_all[best_idx]  # [h, N] - only selected_horizons have meaningful values
                            visibility_results_for_dict = visibility_results
                        
                        # Reuse visibility results for retargeted from best_idx_retargeted (same as above)
                        retargeted_active_visibility_results_for_dict = None
                        if len(retargeted_active_camera_pose_list) > 0:
                            # Get best_idx_retargeted
                            if visual_info is not None and 'final_reward_dict_retargeted' in visual_info and visual_info['final_reward_dict_retargeted'] is not None:
                                reward_dict_retargeted = visual_info['final_reward_dict_retargeted']
                                total_reward_retargeted = reward_dict_retargeted['total_reward'].detach().cpu().numpy()
                                best_idx_retargeted = int(np.argmax(total_reward_retargeted))
                            else:
                                best_idx_retargeted = best_idx  # Fallback to original best_idx
                            
                            if visibility_reward_type == 'wo_consider_tracking_lost':
                                retargeted_active_visibility_results_for_dict = active_visibility_results_wo_tracking_lost_retargeted_all[best_idx_retargeted]  # [h, N] - only selected_horizons have meaningful values
                            elif visibility_reward_type == 'w_consider_tracking_lost':
                                retargeted_active_visibility_results_for_dict = active_visibility_results_retargeted_all[best_idx_retargeted]  # [h, N] - only selected_horizons have meaningful values
                        
                        selected_horizons = task_kwargs['selected_horizons']
                        
                        # Reuse visibility_results for view_action (already computed above)
                        # visibility_results is already computed above at line 1965
                        
                        # Only use selected_horizons for visibility_results_dict (only these have meaningful values)
                        current_visibility_results_dict = {h: visibility_results_for_dict[h] for h in selected_horizons}
                        active_visibility_results_dict = {h: active_visibility_results_for_dict[h] for h in selected_horizons}
                        
                        # Create retargeted_active_visibility_results_dict (same as active_visibility_results_dict)
                        retargeted_active_visibility_results_dict = None
                        if retargeted_active_visibility_results_for_dict is not None:
                            retargeted_active_visibility_results_dict = {h: retargeted_active_visibility_results_for_dict[h] for h in selected_horizons}

                        # Extract visibility_info from visual_info (from run_svdd) for active camera
                        active_visibility_results_info = None
                        if visual_info is not None and selected_horizons is not None:
                            final_reward_dict = visual_info.get('final_reward_dict')
                            if final_reward_dict is not None:
                                # Use best_idx (already computed above at line 1988)
                                # Extract visibility_info from final_reward_dict
                                if 'visibility_info' in final_reward_dict and final_reward_dict['visibility_info'] is not None:
                                    visibility_info_all = final_reward_dict['visibility_info']  # {h: {'batch_visible': [bs, N], 'frustum_visible': [bs, N], 'mask_filtered_point_indices': [K] or None}}
                                    # Extract best_idx batch for each horizon
                                    active_visibility_results_info = {}
                                    for h in selected_horizons:
                                        if h in visibility_info_all:
                                            h_info = visibility_info_all[h]
                                            active_visibility_results_info[h] = {
                                                'batch_visible': h_info['batch_visible'][best_idx].copy() if h_info['batch_visible'] is not None else None,  # [N]
                                                'frustum_visible': h_info['frustum_visible'][best_idx].copy() if h_info['frustum_visible'] is not None else None,  # [N]
                                                'mask_filtered_point_indices': h_info['mask_filtered_point_indices'].copy() if h_info['mask_filtered_point_indices'] is not None else None,  # [K] or None
                                                'hit_distances': final_reward_dict.get('hit_distances', None)[best_idx, h, :].copy() if final_reward_dict.get('hit_distances') is not None else None,  # [N]
                                            }
                        
                        # Extract visibility_info from visual_info (from run_svdd) for retargeted camera
                        retargeted_active_visibility_results_info = None
                        if visual_info is not None and selected_horizons is not None:
                            final_reward_dict_retargeted = visual_info.get('final_reward_dict_retargeted')
                            if final_reward_dict_retargeted is not None:
                                # Use best_idx_retargeted (already computed above, check if it exists)
                                if 'best_idx_retargeted' in locals() and best_idx_retargeted is not None:
                                    # Extract visibility_info from final_reward_dict_retargeted
                                    if 'visibility_info' in final_reward_dict_retargeted and final_reward_dict_retargeted['visibility_info'] is not None:
                                        visibility_info_retargeted_all = final_reward_dict_retargeted['visibility_info']  # {h: {'batch_visible': [bs, N], 'frustum_visible': [bs, N], 'mask_filtered_point_indices': [K] or None}}
                                        # Extract best_idx_retargeted batch for each horizon
                                        retargeted_active_visibility_results_info = {}
                                        for h in selected_horizons:
                                            if h in visibility_info_retargeted_all:
                                                h_info = visibility_info_retargeted_all[h]
                                                retargeted_active_visibility_results_info[h] = {
                                                    'batch_visible': h_info['batch_visible'][best_idx_retargeted].copy() if h_info['batch_visible'] is not None else None,  # [N]
                                                    'frustum_visible': h_info['frustum_visible'][best_idx_retargeted].copy() if h_info['frustum_visible'] is not None else None,  # [N]
                                                    'mask_filtered_point_indices': h_info['mask_filtered_point_indices'].copy() if h_info['mask_filtered_point_indices'] is not None else None,  # [K] or None
                                                    'hit_distances': final_reward_dict_retargeted.get('hit_distances', None)[best_idx_retargeted, h, :].copy() if final_reward_dict_retargeted.get('hit_distances') is not None else None,  # [N]
                                                }
                        
                        # Only for robot mesh visualization
                        from egoavflow.svdd import compute_robot_meshes_for_poses
                        T_B_M = task_kwargs.get('T_B_M')
                        robot_future_action = task_kwargs.get('robot_future_action')
                        T_mw = task_kwargs.get('T_mw')
                        T_W_B = np.linalg.inv(T_B_M @ T_mw)
                        end_effector_poses_marker = robot_future_action[:, :9] # [horizon, 9]


                        gripper_action = robot_future_action[:, -1] # [horizon]
                        # NOTE: assume gripper_action is in [0 (open), 1 (close)]
                        
                        gripper_action *= -1.0 # [-1 (close), 0 (open)]
                        gripper_action += 1.0 # [0 (close), 1 (open)]
                        gripper_action *= 0.04 # [0, 0.04] m
                        gripper_angle = gripper_action.copy()
                        
                        
                        robot_meshes = compute_robot_meshes_for_poses(
                            episode_robot_viz, end_effector_poses_marker, gripper_angle, T_B_M, T_W_B, 1, [0]
                        )[0]

                        combined_mesh = scene_mesh
                        
                        for link_name, robot_mesh in robot_meshes.items():
                            if robot_mesh is not None and len(robot_mesh.vertices) > 0:
                                combined_mesh = combined_mesh + robot_mesh

                        # Store visualization data before calling visualizer.visualize()
                        viz_data = {
                            'color_mesh': combined_mesh,
                            'point_cloud': point_cloud,
                            'camera_pose': pose_tensor,
                            'query_points': query_tracking_point_in_world_coordinate,
                            'camera_intrinsics': intrinsics,
                            'image_size': eval_dataset.pixel_tracking_img_size,
                            'current_camera_pose_list': current_camera_pose_list,
                            'current_visibility_results_dict': current_visibility_results_dict,
                            'active_camera_pose_list': active_camera_pose_list,
                            'active_visibility_results_dict': active_visibility_results_dict if 'active_visibility_results_dict' in locals() else None,
                            'retargeted_active_camera_pose_list': retargeted_active_camera_pose_list,
                            'retargeted_active_visibility_results_dict': retargeted_active_visibility_results_dict,
                            'active_visibility_results_info': active_visibility_results_info,
                            'retargeted_active_visibility_results_info': retargeted_active_visibility_results_info,
                            'selected_indices': task_kwargs['selected_horizons'],
                            'query_point_indices': query_point_indices,
                        }
                        visualization_data_list.append(viz_data)
                        
                        visualizer.visualize(
                            color_mesh=combined_mesh, # scene_mesh, 
                            point_cloud=point_cloud,
                            camera_pose=pose_tensor,
                            query_points=query_tracking_point_in_world_coordinate, # [N, 3 or 4] or [h, N, 3 or 4]
                            camera_intrinsics=intrinsics,  # camera intrinsics
                            image_size=eval_dataset.pixel_tracking_img_size,  # image size
                            current_camera_pose_list=current_camera_pose_list,  # current camera pose list
                            current_visibility_results_dict=current_visibility_results_dict,  # visibility results of the current camera
                            active_camera_pose_list=active_camera_pose_list,  # active camera pose list
                            active_visibility_results_dict=active_visibility_results_dict if 'active_visibility_results_dict' in locals() else None,  # visibility results of all selected active cameras
                            retargeted_active_camera_pose_list=retargeted_active_camera_pose_list,  # retargeted active camera pose list
                            retargeted_active_visibility_results_dict=retargeted_active_visibility_results_dict,  # visibility results of the retargeted active cameras
                            active_visibility_results_info=active_visibility_results_info,
                            retargeted_active_visibility_results_info=retargeted_active_visibility_results_info,
                            selected_indices=task_kwargs['selected_horizons'],
                            query_point_indices=query_point_indices,
                            step = step,
                        )


                    # project into images
                    """
                    points_3d: (T, N, 3)
                    intrinsics: (T, 3, 3)
                    extrinsics: (T, 4, 4)
                    Returns:
                        points_2d: (T, N, 2)
                    """
                    

                    # compare generated pixel flow v.s. ground truth pixel flow
                    # Handle padding when step+1+pred_horizon exceeds the available data length
                    # NOTE: episode_pixel_tracking is already processed (moving_mask applied, sampled, normalized)
                    # We need to use the actual episode length, not the padded length
                    start_idx = step + 1
                    end_idx = step + 1 + pred_horizon
                    assert start_idx <= episode_length, f"start_idx: {start_idx}, episode_length: {episode_length}"
                    
                    # Use the actual episode length for bounds checking
                    if end_idx <= episode_length:
                        # Normal case: we have enough data
                        gt_future_pixel_flow = episode_pixel_tracking.copy()[start_idx:end_idx] # [pred_horizon, N, 3] # normalized
                    else:
                        # Need to pad with the last available index
                        available_data = episode_pixel_tracking.copy()[start_idx:episode_length] # Get all available data from start_idx
                        remaining_length = pred_horizon - len(available_data)
                        
                        if remaining_length > 0:
                            # Pad with the last available frame 
                            last_frame = episode_pixel_tracking.copy()[episode_length-1:episode_length] # [1, N, 3] - use actual last frame
                            padding = np.tile(last_frame, (remaining_length, 1, 1)) # [remaining_length, N, 3]
                            gt_future_pixel_flow = np.concatenate([available_data, padding], axis=0) # [pred_horizon, N, 3]
                        else:
                            raise NotImplementedError('this code should not be used')
                            gt_future_pixel_flow = available_data[:pred_horizon] # [pred_horizon, N, 3]
                    # unnormalize
                    gt_future_pixel_flow[..., 0] *= eval_dataset.point_tracking_img_size[0]
                    gt_future_pixel_flow[..., 1] *= eval_dataset.point_tracking_img_size[1]
                    gt_future_pixel_flow = gt_future_pixel_flow.astype(np.int32)
                    gt_future_pixel_flow[..., :2] = np.clip(
                        gt_future_pixel_flow[..., :2], a_min=np.zeros(2), a_max=np.array(eval_dataset.point_tracking_img_size) - 1
                    )

                    
                    # visualization (overlay the whole action_pred sequence on the current frame; later steps are more transparent)
                    frame = pixel_tracking_frames[step].copy() # (H,W,3)
                    def draw_pose_axis(image, pos, rot, K, extrinsics, axis_length=0.1, alpha=1.0, color='red_green_blue'):
                        rmat = R.from_euler('xyz', rot).as_matrix()
                        axis = np.eye(3) * axis_length  # (3,3)
                        axis_rot = (rmat @ axis.T).T  # (3,3)
                        axis_pts = pos[None, :] + axis_rot  # (3,3)
                        pts_3d = np.vstack([pos[None, :], axis_pts])  # (4,3)
                        pts_2d = project_points_to_image(
                            pts_3d[None, :, :], K[None, :, :], extrinsics[None]
                        )[0].astype(int)
                        overlay = image.copy()
                        if color == 'red_green_blue': # NOTE: default in cv2 is BGR, but if image is RGB, then you have to also set color as RGB
                            colors = [(255,0,0), (0,255,0), (0,0,255)]  # RGB: X-red, Y-green, Z-blue
                        elif color == 'magenta_cyan_yellow':
                            colors = [(255,0,255), (255,255,0), (0,255,255)]  # RGB: X-magenta, Y-cyan, Z-yellow
                        elif color == 'violet_lime_orange':
                            colors = [(128, 0, 128), (50, 205, 50), (255, 140, 0)]  # X-purple, Y-lime, Z-orange
                        else:
                            raise NotImplementedError
                        for idx, c in enumerate(colors):
                            cv2.line(overlay, tuple(pts_2d[0]), tuple(pts_2d[idx+1]), c, 2)
                        cv2.addWeighted(overlay, alpha, image, 1-alpha, 0, image)
                        return image
                    
                    n_action = action.shape[0]
                    for k in range(n_action):
                        pos = action[k, :3]
                        ortho6d = action[k, 3:9]
                        rotation_matrix = compute_rotation_matrix_from_ortho6d(ortho6d[None])[0]
                        rot = R.from_matrix(rotation_matrix).as_euler('xyz')
                        alpha = 1.0 - 0.7*(k / n_action)
                        frame = draw_pose_axis(frame, pos, rot, intrinsics, future_extrinsics[0], axis_length=0.025, alpha=alpha, color='red_green_blue')
                        
                        # Visualize retargeted action
                        if len(actions_retargeted_list) > 0 and step < len(actions_retargeted_list):
                            action_retargeted = actions_retargeted_list[step]
                            pos_retargeted = action_retargeted[k, :3]
                            ortho6d_retargeted = action_retargeted[k, 3:9]
                            rotation_matrix_retargeted = compute_rotation_matrix_from_ortho6d(ortho6d_retargeted[None])[0]
                            rot_retargeted = R.from_matrix(rotation_matrix_retargeted).as_euler('xyz')
                            alpha_retargeted = 1.0 - 0.7*(k / n_action)
                            frame = draw_pose_axis(frame, pos_retargeted, rot_retargeted, intrinsics, future_extrinsics[0], axis_length=0.025, alpha=alpha_retargeted, color='magenta_cyan_yellow')
                        

                    # Visualize predicted flow if predict_separate_flow=true or predict_flow=true (from model, not view_model)
                    if flow_pred_unnorm is not None:
                        # flow_pred_unnorm: [pred_horizon, N, D] (marker coordinate, unnormalized)
                        # Get extrinsics for current step (T_mc: marker to camera)
                        if use_marker_coordinate:
                            current_extrinsics = episode_T_mc_opt[step]  # (4, 4)
                        else:
                            current_extrinsics = episode_extrinsics[step]  # (4, 4)
                        
                        # Project predicted flow points to 2D for visualization
                        # flow_pred_unnorm: [pred_horizon, N, D]
                        pred_flow_points_xyz = flow_pred_unnorm[..., :3]  # [pred_horizon, N, 3] (marker coordinate)
                        if flow_pred_unnorm.shape[-1] == 4:
                            pred_flow_vis = flow_pred_unnorm[..., -1:]  # [pred_horizon, N, 1] (visibility)
                        
                        
                        # Reshape for project_points_to_image: [pred_horizon, N, 3] -> [pred_horizon*N, 3]
                        pred_flow_points_flat = pred_flow_points_xyz.reshape(-1, 3)  # [pred_horizon*N, 3]
                        
                        # Project to 2D
                        intrinsics_reshaped = intrinsics[None, :, :]  # (1, 3, 3)
                        extrinsics_reshaped = current_extrinsics[None, :, :]  # (1, 4, 4)
                        pred_flow_points_2d = project_points_to_image(
                            pred_flow_points_flat[None, :, :], intrinsics_reshaped, extrinsics_reshaped
                        )[0]  # [pred_horizon*N, 2]
                        
                        # Reshape back to [pred_horizon, N, 2]
                        pred_flow_points_2d = pred_flow_points_2d.reshape(pred_horizon, -1, 2)  # [pred_horizon, N, 2]
                        
                        # Reshape visibility if available: [pred_horizon, N, 1] -> [pred_horizon, N]
                        if flow_pred_unnorm.shape[-1] == 4:
                            pred_flow_vis_reshaped = pred_flow_vis.reshape(pred_horizon, -1)  # [pred_horizon, N]
                        else:
                            pred_flow_vis_reshaped = None
                        
                        # Draw predicted flow points on frame (each horizon with different transparency)
                        # Colors: green for visible (1), red for invisible (0)
                        green_color = (0, 255, 0)  # Green for visible (RGB)
                        red_color = (255, 0, 0)  # Red for invisible (RGB)
                        for h in range(pred_horizon):
                            alpha_flow = 1.0 - 0.7 * (h / pred_horizon)  # Decreasing transparency
                            points_2d = pred_flow_points_2d[h]  # [N, 2]
                            
                            overlay = frame.copy()
                            for point_idx, point_2d in enumerate(points_2d):
                                u, v = int(point_2d[0]), int(point_2d[1])
                                img_h, img_w = frame.shape[:2]
                                if 0 <= u < img_w and 0 <= v < img_h:
                                    # Determine color based on visibility
                                    if pred_flow_vis_reshaped is not None:
                                        is_visible = pred_flow_vis_reshaped[h, point_idx] > 0.5  # 1 if visible, 0 if invisible
                                        point_color = green_color if is_visible else red_color
                                    else:
                                        # If no visibility info, use green (default)
                                        point_color = green_color
                                    cv2.circle(overlay, (u, v), 3, point_color, -1)
                            cv2.addWeighted(overlay, alpha_flow, frame, 1-alpha_flow, 0, frame)
                    
                    
                    step_info = f"Step: {step}/{len(rgb)-1}"
                    frames.append(frame)
                    

                    if step < len(rgb) - 1:
                        point_tracking_frame = pixel_tracking_frames[step+1].copy()
                        start = time.time()
                        if "cotracker3" in tracker_type:
                            window_frames.pop(0)
                            window_frames.append(point_tracking_frame)
                            video_chunk = torch.tensor(
                                np.stack(window_frames), dtype=torch.float32, device='cuda'
                            ).permute(0, 3, 1, 2)[None]
                            
                            pred_tracks, pred_visibility = tracker_model(video_chunk, one_frame=True)
                            pred_tracks = pred_tracks[0, -1, :queries.shape[1], :] # [B(1), T(1), N, 3] -> [N, 2]
                            pred_visibility = pred_visibility[0, -1, :queries.shape[1]].unsqueeze(-1) # [B(1), T(1), N] -> [N, 1]
                            current_flow_2d = np.concatenate([pred_tracks.detach().cpu().numpy(), pred_visibility.detach().cpu().numpy()], axis=-1)[:, None] # [N, T(1), 3]
                        
                        end = time.time()
                        if step % 20 == 0 :
                            print(f'In evaluation, {tracker_type} pixel tracking time taken: {end - start}')
                        
                        current_flow_2d[..., :2] = np.clip(
                            current_flow_2d[..., :2], a_min=np.zeros(2), a_max=np.array(eval_dataset.pixel_tracking_img_size) - 1
                        )

                        current_flow_2d_norm = current_flow_2d.copy()
                        current_flow_2d_norm[..., 0] = current_flow_2d_norm[..., 0] / eval_dataset.pixel_tracking_img_size[0]
                        current_flow_2d_norm[..., 1] = current_flow_2d_norm[..., 1] / eval_dataset.pixel_tracking_img_size[1]
                        
                        
                        track_2d_list.append(current_flow_2d.copy().astype(np.int32))

                        
                        # varying valeus during episode
                        T_mc = T_mc_transformation[step+1]
                        T_W_C = np.linalg.inv(T_mw) @ T_mc
                        scene_mesh, pose_tensor = get_scene_mesh(pixel_tracking_frames[step+1], pixel_tracking_depths_fake_depth[step+1], intrinsics, relative_poses[step+1], VOXEL_SIZE, episode_mapper)
                    
                    
                # Convert to numpy arrays
                actions_with_masking = np.stack(actions_with_masking) # [T, h, dim (3+6+1)]
                actions_with_masking_pos = actions_with_masking[..., :3]
                actions_with_masking_ortho6d = actions_with_masking[...,  3:9]
                T,H = actions_with_masking_ortho6d.shape[:2]
                actions_with_masking_quat = R.from_matrix(compute_rotation_matrix_from_ortho6d(actions_with_masking_ortho6d.reshape(-1, 6))).as_quat().reshape(T, H, 4)

                # Convert actions_retargeted_list to numpy arrays if available
                if len(actions_retargeted_list) > 0:
                    actions_retargeted = np.stack(actions_retargeted_list) # [T, h, dim (3+6+1)]
                    actions_retargeted_pos = actions_retargeted[..., :3]
                    actions_retargeted_ortho6d = actions_retargeted[..., 3:9]
                    T_retargeted, H_retargeted = actions_retargeted_ortho6d.shape[:2]
                    actions_retargeted_quat = R.from_matrix(compute_rotation_matrix_from_ortho6d(actions_retargeted_ortho6d.reshape(-1, 6))).as_quat().reshape(T_retargeted, H_retargeted, 4)
                else:
                    actions_retargeted_pos = None
                    actions_retargeted_quat = None
                
                actions_gt_list = np.stack(actions_gt_list) # [T, h, dim (3+6+1)]
                actions_gt_pos = actions_gt_list[..., :3]
                actions_gt_ortho6d = actions_gt_list[..., 3:9]
                T,H = actions_gt_ortho6d.shape[:2]
                actions_gt_quat = R.from_matrix(compute_rotation_matrix_from_ortho6d(actions_gt_ortho6d.reshape(-1, 6))).as_quat().reshape(T, H, 4)


                proprioceptions_list = np.stack(proprioceptions_list) # [T, h, dim (3+6+1)]
                proprioceptions_pos = proprioceptions_list[..., :3]
                proprioceptions_ortho6d = proprioceptions_list[..., 3:9]
                T,H = proprioceptions_ortho6d.shape[:2]
                proprioceptions_quat = R.from_matrix(compute_rotation_matrix_from_ortho6d(proprioceptions_ortho6d.reshape(-1, 6))).as_quat().reshape(T, H, 4)

                for t in range(T):
                    # Create subplots
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Position plot
                    pos_colors = ['red', 'green', 'blue']
                    pos_labels = ['X', 'Y', 'Z']
                    
                    for idx in range(3):
                        ax1.plot(actions_with_masking_pos[t, :, idx], color=pos_colors[idx], linestyle='-', linewidth=2, 
                                label=f'Actions with Masking {pos_labels[idx]}', alpha=0.8)
                        if actions_retargeted_pos is not None and t < actions_retargeted_pos.shape[0]:
                            ax1.plot(actions_retargeted_pos[t, :, idx], color=pos_colors[idx], linestyle='--', linewidth=2, 
                                    label=f'Actions Retargeted {pos_labels[idx]}', alpha=0.8)
                        ax1.plot(actions_gt_pos[t, :, idx], color=pos_colors[idx], linestyle=':', linewidth=2,  
                                label=f'Actions GT {pos_labels[idx]}', alpha=0.8)
                        ax1.plot(proprioceptions_pos[t, :, idx], color=pos_colors[idx], linestyle='-.', linewidth=2,  
                                label=f'Proprioceptions {pos_labels[idx]}', alpha=0.8)
                    ax1.set_xlabel('Step')
                    ax1.set_ylabel('Position')
                    ax1.set_title('Position Comparison')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Orientation plot (quaternion)
                    orient_colors = ['red', 'green', 'blue', 'orange']
                    orient_labels = ['Qx', 'Qy', 'Qz', 'Qw']
                    
                    for idx in range(4):
                        ax2.plot(actions_with_masking_quat[t, :, idx], color=orient_colors[idx], linestyle='-', linewidth=2, 
                                label=f'Actions with Masking {orient_labels[idx]}', alpha=0.8)
                        if actions_retargeted_quat is not None and t < actions_retargeted_quat.shape[0]:
                            ax2.plot(actions_retargeted_quat[t, :, idx], color=orient_colors[idx], linestyle='--', linewidth=2, 
                                    label=f'Actions Retargeted {orient_labels[idx]}', alpha=0.8)
                        ax2.plot(actions_gt_quat[t, :, idx], color=orient_colors[idx], linestyle=':', linewidth=2,  
                                label=f'Actions GT {orient_labels[idx]}', alpha=0.8)
                        ax2.plot(proprioceptions_quat[t, :, idx], color=orient_colors[idx], linestyle='-.', linewidth=2,  
                                label=f'Proprioceptions {orient_labels[idx]}', alpha=0.8)
                    ax2.set_xlabel('Step')
                    ax2.set_ylabel('Orientation (Quaternion)')
                    ax2.set_title('Orientation Comparison')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    # Create reward_type-specific save path
                    reward_type_save_path = os.path.join(buffer_result_save_path, f'reward_{reward_type}')
                    os.makedirs(os.path.join(reward_type_save_path, 'debug'), exist_ok=True)
                    plt.savefig(os.path.join(reward_type_save_path, f'debug/policy_action_with_masking_vs_gt_main_loop_{i}_{t}.png'))
                    plt.close()
                
                # Convert view policy data to numpy arrays
                view_actions_list = np.stack(view_actions_list)  # [T, h, 9]
                view_actions_pos = view_actions_list[..., :3]
                view_actions_ortho6d = view_actions_list[..., 3:9]
                T_view, H_view = view_actions_ortho6d.shape[:2]
                view_actions_quat = R.from_matrix(compute_rotation_matrix_from_ortho6d(view_actions_ortho6d.reshape(-1, 6))).as_quat().reshape(T_view, H_view, 4)
                
                view_actions_svdd_list = np.stack(view_actions_svdd_list)  # [T, h, 9]
                view_actions_svdd_pos = view_actions_svdd_list[..., :3]
                view_actions_svdd_ortho6d = view_actions_svdd_list[..., 3:9]
                T_view_svdd, H_view_svdd = view_actions_svdd_ortho6d.shape[:2]
                view_actions_svdd_quat = R.from_matrix(compute_rotation_matrix_from_ortho6d(view_actions_svdd_ortho6d.reshape(-1, 6))).as_quat().reshape(T_view_svdd, H_view_svdd, 4)
                
                # Convert view_actions_svdd_retargeted_list to numpy arrays if available
                if len(view_actions_svdd_retargeted_list) > 0 and all(x is not None for x in view_actions_svdd_retargeted_list):
                    view_actions_svdd_retargeted = np.stack(view_actions_svdd_retargeted_list) # [T, h, 9]
                    view_actions_svdd_retargeted_pos = view_actions_svdd_retargeted[..., :3]
                    view_actions_svdd_retargeted_ortho6d = view_actions_svdd_retargeted[..., 3:9]
                    T_view_svdd_retargeted, H_view_svdd_retargeted = view_actions_svdd_retargeted_ortho6d.shape[:2]
                    view_actions_svdd_retargeted_quat = R.from_matrix(compute_rotation_matrix_from_ortho6d(view_actions_svdd_retargeted_ortho6d.reshape(-1, 6))).as_quat().reshape(T_view_svdd_retargeted, H_view_svdd_retargeted, 4)
                else:
                    view_actions_svdd_retargeted_pos = None
                    view_actions_svdd_retargeted_quat = None
                
                viewpoint_gt_list = np.stack(viewpoint_gt_list)  # [T, h, 9]
                viewpoint_gt_pos = viewpoint_gt_list[..., :3]
                viewpoint_gt_ortho6d = viewpoint_gt_list[..., 3:9]
                T_view_gt, H_view_gt = viewpoint_gt_ortho6d.shape[:2]
                viewpoint_gt_quat = R.from_matrix(compute_rotation_matrix_from_ortho6d(viewpoint_gt_ortho6d.reshape(-1, 6))).as_quat().reshape(T_view_gt, H_view_gt, 4)
                
                # Create plots for view policy action vs viewpoint ground truth
                for t in range(T_view):
                    # Create subplots
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Position plot
                    pos_colors = ['red', 'green', 'blue']
                    pos_labels = ['X', 'Y', 'Z']
                    
                    for idx in range(3):
                        ax1.plot(view_actions_pos[t, :, idx], color=pos_colors[idx], linestyle='-', linewidth=2, 
                                label=f'View Action {pos_labels[idx]}', alpha=0.8)
                        ax1.plot(view_actions_svdd_pos[t, :, idx], color=pos_colors[idx], linestyle='-.', linewidth=2, 
                                label=f'View Action SVDD {pos_labels[idx]}', alpha=0.8)
                        if view_actions_svdd_retargeted_pos is not None and t < view_actions_svdd_retargeted_pos.shape[0]:
                            ax1.plot(view_actions_svdd_retargeted_pos[t, :, idx], color=pos_colors[idx], linestyle=':', linewidth=2, 
                                    label=f'View Action SVDD Retargeted {pos_labels[idx]}', alpha=0.8)
                        ax1.plot(viewpoint_gt_pos[t, :, idx], color=pos_colors[idx], linestyle='--', linewidth=2, 
                                label=f'Viewpoint GT {pos_labels[idx]}', alpha=0.8)
                    ax1.set_xlabel('Step')
                    ax1.set_ylabel('Position')
                    ax1.set_title('View Policy Position Comparison')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Orientation plot (quaternion)
                    orient_colors = ['red', 'green', 'blue', 'orange']
                    orient_labels = ['Qx', 'Qy', 'Qz', 'Qw']
                    
                    for idx in range(4):
                        ax2.plot(view_actions_quat[t, :, idx], color=orient_colors[idx], linestyle='-', linewidth=2, 
                                label=f'View Action {orient_labels[idx]}', alpha=0.8)
                        ax2.plot(view_actions_svdd_quat[t, :, idx], color=orient_colors[idx], linestyle='-.', linewidth=2, 
                                label=f'View Action SVDD {orient_labels[idx]}', alpha=0.8)
                        if view_actions_svdd_retargeted_quat is not None and t < view_actions_svdd_retargeted_quat.shape[0]:
                            ax2.plot(view_actions_svdd_retargeted_quat[t, :, idx], color=orient_colors[idx], linestyle=':', linewidth=2, 
                                    label=f'View Action SVDD Retargeted {orient_labels[idx]}', alpha=0.8)
                        ax2.plot(viewpoint_gt_quat[t, :, idx], color=orient_colors[idx], linestyle='--', linewidth=2,  
                                label=f'Viewpoint GT {orient_labels[idx]}', alpha=0.8)
                    ax2.set_xlabel('Step')
                    ax2.set_ylabel('Orientation (Quaternion)')
                    ax2.set_title('View Policy Orientation Comparison')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    # Create reward_type-specific save path
                    reward_type_save_path = os.path.join(buffer_result_save_path, f'reward_{reward_type}')
                    os.makedirs(os.path.join(reward_type_save_path, 'debug'), exist_ok=True)
                    plt.savefig(os.path.join(reward_type_save_path, f'debug/view_policy_action_vs_viewpoint_gt_main_loop_{i}_{t}.png'))
                    plt.close()
                

                # Plot visibility rewards comparison at the end of episode
                if len(episode_visibility_rewards) > 0 and len(episode_active_visibility_rewards) > 0:
                    timesteps = np.arange(len(episode_visibility_rewards))
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(timesteps, episode_visibility_rewards, label='visibility_rewards', marker='o', linewidth=1, markersize=2)
                    
                    # Plot each batch index separately
                    # Find max batch size across all timesteps
                    max_batch_size = max([len(rewards) if isinstance(rewards, list) else 1 for rewards in episode_active_visibility_rewards])
                    
                    # Plot each batch index
                    for b_idx in range(max_batch_size):
                        batch_rewards = []
                        batch_timesteps = []
                        for t, rewards in enumerate(episode_active_visibility_rewards):
                            if isinstance(rewards, list) and b_idx < len(rewards):
                                batch_rewards.append(rewards[b_idx])
                                batch_timesteps.append(t)
                            elif not isinstance(rewards, list) and b_idx == 0:
                                # Handle scalar case (backward compatibility)
                                batch_rewards.append(rewards)
                                batch_timesteps.append(t)
                        
                        if len(batch_rewards) > 0:
                            label = f'active_visibility_rewards_b{b_idx}'
                            ax.plot(batch_timesteps, batch_rewards, label=label, marker='s', linewidth=1, markersize=2)
                    
                    # Plot retargeted visibility rewards
                    if len(episode_active_visibility_rewards_retargeted) > 0:
                        max_batch_size_retargeted = max([len(rewards) if isinstance(rewards, list) else 1 for rewards in episode_active_visibility_rewards_retargeted])
                        for b_idx in range(max_batch_size_retargeted):
                            batch_rewards_retargeted = []
                            batch_timesteps_retargeted = []
                            for t, rewards in enumerate(episode_active_visibility_rewards_retargeted):
                                if isinstance(rewards, list) and b_idx < len(rewards) and rewards[b_idx] is not None:
                                    batch_rewards_retargeted.append(rewards[b_idx])
                                    batch_timesteps_retargeted.append(t)
                                elif not isinstance(rewards, list) and b_idx == 0 and rewards is not None:
                                    batch_rewards_retargeted.append(rewards)
                                    batch_timesteps_retargeted.append(t)
                            
                            if len(batch_rewards_retargeted) > 0:
                                label = f'active_visibility_rewards_retargeted_b{b_idx}'
                                ax.plot(batch_timesteps_retargeted, batch_rewards_retargeted, label=label, marker='^', linewidth=1, markersize=2, linestyle='--', alpha=0.7)
                    
                    ax.set_xlabel('Timestep', fontsize=12)
                    ax.set_ylabel('Visibility Reward', fontsize=12)
                    ax.set_title(f'Visibility Rewards Comparison - Episode {i} (reward_type: {reward_type})', fontsize=14)
                    ax.legend(fontsize=11)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    # Create reward_type-specific save path
                    reward_type_save_path = os.path.join(buffer_result_save_path, f'reward_{reward_type}')
                    os.makedirs(os.path.join(reward_type_save_path, 'debug'), exist_ok=True)
                    plt.savefig(os.path.join(reward_type_save_path, f'debug/visibility_rewards_comparison_episode_{i}.png'), dpi=150)
                    plt.close()
                    print(f"Saved visibility rewards comparison plot for episode {i} (reward_type: {reward_type})")
                    
                    # Plot simplified comparison: visibility_rewards vs active_visibility_rewards_max only
                    max_active_rewards = []
                    max_timesteps = []
                    for t, rewards in enumerate(episode_active_visibility_rewards):
                        if isinstance(rewards, list) and len(rewards) > 0:
                            max_reward = max(rewards)
                            max_active_rewards.append(max_reward)
                            max_timesteps.append(t)
                        elif not isinstance(rewards, list):
                            # Handle scalar case (backward compatibility)
                            max_active_rewards.append(rewards)
                            max_timesteps.append(t)
                    
                    # Compute max retargeted rewards
                    max_active_rewards_retargeted = []
                    max_timesteps_retargeted = []
                    for t, rewards in enumerate(episode_active_visibility_rewards_retargeted):
                        if isinstance(rewards, list) and len(rewards) > 0:
                            valid_rewards = [r for r in rewards if r is not None]
                            if len(valid_rewards) > 0:
                                max_reward = max(valid_rewards)
                                max_active_rewards_retargeted.append(max_reward)
                                max_timesteps_retargeted.append(t)
                        elif not isinstance(rewards, list) and rewards is not None:
                            max_active_rewards_retargeted.append(rewards)
                            max_timesteps_retargeted.append(t)
                    
                    if len(max_active_rewards) > 0:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(timesteps, episode_visibility_rewards, label='visibility_rewards', linewidth=1, marker='s')
                        ax.plot(max_timesteps, max_active_rewards, label='active_visibility_rewards_max', linewidth=1, marker='s')
                        if len(max_active_rewards_retargeted) > 0:
                            ax.plot(max_timesteps_retargeted, max_active_rewards_retargeted, label='active_visibility_rewards_retargeted_max', linewidth=1, marker='^', linestyle='--', alpha=0.7)
                        ax.set_xlabel('Timestep', fontsize=12)
                        ax.set_ylabel('Visibility Reward', fontsize=12)
                        ax.set_title(f'Visibility Rewards Comparison (Simplified) - Episode {i} (reward_type: {reward_type})', fontsize=14)
                        ax.legend(fontsize=11)
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(reward_type_save_path, f'debug/visibility_rewards_comparison_simplified_episode_{i}.png'), dpi=150)
                        plt.close()
                        print(f"Saved simplified visibility rewards comparison plot for episode {i} (reward_type: {reward_type})")
                
                # Plot visibility rewards among tracking points comparison at the end of episode
                if len(episode_visibility_rewards_among_tracking) > 0 and len(episode_active_visibility_rewards_among_tracking) > 0:
                    timesteps = np.arange(len(episode_visibility_rewards_among_tracking))
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(timesteps, episode_visibility_rewards_among_tracking, label='visibility_rewards_among_tracking', marker='o', linewidth=1, markersize=2)
                    
                    # Plot each batch index separately
                    # Find max batch size across all timesteps
                    max_batch_size = max([len(rewards) if isinstance(rewards, list) else 1 for rewards in episode_active_visibility_rewards_among_tracking])
                    
                    # Plot each batch index
                    for b_idx in range(max_batch_size):
                        batch_rewards = []
                        batch_timesteps = []
                        for t, rewards in enumerate(episode_active_visibility_rewards_among_tracking):
                            if isinstance(rewards, list) and b_idx < len(rewards):
                                batch_rewards.append(rewards[b_idx])
                                batch_timesteps.append(t)
                            elif not isinstance(rewards, list) and b_idx == 0:
                                # Handle scalar case (backward compatibility)
                                batch_rewards.append(rewards)
                                batch_timesteps.append(t)
                        
                        if len(batch_rewards) > 0:
                            label = f'active_visibility_rewards_among_tracking_b{b_idx}'
                            ax.plot(batch_timesteps, batch_rewards, label=label, marker='s', linewidth=1, markersize=2)
                    
                    # Plot retargeted visibility rewards among tracking
                    if len(episode_active_visibility_rewards_among_tracking_retargeted) > 0:
                        max_batch_size_retargeted = max([len(rewards) if isinstance(rewards, list) else 1 for rewards in episode_active_visibility_rewards_among_tracking_retargeted])
                        for b_idx in range(max_batch_size_retargeted):
                            batch_rewards_retargeted = []
                            batch_timesteps_retargeted = []
                            for t, rewards in enumerate(episode_active_visibility_rewards_among_tracking_retargeted):
                                if isinstance(rewards, list) and b_idx < len(rewards) and rewards[b_idx] is not None:
                                    batch_rewards_retargeted.append(rewards[b_idx])
                                    batch_timesteps_retargeted.append(t)
                                elif not isinstance(rewards, list) and b_idx == 0 and rewards is not None:
                                    batch_rewards_retargeted.append(rewards)
                                    batch_timesteps_retargeted.append(t)
                            
                            if len(batch_rewards_retargeted) > 0:
                                label = f'active_visibility_rewards_among_tracking_retargeted_b{b_idx}'
                                ax.plot(batch_timesteps_retargeted, batch_rewards_retargeted, label=label, marker='^', linewidth=1, markersize=2, linestyle='--', alpha=0.7)
                    
                    ax.set_xlabel('Timestep', fontsize=12)
                    ax.set_ylabel('Visibility Reward (Among Tracking Points)', fontsize=12)
                    ax.set_title(f'Visibility Rewards Among Tracking Points Comparison - Episode {i} (reward_type: {reward_type})', fontsize=14)
                    ax.legend(fontsize=11)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    # Create reward_type-specific save path
                    reward_type_save_path = os.path.join(buffer_result_save_path, f'reward_{reward_type}')
                    os.makedirs(os.path.join(reward_type_save_path, 'debug'), exist_ok=True)
                    plt.savefig(os.path.join(reward_type_save_path, f'debug/visibility_rewards_among_tracking_points_comparison_episode_{i}.png'), dpi=150)
                    plt.close()
                    print(f"Saved visibility rewards among tracking points comparison plot for episode {i} (reward_type: {reward_type})")
                    
                    # Plot simplified comparison: visibility_rewards_among_tracking vs active_visibility_rewards_among_tracking_max only
                    max_active_rewards = []
                    max_timesteps = []
                    for t, rewards in enumerate(episode_active_visibility_rewards_among_tracking):
                        if isinstance(rewards, list) and len(rewards) > 0:
                            max_reward = max(rewards)
                            max_active_rewards.append(max_reward)
                            max_timesteps.append(t)
                        elif not isinstance(rewards, list):
                            # Handle scalar case (backward compatibility)
                            max_active_rewards.append(rewards)
                            max_timesteps.append(t)
                    
                    # Compute max retargeted rewards among tracking
                    max_active_rewards_among_tracking_retargeted = []
                    max_timesteps_among_tracking_retargeted = []
                    for t, rewards in enumerate(episode_active_visibility_rewards_among_tracking_retargeted):
                        if isinstance(rewards, list) and len(rewards) > 0:
                            valid_rewards = [r for r in rewards if r is not None]
                            if len(valid_rewards) > 0:
                                max_reward = max(valid_rewards)
                                max_active_rewards_among_tracking_retargeted.append(max_reward)
                                max_timesteps_among_tracking_retargeted.append(t)
                        elif not isinstance(rewards, list) and rewards is not None:
                            max_active_rewards_among_tracking_retargeted.append(rewards)
                            max_timesteps_among_tracking_retargeted.append(t)
                    
                    if len(max_active_rewards) > 0:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(timesteps, episode_visibility_rewards_among_tracking, label='visibility_rewards_among_tracking', linewidth=1, marker='s')
                        ax.plot(max_timesteps, max_active_rewards, label='active_visibility_rewards_among_tracking_max', linewidth=1, marker='s')
                        if len(max_active_rewards_among_tracking_retargeted) > 0:
                            ax.plot(max_timesteps_among_tracking_retargeted, max_active_rewards_among_tracking_retargeted, label='active_visibility_rewards_among_tracking_retargeted_max', linewidth=1, marker='^', linestyle='--', alpha=0.7)
                        ax.set_xlabel('Timestep', fontsize=12)
                        ax.set_ylabel('Visibility Reward (Among Tracking Points)', fontsize=12)
                        ax.set_title(f'Visibility Rewards Among Tracking Points Comparison (Simplified) - Episode {i} (reward_type: {reward_type})', fontsize=14)
                        ax.legend(fontsize=11)
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(reward_type_save_path, f'debug/visibility_rewards_among_tracking_points_comparison_simplified_episode_{i}.png'), dpi=150)
                        plt.close()
                        print(f"Saved simplified visibility rewards among tracking points comparison plot for episode {i} (reward_type: {reward_type})")
                
                # Plot visibility rewards wo_tracking_lost comparison at the end of episode
                if len(episode_visibility_rewards_wo_tracking_lost) > 0 and len(episode_active_visibility_rewards_wo_tracking_lost) > 0:
                    timesteps = np.arange(len(episode_visibility_rewards_wo_tracking_lost))
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(timesteps, episode_visibility_rewards_wo_tracking_lost, label='visibility_rewards_wo_tracking_lost', marker='o', linewidth=1, markersize=2)
                    
                    # Plot each batch index separately
                    # Find max batch size across all timesteps
                    max_batch_size = max([len(rewards) if isinstance(rewards, list) else 1 for rewards in episode_active_visibility_rewards_wo_tracking_lost])
                    
                    # Plot each batch index
                    for b_idx in range(max_batch_size):
                        batch_rewards = []
                        batch_timesteps = []
                        for t, rewards in enumerate(episode_active_visibility_rewards_wo_tracking_lost):
                            if isinstance(rewards, list) and b_idx < len(rewards):
                                batch_rewards.append(rewards[b_idx])
                                batch_timesteps.append(t)
                            elif not isinstance(rewards, list) and b_idx == 0:
                                # Handle scalar case (backward compatibility)
                                batch_rewards.append(rewards)
                                batch_timesteps.append(t)
                        
                        if len(batch_rewards) > 0:
                            label = f'active_visibility_rewards_wo_tracking_lost_b{b_idx}'
                            ax.plot(batch_timesteps, batch_rewards, label=label, marker='s', linewidth=1, markersize=2)
                    
                    # Plot retargeted visibility rewards wo_tracking_lost
                    if len(episode_active_visibility_rewards_wo_tracking_lost_retargeted) > 0:
                        max_batch_size_retargeted = max([len(rewards) if isinstance(rewards, list) else 1 for rewards in episode_active_visibility_rewards_wo_tracking_lost_retargeted])
                        for b_idx in range(max_batch_size_retargeted):
                            batch_rewards_retargeted = []
                            batch_timesteps_retargeted = []
                            for t, rewards in enumerate(episode_active_visibility_rewards_wo_tracking_lost_retargeted):
                                if isinstance(rewards, list) and b_idx < len(rewards) and rewards[b_idx] is not None:
                                    batch_rewards_retargeted.append(rewards[b_idx])
                                    batch_timesteps_retargeted.append(t)
                                elif not isinstance(rewards, list) and b_idx == 0 and rewards is not None:
                                    batch_rewards_retargeted.append(rewards)
                                    batch_timesteps_retargeted.append(t)
                            
                            if len(batch_rewards_retargeted) > 0:
                                label = f'active_visibility_rewards_wo_tracking_lost_retargeted_b{b_idx}'
                                ax.plot(batch_timesteps_retargeted, batch_rewards_retargeted, label=label, marker='^', linewidth=1, markersize=2, linestyle='--', alpha=0.7)
                    
                    ax.set_xlabel('Timestep', fontsize=12)
                    ax.set_ylabel('Visibility Reward (Without Tracking Lost)', fontsize=12)
                    ax.set_title(f'Visibility Rewards Without Tracking Lost Comparison - Episode {i} (reward_type: {reward_type})', fontsize=14)
                    ax.legend(fontsize=11)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    # Create reward_type-specific save path
                    reward_type_save_path = os.path.join(buffer_result_save_path, f'reward_{reward_type}')
                    os.makedirs(os.path.join(reward_type_save_path, 'debug'), exist_ok=True)
                    plt.savefig(os.path.join(reward_type_save_path, f'debug/visibility_rewards_wo_tracking_lost_comparison_episode_{i}.png'), dpi=150)
                    plt.close()
                    print(f"Saved visibility rewards wo_tracking_lost comparison plot for episode {i} (reward_type: {reward_type})")
                    
                    # Plot simplified comparison: visibility_rewards_wo_tracking_lost vs active_visibility_rewards_wo_tracking_lost_max only
                    max_active_rewards = []
                    max_timesteps = []
                    for t, rewards in enumerate(episode_active_visibility_rewards_wo_tracking_lost):
                        if isinstance(rewards, list) and len(rewards) > 0:
                            max_reward = max(rewards)
                            max_active_rewards.append(max_reward)
                            max_timesteps.append(t)
                        elif not isinstance(rewards, list):
                            # Handle scalar case (backward compatibility)
                            max_active_rewards.append(rewards)
                            max_timesteps.append(t)
                    
                    # Compute max retargeted rewards wo_tracking_lost
                    max_active_rewards_wo_tracking_lost_retargeted = []
                    max_timesteps_wo_tracking_lost_retargeted = []
                    for t, rewards in enumerate(episode_active_visibility_rewards_wo_tracking_lost_retargeted):
                        if isinstance(rewards, list) and len(rewards) > 0:
                            valid_rewards = [r for r in rewards if r is not None]
                            if len(valid_rewards) > 0:
                                max_reward = max(valid_rewards)
                                max_active_rewards_wo_tracking_lost_retargeted.append(max_reward)
                                max_timesteps_wo_tracking_lost_retargeted.append(t)
                        elif not isinstance(rewards, list) and rewards is not None:
                            max_active_rewards_wo_tracking_lost_retargeted.append(rewards)
                            max_timesteps_wo_tracking_lost_retargeted.append(t)
                    
                    if len(max_active_rewards) > 0:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(timesteps, episode_visibility_rewards_wo_tracking_lost, label='visibility_rewards_wo_tracking_lost', linewidth=1, marker='s')
                        ax.plot(max_timesteps, max_active_rewards, label='active_visibility_rewards_wo_tracking_lost_max', linewidth=1, marker='s')
                        if len(max_active_rewards_wo_tracking_lost_retargeted) > 0:
                            ax.plot(max_timesteps_wo_tracking_lost_retargeted, max_active_rewards_wo_tracking_lost_retargeted, label='active_visibility_rewards_wo_tracking_lost_retargeted_max', linewidth=1, marker='^', linestyle='--', alpha=0.7)
                        ax.set_xlabel('Timestep', fontsize=12)
                        ax.set_ylabel('Visibility Reward (Without Tracking Lost)', fontsize=12)
                        ax.set_title(f'Visibility Rewards Without Tracking Lost Comparison (Simplified) - Episode {i} (reward_type: {reward_type})', fontsize=14)
                        ax.legend(fontsize=11)
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(reward_type_save_path, f'debug/visibility_rewards_wo_tracking_lost_comparison_simplified_episode_{i}.png'), dpi=150)
                        plt.close()
                        print(f"Saved simplified visibility rewards wo_tracking_lost comparison plot for episode {i} (reward_type: {reward_type})")
                
                # Plot detailed SVDD reward components at the end of episode (separate plot for each batch index)
                if len(episode_svdd_total_rewards) > 0:
                    # Find max batch size across all timesteps
                    max_batch_size = 1
                    for reward_list in [episode_svdd_total_rewards, episode_svdd_visibility_rewards, 
                                       episode_svdd_close_to_nominal_rewards, episode_svdd_close_to_query_points_rewards,
                                       episode_svdd_smoothness_rewards]:
                        for reward in reward_list:
                            if reward is not None and isinstance(reward, list):
                                max_batch_size = max(max_batch_size, len(reward))
                    
                    # Create separate plot for each batch index
                    for b_idx in range(max_batch_size):
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Helper function to extract rewards for a specific batch index
                        def extract_batch_rewards(reward_list, batch_idx):
                            """Extract rewards for a specific batch index from reward_list"""
                            batch_rewards = []
                            timesteps = []
                            for timestep, reward in enumerate(reward_list):
                                if reward is not None:
                                    if isinstance(reward, list):
                                        if batch_idx < len(reward):
                                            batch_rewards.append(float(reward[batch_idx]))
                                            timesteps.append(timestep)
                                    else:
                                        # Scalar case - only use for batch_idx == 0
                                        if batch_idx == 0:
                                            batch_rewards.append(float(reward))
                                            timesteps.append(timestep)
                            return timesteps, batch_rewards
                        
                        # Extract rewards for this batch index
                        timesteps_total, rewards_total = extract_batch_rewards(episode_svdd_total_rewards, b_idx)
                        timesteps_vis, rewards_vis = extract_batch_rewards(episode_svdd_visibility_rewards, b_idx)
                        timesteps_close, rewards_close = extract_batch_rewards(episode_svdd_close_to_nominal_rewards, b_idx)
                        timesteps_close_query, rewards_close_query = extract_batch_rewards(episode_svdd_close_to_query_points_rewards, b_idx)
                        timesteps_smoothness, rewards_smoothness = extract_batch_rewards(episode_svdd_smoothness_rewards, b_idx)
                        
                        # Extract retargeted rewards for this batch index
                        timesteps_total_retargeted, rewards_total_retargeted = extract_batch_rewards(episode_svdd_total_rewards_retargeted, b_idx)
                        timesteps_vis_retargeted, rewards_vis_retargeted = extract_batch_rewards(episode_svdd_visibility_rewards_retargeted, b_idx)
                        timesteps_close_retargeted, rewards_close_retargeted = extract_batch_rewards(episode_svdd_close_to_nominal_rewards_retargeted, b_idx)
                        timesteps_close_query_retargeted, rewards_close_query_retargeted = extract_batch_rewards(episode_svdd_close_to_query_points_rewards_retargeted, b_idx)
                        timesteps_smoothness_retargeted, rewards_smoothness_retargeted = extract_batch_rewards(episode_svdd_smoothness_rewards_retargeted, b_idx)
                        
                        # Plot total reward
                        if len(rewards_total) > 0:
                            ax.plot(timesteps_total, rewards_total, label='SVDD Total Reward', 
                                   marker='o', linewidth=1, markersize=2, color='blue')
                        
                        # Plot visibility reward component
                        if len(rewards_vis) > 0:
                            ax.plot(timesteps_vis, rewards_vis, label='SVDD Visibility Reward', 
                                   marker='s', linewidth=1, markersize=2, color='green')
                        
                        # Plot close_to_nominal reward component (if available)
                        if len(rewards_close) > 0:
                            ax.plot(timesteps_close, rewards_close, label='SVDD Close-to-Nominal Reward', 
                                   marker='^', linewidth=1, markersize=2, color='orange')
                        
                        # Plot close_to_query_points reward component (if available)
                        if len(rewards_close_query) > 0:
                            ax.plot(timesteps_close_query, rewards_close_query, label='SVDD Close-to-Query-Points Reward', 
                                   marker='v', linewidth=1, markersize=2, color='purple')
                        
                        # Plot smoothness penalty component (if available)
                        if len(rewards_smoothness) > 0:
                            ax.plot(timesteps_smoothness, rewards_smoothness, label='SVDD Smoothness Penalty', 
                                   marker='x', linewidth=1, markersize=2, color='red')
                        
                        # Plot retargeted rewards (with dashed lines to distinguish)
                        if len(rewards_total_retargeted) > 0:
                            ax.plot(timesteps_total_retargeted, rewards_total_retargeted, label='SVDD Total Reward (Retargeted)', 
                                   marker='o', linewidth=1, markersize=2, color='blue', linestyle='--', alpha=0.7)
                        
                        if len(rewards_vis_retargeted) > 0:
                            ax.plot(timesteps_vis_retargeted, rewards_vis_retargeted, label='SVDD Visibility Reward (Retargeted)', 
                                   marker='s', linewidth=1, markersize=2, color='green', linestyle='--', alpha=0.7)
                        
                        if len(rewards_close_retargeted) > 0:
                            ax.plot(timesteps_close_retargeted, rewards_close_retargeted, label='SVDD Close-to-Nominal Reward (Retargeted)', 
                                   marker='^', linewidth=1, markersize=2, color='orange', linestyle='--', alpha=0.7)
                        
                        if len(rewards_close_query_retargeted) > 0:
                            ax.plot(timesteps_close_query_retargeted, rewards_close_query_retargeted, label='SVDD Close-to-Query-Points Reward (Retargeted)', 
                                   marker='v', linewidth=1, markersize=2, color='purple', linestyle='--', alpha=0.7)
                        
                        if len(rewards_smoothness_retargeted) > 0:
                            ax.plot(timesteps_smoothness_retargeted, rewards_smoothness_retargeted, label='SVDD Smoothness Penalty (Retargeted)', 
                                   marker='x', linewidth=1, markersize=2, color='red', linestyle='--', alpha=0.7)
                        
                        ax.set_xlabel('Timestep', fontsize=12)
                        ax.set_ylabel('Reward', fontsize=12)
                        ax.set_title(f'SVDD Reward Components - Episode {i}, Batch {b_idx} (reward_type: {reward_type})', fontsize=14)
                        ax.legend(fontsize=11)
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        # Create reward_type-specific save path
                        reward_type_save_path = os.path.join(buffer_result_save_path, f'reward_{reward_type}')
                        os.makedirs(os.path.join(reward_type_save_path, 'debug'), exist_ok=True)
                        plt.savefig(os.path.join(reward_type_save_path, f'debug/svdd_reward_components_episode_{i}_b_{b_idx}.png'), dpi=150)
                        plt.close()
                        print(f"Saved SVDD reward components plot for episode {i}, batch {b_idx} (reward_type: {reward_type})")


                # visualize tool from Im2Flow2Act
                from egoavflow.common.utility.viz import viz_point_tracking_flow
                viz_threshold = 10 # unit: pixel
                viz_num_point = len(track_2d_list)
                viz_offset = 0
                
                # convert to pixel_tracking_img_size
                video = np.stack(frames, axis=0) # [T, H, W, C]
                traj_2d = np.concatenate(track_2d_list, axis=1) # [N, T, 3]
                
                    
                viz_seq = traj_2d.copy()
                if len(viz_seq) == 0:
                    print(f'episode {i} has no moving points!')
                    continue
                else:
                    if viz_num_point != -1:
                        if eval_dataset.use_dift: # viz all
                            pass
                        else:
                            viz_mask = np.round(
                                np.linspace(0, len(viz_seq) - 1, viz_num_point)
                            ).astype(int)
                            viz_seq = viz_seq[viz_mask]

                    # Create reward_type-specific save path
                    reward_type_save_path = os.path.join(buffer_result_save_path, f'reward_{reward_type}')
                    os.makedirs(reward_type_save_path, exist_ok=True)
                    
                    viz_point_tracking_flow(
                        video, # [T, H, W, C]
                        viz_seq,
                        os.path.join(
                            reward_type_save_path,
                            f"episode_{i+viz_offset}_flow_policy_3d.mp4",
                        ),
                        viz_key=[1],
                        point_per_key=-1, # len(viz_seq),
                        viz_horizon=pred_horizon,
                        draw_line=True,
                        output_format='mp4',
                        fps=10,
                    )
                

                # save nvblox visualization 
                episode_mapper.update_color_mesh()
                episode_mapper.get_color_mesh().save(os.path.join(reward_type_save_path, f"episode_{i+viz_offset}_3d_visualization.ply"))
                
                # save video
                if visualizer.video_save and (len(visualizer.captured_frames_mesh) > 0 or len(visualizer.captured_frames_point_cloud) > 0):
                    # Generate video path with episode_idx and reward_type
                    video_filename = f"visualization_{i}_{reward_type}.mp4"
                    episode_idx_for_video = f"{i}_{reward_type}"  # Combine episode_idx and reward_type for unique video path
                    visualizer.save_video(episode_idx_for_video, buffer_result_save_path)
                    print(f"Mesh frames: {len(visualizer.captured_frames_mesh)}, Point cloud frames: {len(visualizer.captured_frames_point_cloud)}")
                
                episode_mapper.clear()

                episode_tracking_data = {
                    'T_mc': _stack_or_empty(episode_T_mc_list, trailing_shape=(4, 4), dtype=np.float32),
                    'track_2d': _stack_or_empty(episode_track_2d_list, dtype=np.float32),
                    'track_3d': _stack_or_empty(episode_track_3d_list, dtype=np.float32),
                }
                tracking_lengths = {key: value.shape[0] for key, value in episode_tracking_data.items()}
                if len(set(tracking_lengths.values())) != 1:
                    print(f"Warning: tracking data lengths differ for episode {i}: {tracking_lengths}")

                valid_gt_flow_mse = [m for m in action_mse_with_gt_future_flow_list if m is not None]
                action_comparison_summary = {
                    'predicted_flow_action_mse_mean': {
                        key: float(np.mean([m[key] for m in action_mse_with_masking_list]))
                        for key in ['mse_all', 'mse_pos', 'mse_rot6d', 'mse_gripper']
                    } if len(action_mse_with_masking_list) > 0 else None,
                    'gt_future_flow_action_mse_mean': {
                        key: float(np.mean([m[key] for m in valid_gt_flow_mse]))
                        for key in ['mse_all', 'mse_pos', 'mse_rot6d', 'mse_gripper']
                    } if len(valid_gt_flow_mse) > 0 else None,
                }
                action_comparison_data = {
                    'episode_idx': i,
                    'reward_type': reward_type,
                    'episode_tracking_data': episode_tracking_data,
                    'episode_tracking_lengths': tracking_lengths,
                    'actions_with_predicted_flow': actions_with_masking,
                    'actions_with_gt_future_flow': actions_with_gt_future_flow,
                    'actions_gt': actions_gt_list,
                    'pred_future_flow': pred_future_flow_list,
                    'gt_future_flow': gt_future_flow_list,
                    'action_mse_with_predicted_flow': action_mse_with_masking_list,
                    'action_mse_with_gt_future_flow': action_mse_with_gt_future_flow_list,
                    'action_comparison_summary': action_comparison_summary,
                    'gt_future_flow_action_errors': gt_future_flow_action_errors,
                }
                reward_type_save_path = os.path.join(buffer_result_save_path, f'reward_{reward_type}')
                os.makedirs(reward_type_save_path, exist_ok=True)
                action_comparison_save_path = os.path.join(reward_type_save_path, f'action_flow_comparison_episode_{i}.pkl')
                with open(action_comparison_save_path, 'wb') as f:
                    pickle.dump(action_comparison_data, f)
                print(f"Action/flow comparison data saved to {action_comparison_save_path}")

                # Save visualization data for this reward_type 
                if len(visualization_data_list) > 0:
                    from egoavflow.common.utility.visualization_data import save_visualization_data
                    
                    # Create reward_type-specific save path
                    reward_type_save_path = os.path.join(buffer_result_save_path, f'reward_{reward_type}')
                    os.makedirs(reward_type_save_path, exist_ok=True)
                    
                    # Save visualization data with reward_type in filename
                    save_filename = f"visualization_data_episode_{i}.pkl"
                    save_path = os.path.join(reward_type_save_path, save_filename)
                    save_visualization_data(visualization_data_list, save_path)
                    print(f"Visualization data saved to {save_path}")

                # End of reward_type loop
                print(f"Completed processing Episode {i} with reward_type: {reward_type}")