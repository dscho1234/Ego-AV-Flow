import torch
import numpy as np
import time
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from itertools import accumulate


from egoavflow.diffusion_policy.dataloader.diffusion_bc_dataset import unnormalize_data

set_alpha_to_one = True # as in DDIMScheduler of diffusers
from egoavflow.visibility import OFFSET_DISTANCE, batch_raycasting_with_scene
from egoavflow.utils import compute_ortho6d_from_rotation_matrix, compute_rotation_matrix_from_ortho6d, compute_fix_mask_weights


def get_random_flow_mask_count(args):
    count = getattr(args.training, "random_flow_mask_count", 0)
    return 0 if count is None else int(count)


def apply_random_flow_mask_count(flow_mask, random_flow_mask_count):
    if flow_mask is None or random_flow_mask_count <= 0:
        return flow_mask

    flow_mask = flow_mask.clone()
    B, N = flow_mask.shape
    target_masked = min(int(random_flow_mask_count), max(N - 1, 0))
    for b in range(B):
        already_masked = int((~flow_mask[b]).sum().item())
        num_to_add = target_masked - already_masked
        if num_to_add <= 0:
            continue

        visible_indices = torch.where(flow_mask[b])[0]
        # Keep at least one visible point for attention stability.
        num_to_add = min(num_to_add, max(int(visible_indices.numel()) - 1, 0))
        if num_to_add <= 0:
            continue

        selected = visible_indices[
            torch.randperm(visible_indices.numel(), device=flow_mask.device)[:num_to_add]
        ]
        flow_mask[b, selected] = False

    return flow_mask


def use_pixel_flow_for_robot_policy(args):
    return bool(getattr(args.training, "use_pixel_flow_for_robot_policy", False))


def get_condition_input_flow_dim(args):
    return int(getattr(args.training, "robot_policy_input_flow_dim", args.training.input_flow_dim))


def create_combined_scene_with_robot(scene_mesh, robot_meshes):
    """
    Create a raycasting scene that combines the scene mesh and the robot meshes
    
    Args:
        scene_mesh: nvblox ColorMesh or Open3D mesh
        robot_meshes: dict of robot link meshes {link_name: mesh}
    
    Returns:
        scene: o3d.t.geometry.RaycastingScene
    """
    
    # 1. convert the scene mesh to Open3D
    
    # already an Open3D mesh
    scene_open3d = scene_mesh
    
    if scene_open3d is None:
        print("    Failed to convert scene mesh, returning None")
        return None
    
    # 2. merge all meshes into one
    combined_mesh = scene_open3d
    
    # 3. add each robot link mesh
    for link_name, robot_mesh in robot_meshes.items():
        if robot_mesh is not None and len(robot_mesh.vertices) > 0:
            combined_mesh = combined_mesh + robot_mesh
    
    
    # 4. convert the merged mesh to a tensor mesh
    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(combined_mesh)
    if mesh_tensor is None:
        print("    ERROR: Failed to convert mesh to tensor")
        return None
    # 5. create the ray casting scene
    scene = o3d.t.geometry.RaycastingScene()
    scene_id = scene.add_triangles(mesh_tensor)
    
    
    return scene

def is_point_in_camera_frustum_batch(points_3d, camera_positions, camera_rotations_ortho6d, camera_intrinsics, image_size, pixel_bias = 5.0):
    """
    Check whether batched 3D points lie inside the camera frustums
    
    Args:
        points_3d: 3D points [M*N, 3] (world frame)
        camera_positions: camera positions [M, 3] (world frame)
        camera_rotations_rpy: camera rotations [M, 3] (roll, pitch, yaw in degrees)
        camera_intrinsics: camera intrinsics (3x3)
        image_size: image size (width, height)
    
    Returns:
        frustum_visible: [M*N] boolean array, True if point is in frustum
    """
    M = camera_positions.shape[0]  # Number of cameras
    N = points_3d.shape[0] // M    # Number of query points per camera
    assert points_3d.shape[0] == M * N, f"Expected {M*N} points, got {points_3d.shape[0]}"
    
    # Initialize result array
    frustum_visible = np.zeros(M * N, dtype=bool)
    
    # Process each camera
    for m in range(M):
        start_idx = m * N
        end_idx = (m + 1) * N
        
        # Get points for this camera
        points_for_camera = points_3d[start_idx:end_idx]  # [N, 3]
        camera_position = camera_positions[m]  # [3]
        camera_rotation_ortho6d = camera_rotations_ortho6d[m]  # [6]
        
        # Convert to camera coordinates
        points_3d_np = np.array(points_for_camera, dtype=np.float64)
        camera_position_np = np.array(camera_position, dtype=np.float64)
        
        # Camera rotation matrix (RPY order)
        R_cam = compute_rotation_matrix_from_ortho6d(camera_rotation_ortho6d[None])[0] # [3, 3]
        
        
        # Transform to camera coordinates
        points_cam = (R_cam.T @ (points_3d_np - camera_position_np).T).T  # [N, 3]
        
        # Check if points are in front of camera
        valid_depth = points_cam[:, 2] > 0  # [N]
        
        # Project to image plane
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        u = fx * points_cam[:, 0] / points_cam[:, 2] + cx  # [N]
        v = fy * points_cam[:, 1] / points_cam[:, 2] + cy  # [N]
        
        # Check if points are within image bounds
        width, height = image_size
        in_bounds = (u >= 0+pixel_bias) & (u < width-pixel_bias) & (v >= 0+pixel_bias) & (v < height-pixel_bias)  # [N]
        
        # Final frustum check: valid depth AND in bounds
        frustum_visible[start_idx:end_idx] = valid_depth & in_bounds
    
    return frustum_visible


def _get_variance(alphas_cumprod, timestep, prev_timestep):
    alpha_prod_t = alphas_cumprod[timestep]
    
    final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else alphas_cumprod[0]

    alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance

# NOTE: same as DDIMScheduler, except the default eta value! (without eta, sampling becomes completely deterministic)
def custom_ddim_step(
    args,
    noise_scheduler,
    num_inference_steps,
    model_output, # noise_pred # eps_theta(x_t)
    # old_model_output: torch.FloatTensor,
    timestep, # t
    sample, # latent # x_t
    eta = 1.0,
    use_clipped_model_output = False,
    generator=None,
    variance_noise=None,
    ):
    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    # 1. get previous step value (=t-1)
    prev_timestep = timestep - args.noise_scheduler.num_train_timesteps // num_inference_steps # 1000 /50

    # 2. compute alphas, betas
    alphas_cumprod = noise_scheduler.alphas_cumprod.clone()

    alpha_prod_t = alphas_cumprod[timestep]
    
    final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else alphas_cumprod[0]

    alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if args.noise_scheduler.prediction_type == 'epsilon': # default
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
        
    else:
        raise NotImplementedError

    # 4. Clip or threshold "predicted x_0"
    
    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = _get_variance(alphas_cumprod, timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon
    

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    
    
    if eta > 0:
        if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape).to(model_output.device)
            
        variance = std_dev_t * variance_noise
        
        prev_sample = prev_sample_mean + variance

    else:
        prev_sample = prev_sample_mean
        
    
    return prev_sample.to(dtype=sample.dtype)


def compute_se3_distance(T1, T2):
    """
    Compute SE(3) distance between two transformation matrices using log map (fully vectorized)
    
    Args:
        T1: [4, 4] or [B, 4, 4] or [B, H, 4, 4] SE(3) transformation matrix
        T2: [4, 4] or [B, 4, 4] or [B, H, 4, 4] SE(3) transformation matrix
    
    Returns:
        distance: scalar or [B] or [B, H] SE(3) distance
    """
    # Compute relative transformation: T_rel = T1^(-1) @ T2
    if T1.ndim == 2:
        # Single matrix case [4, 4]
        T1_inv = np.linalg.inv(T1)
        T_rel = T1_inv @ T2
        # Compute log map
        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]
        
        # SO(3) logarithm
        theta = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
        if theta < 1e-12:
            w = np.zeros(3)
        else:
            w = theta / (2 * np.sin(theta)) * np.array([R_rel[2,1] - R_rel[1,2], R_rel[0,2] - R_rel[2,0], R_rel[1,0] - R_rel[0,1]])
        
        # SE(3) logarithm
        if theta < 1e-12:
            V_inv = np.eye(3)
        else:
            skew_w = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
            V_inv = np.eye(3) - 0.5 * skew_w + (1 - np.sin(theta) / theta) / (theta**2) * (skew_w @ skew_w)
        
        v = V_inv @ t_rel
        se3_log_coords = np.hstack([v, w])
        
        # Compute distance as L2 norm of se(3) log coordinates
        distance = np.linalg.norm(se3_log_coords)
        return distance
    elif T1.ndim == 3:
        # [B, 4, 4] - fully vectorized
        B = T1.shape[0]
        
        # Compute inverse of T1: [B, 4, 4]
        T1_inv = np.linalg.inv(T1)  # [B, 4, 4]
        
        # Compute relative transformation: T_rel = T1_inv @ T2
        T_rel = np.einsum('bij,bjk->bik', T1_inv, T2)  # [B, 4, 4]
        
        # Extract rotation and translation
        R_rel = T_rel[:, :3, :3]  # [B, 3, 3]
        t_rel = T_rel[:, :3, 3]  # [B, 3]
        
        # SO(3) logarithm (vectorized)
        # Trace for each batch: [B]
        trace_R = np.trace(R_rel, axis1=1, axis2=2)  # [B]
        theta = np.arccos(np.clip((trace_R - 1) / 2, -1, 1))  # [B]
        
        # Compute w (axis-angle) for each batch
        # w = theta / (2 * sin(theta)) * [R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]
        w = np.zeros((B, 3))  # [B, 3]
        sin_theta = np.sin(theta)  # [B]
        valid_mask = theta >= 1e-12  # [B]
        
        if np.any(valid_mask):
            # Extract skew-symmetric part of rotation matrix
            w[valid_mask, 0] = (R_rel[valid_mask, 2, 1] - R_rel[valid_mask, 1, 2]) * theta[valid_mask] / (2 * sin_theta[valid_mask])
            w[valid_mask, 1] = (R_rel[valid_mask, 0, 2] - R_rel[valid_mask, 2, 0]) * theta[valid_mask] / (2 * sin_theta[valid_mask])
            w[valid_mask, 2] = (R_rel[valid_mask, 1, 0] - R_rel[valid_mask, 0, 1]) * theta[valid_mask] / (2 * sin_theta[valid_mask])
        
        # SE(3) logarithm (vectorized)
        # Compute skew-symmetric matrices for w: [B, 3, 3]
        skew_w = np.zeros((B, 3, 3))
        skew_w[:, 0, 1] = -w[:, 2]
        skew_w[:, 0, 2] = w[:, 1]
        skew_w[:, 1, 0] = w[:, 2]
        skew_w[:, 1, 2] = -w[:, 0]
        skew_w[:, 2, 0] = -w[:, 1]
        skew_w[:, 2, 1] = w[:, 0]
        
        # Compute V_inv for each batch: [B, 3, 3]
        V_inv = np.tile(np.eye(3)[None, :, :], (B, 1, 1))  # [B, 3, 3]
        
        if np.any(valid_mask):
            # V_inv = I - 0.5 * skew_w + (1 - sin(theta)/theta) / theta^2 * (skew_w @ skew_w)
            skew_w_sq = np.einsum('bij,bjk->bik', skew_w, skew_w)  # [B, 3, 3]
            coeff = np.zeros(B)
            coeff[valid_mask] = (1 - sin_theta[valid_mask] / theta[valid_mask]) / (theta[valid_mask] ** 2)
            
            V_inv[valid_mask] = (np.eye(3)[None, :, :] - 
                                 0.5 * skew_w[valid_mask] + 
                                 coeff[valid_mask, None, None] * skew_w_sq[valid_mask])
        
        # Compute v = V_inv @ t_rel: [B, 3]
        v = np.einsum('bij,bj->bi', V_inv, t_rel)  # [B, 3]
        
        # Stack v and w: [B, 6]
        se3_log_coords = np.hstack([v, w])  # [B, 6]
        
        # Compute distance as L2 norm of se(3) log coordinates: [B]
        distances = np.linalg.norm(se3_log_coords, axis=1)  # [B]
        
        return distances
    elif T1.ndim == 4:
        # [B, H, 4, 4] - reshape to [B*H, 4, 4] and use vectorized [B, 4, 4] version
        B, H = T1.shape[:2]
        T1_reshaped = T1.reshape(B * H, 4, 4)  # [B*H, 4, 4]
        T2_reshaped = T2.reshape(B * H, 4, 4)  # [B*H, 4, 4]
        
        distances_flat = compute_se3_distance(T1_reshaped, T2_reshaped)  # [B*H]
        distances = distances_flat.reshape(B, H)  # [B, H]
        return distances
    else:
        raise ValueError(f"Unsupported T1.ndim: {T1.ndim}")

def compute_smoothness(trajectory, scale=1.0):
    """
    Compute smoothness-based reward penalty for an SE(3) trajectory.
    Penalty is based on SE(3) distances between adjacent indices in the trajectory.
    
    Args:
        trajectory: [H, 4, 4] or [B, H, 4, 4] SE(3) transformation matrices
    
    Returns:
        penalty: scalar or [B] smoothness penalty (higher value = less smooth = more penalty)
    """

     # ex) only translation 2cm -> dist = 0.02

    if trajectory.ndim == 3:
        # [H, 4, 4] - single trajectory
        H = trajectory.shape[0]
        if H < 2:
            # Need at least 2 poses to compute smoothness
            return 0.0
        
        # Compute distances between adjacent poses
        # trajectory[i] and trajectory[i+1] for i in range(H-1)
        T_current = trajectory[:-1]  # [H-1, 4, 4]
        T_next = trajectory[1:]      # [H-1, 4, 4]
        
        # Compute SE(3) distances for all adjacent pairs
        distances = compute_se3_distance(T_current, T_next)  # [H-1]
        
        # Penalty is the sum (or mean) of distances
        penalty = np.sum(distances)
        return penalty
    
    elif trajectory.ndim == 4:
        # [B, H, 4, 4] - batch of trajectories
        B, H = trajectory.shape[:2]
        if H < 2:
            # Need at least 2 poses to compute smoothness
            return np.zeros(B)
        
        # Compute distances between adjacent poses for each batch
        # trajectory[:, i] and trajectory[:, i+1] for i in range(H-1)
        T_current = trajectory[:, :-1, :, :]  # [B, H-1, 4, 4]
        T_next = trajectory[:, 1:, :, :]       # [B, H-1, 4, 4]
        
        # Reshape to [B*(H-1), 4, 4] for vectorized computation
        T_current_reshaped = T_current.reshape(B * (H - 1), 4, 4)  # [B*(H-1), 4, 4]
        T_next_reshaped = T_next.reshape(B * (H - 1), 4, 4)       # [B*(H-1), 4, 4]
        
        # Compute SE(3) distances for all adjacent pairs
        distances_flat = compute_se3_distance(T_current_reshaped, T_next_reshaped)  # [B*(H-1)]
        distances = distances_flat.reshape(B, H - 1)  # [B, H-1]
        
        # Penalty is the mean of distances for each batch
        penalty = np.mean(distances, axis=1)  # [B]
        return penalty * scale
    
    else:
        raise ValueError(f"Unsupported trajectory.ndim: {trajectory.ndim}. Expected 3 ([H, 4, 4]) or 4 ([B, H, 4, 4])")


def compute_proprioception_penalty(camera_poses, robot_future_action):
    """
    Compute proprioception-based reward penalty.
    Penalty is based on Euclidean distance between camera position and robot end effector position.
    Closer distances result in higher penalty (to avoid camera getting too close to robot).
    Uses exponential decay: penalty = exp(-distance / decay_scale)
    
    Args:
        camera_poses: [B, H, 9] or [H, 9] - camera poses (position + ortho6d)
        robot_future_action: [H, 9] or [H, 10] - robot end effector poses (position + ortho6d + optional gripper)
    
    Returns:
        penalty: scalar or [B] proprioception penalty (higher value = closer = more penalty)
    """
    # Extract camera positions
    if camera_poses.ndim == 2:
        # [H, 9] - single trajectory
        camera_positions = camera_poses[:, :3]  # [H, 3]
        H = camera_poses.shape[0]
        B = 1
    elif camera_poses.ndim == 3:
        # [B, H, 9] - batch of trajectories
        camera_positions = camera_poses[:, :, :3]  # [B, H, 3]
        B, H = camera_poses.shape[:2]
    else:
        raise ValueError(f"Unsupported camera_poses.ndim: {camera_poses.ndim}. Expected 2 ([H, 9]) or 3 ([B, H, 9])")
    
    # Extract robot end effector positions
    robot_positions = robot_future_action[:, :3]  # [H, 3]
    
    # Check if robot_future_action has correct horizon
    if robot_positions.shape[0] != H:
        raise ValueError(f"robot_future_action horizon ({robot_positions.shape[0]}) does not match camera_poses horizon ({H})")
    
    # Decay scale for exponential penalty (smaller value = steeper decay = higher penalty for close distances)
    decay_scale = 0.1  # Controls how quickly penalty increases as distance decreases
    
    if camera_poses.ndim == 2:
        # [H, 3] vs [H, 3] - compute distances for each horizon step
        distances = np.linalg.norm(camera_positions - robot_positions, axis=-1)  # [H]
        
        # Compute penalty using exponential decay: penalty = exp(-distance / decay_scale)
        # Closer distances (smaller distance) result in higher penalty
        penalty_array = np.exp(-distances / decay_scale)  # [H]
        
        penalty = np.sum(penalty_array)  # scalar
        return penalty
    else:
        # [B, H, 3] vs [H, 3] - expand robot_positions to match batch dimension
        robot_positions_expanded = np.tile(robot_positions[None, :, :], (B, 1, 1))  # [B, H, 3]
        
        # Compute Euclidean distances: [B, H]
        distances = np.linalg.norm(camera_positions - robot_positions_expanded, axis=-1)  # [B, H]
        
        # Compute penalty using exponential decay: penalty = exp(-distance / decay_scale)
        # Closer distances (smaller distance) result in higher penalty
        penalty_array = np.exp(-distances / decay_scale)  # [B, H]
        
        penalty = np.mean(penalty_array, axis=1)  # [B]
        return penalty

# ============================================================================
# Reward Component Functions
# ============================================================================

def compute_camera_center_reward(x, task_kwargs, args, device, visibility_results):
    """
    Compute camera center reward (points closer to image center get higher reward).
    
    Args:
        x: [bs, horizon, 9] - viewpoint predictions (position + ortho6d)
        task_kwargs: dict containing camera_intrinsics, image_size, query_point, etc.
        args: args object with svdd config
        device: torch device
        visibility_results: [bs, horizon, N] - visibility results for each point
    
    Returns:
        camera_center_reward: [bs] torch.Tensor - reward for each batch
    """
    batch_size, horizon, D = x.shape
    assert D == 9, f"Expected 9D input (position + ortho6d), got {D}"
    
    # Get camera intrinsics and image size from task_kwargs
    camera_intrinsics = task_kwargs.get('camera_intrinsics')
    image_size = task_kwargs.get('image_size')
    if camera_intrinsics is None or image_size is None:
        raise ValueError("camera_intrinsics and image_size must be provided in task_kwargs for 'camera_center' reward")
    
    # Get weight from task_kwargs or args.svdd config
    camera_center_weight = getattr(args.svdd, 'camera_center_weight', 0.5)
    selected_horizons = task_kwargs.get('selected_horizons', list(range(horizon)))
    
    # Get query points
    query_points = task_kwargs.get('query_point')
    query_point_indices = task_kwargs.get('query_point_indices', None)
    predict_separate_flow = task_kwargs.get('predict_separate_flow', False)
    
    # Convert x to numpy if it's a tensor
    x_np = x.copy()  # [bs, horizon, 9] in marker coordinate
    
    # Extract camera positions and rotations directly from marker coordinates
    camera_positions_marker = x_np[:, :, :3]  # [bs, horizon, 3] in marker coordinate
    camera_rotations_ortho6d = x_np[:, :, 3:9]  # [bs, horizon, 6]
    
    # Handle query points (already in marker coordinate)
    if predict_separate_flow and query_points.ndim == 3:
        if query_points.shape[2] == 4:
            query_points_xyz = query_points[:, :, :3]  # [horizon, N, 3] in marker coordinate
        else:
            query_points_xyz = query_points  # [horizon, N, 3] in marker coordinate
        if query_point_indices is not None:
            query_points_xyz = query_points_xyz[:, query_point_indices, :]  # [horizon, M, 3]
        N = query_points_xyz.shape[1]
    else:
        if query_points.ndim == 1:
            query_points = query_points.reshape(1, -1)
        if query_points.shape[1] == 4:
            query_points_xyz = query_points[:, :3]  # [N, 3] in marker coordinate
        else:
            query_points_xyz = query_points  # [N, 3] in marker coordinate
        if query_point_indices is not None:
            query_points_xyz = query_points_xyz[query_point_indices, :]  # [M, 3]
        N = query_points_xyz.shape[0]
        # Expand to [horizon, N, 3] for consistent processing
        query_points_xyz = np.tile(query_points_xyz[None, :, :], (horizon, 1, 1))  # [horizon, N, 3] in marker coordinate
    
    # Get image center from camera intrinsics
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    width, height = image_size
    
    # Compute camera center reward for each selected horizon (fully vectorized)
    camera_center_rewards_list = []
    
    # Normalize by image diagonal for scale-invariant reward
    image_diagonal = np.sqrt(width**2 + height**2)
    distance_scale = 0.3  # Scale factor (adjust as needed)
    
    for h in selected_horizons:
        # Get camera positions and rotations for this horizon (in marker coordinate)
        camera_positions_h = camera_positions_marker[:, h, :]  # [bs, 3] in marker coordinate
        camera_rotations_h = camera_rotations_ortho6d[:, h, :]  # [bs, 6]
        
        # Get query points for this horizon (in marker coordinate)
        query_points_h = query_points_xyz[h]  # [N, 3] in marker coordinate
        
        # Get visibility results for this horizon
        visibility_h = visibility_results[:, h, :]  # [bs, N] - True if visible
        
        # Expand query points for all batches: [bs, N, 3]
        query_points_expanded = np.tile(query_points_h[None, :, :], (batch_size, 1, 1))  # [bs, N, 3]
        
        # Convert all camera rotations to rotation matrices: [bs, 3, 3]
        R_cam_all = compute_rotation_matrix_from_ortho6d(camera_rotations_h)  # [bs, 3, 3]
        
        # Transform all points to camera coordinates (vectorized, from marker coordinate)
        points_relative = query_points_expanded - camera_positions_h[:, None, :]  # [bs, N, 3] in marker coordinate
        points_cam_all = np.einsum('bji,bnj->bni', R_cam_all, points_relative)  # [bs, N, 3] in camera coordinate
        
        # Project to image plane (vectorized)
        valid_depth_all = points_cam_all[:, :, 2] > 0  # [bs, N]
        
        # Project all points: [bs, N]
        u_all = fx * points_cam_all[:, :, 0] / (points_cam_all[:, :, 2] + 1e-8) + cx  # [bs, N]
        v_all = fy * points_cam_all[:, :, 1] / (points_cam_all[:, :, 2] + 1e-8) + cy  # [bs, N]
        
        # Compute distance from image center: [bs, N]
        distances_from_center_all = np.sqrt((u_all - cx)**2 + (v_all - cy)**2)  # [bs, N]
        
        # Normalize distances: [bs, N]
        normalized_distances_all = distances_from_center_all / image_diagonal  # [bs, N]
        
        # Convert distances to rewards: [bs, N]
        point_rewards_all = np.exp(-normalized_distances_all / distance_scale)  # [bs, N]
        
        # Combine visibility and valid depth masks: [bs, N]
        valid_mask = visibility_h & valid_depth_all  # [bs, N]
        
        # Compute average reward for each batch (only for valid points) - fully vectorized
        point_rewards_masked = np.where(valid_mask, point_rewards_all, 0.0)  # [bs, N]
        valid_counts = valid_mask.sum(axis=1)  # [bs] - number of valid points per batch
        batch_rewards = np.where(valid_counts > 0, 
                               point_rewards_masked.sum(axis=1) / valid_counts, 
                               0.0)  # [bs]
        
        camera_center_rewards_list.append(batch_rewards)  # [bs]
    
    # Average over selected horizons: [bs]
    camera_center_reward = np.array(camera_center_rewards_list).mean(axis=0)  # [bs]
    camera_center_reward = torch.from_numpy(camera_center_reward).float().to(device)  # [bs]
    
    return camera_center_reward


def compute_camera_margin_reward(x, task_kwargs, args, device, visibility_reward_type):
    """
    Compute camera margin reward (visibility robustness to camera viewpoint perturbations).
    
    Args:
        x: [bs, horizon, 9] - viewpoint predictions (position + ortho6d)
        task_kwargs: dict containing task parameters
        args: args object with svdd config
        device: torch device
        visibility_reward_type: 'wo_consider_tracking_lost' or 'w_consider_tracking_lost'
    
    Returns:
        camera_margin_reward: [bs] torch.Tensor - reward for each batch
    """
    batch_size, horizon, D = x.shape
    assert D == 9, f"Expected 9D input (position + ortho6d), got {D}"
    
    # Get parameters for camera margin computation
    camera_margin_num_perturbations = getattr(args.svdd, 'camera_margin_num_perturbations', 5)
    camera_margin_perturbation_translation_scale = getattr(args.svdd, 'camera_margin_perturbation_translation_scale', 0.02)
    camera_margin_perturbation_rotation_scale = getattr(args.svdd, 'camera_margin_perturbation_rotation_scale', 0.05)
    camera_margin_variance_penalty_weight = getattr(args.svdd, 'camera_margin_variance_penalty_weight', 0.1)
    
    selected_horizons = task_kwargs.get('selected_horizons', list(range(horizon)))
    
    # Convert x to numpy if it's a tensor
    x_np = x.copy()  # [bs, horizon, 9]
    
    # Generate all perturbations at once: [num_perturbations, bs, horizon, 9]
    x_expanded = np.tile(x_np[None, :, :, :], (camera_margin_num_perturbations, 1, 1, 1))  # [num_perturbations, bs, horizon, 9]
    
    # Perturb translation (first 3 dimensions) - fully vectorized
    translation_perturbation = np.random.randn(camera_margin_num_perturbations, batch_size, horizon, 3) * camera_margin_perturbation_translation_scale
    x_expanded[:, :, :, :3] += translation_perturbation
    
    # Perturb rotation (last 6 dimensions - ortho6d) - fully vectorized
    ortho6d_original = x_expanded[:, :, :, 3:9]  # [num_perturbations, bs, horizon, 6]
    ortho6d_flat = ortho6d_original.reshape(-1, 6)  # [num_perturbations * bs * horizon, 6]
    
    # Convert to rotation matrices - vectorized
    rot_matrices = compute_rotation_matrix_from_ortho6d(ortho6d_flat)  # [num_perturbations * bs * horizon, 3, 3]
    
    # Convert to axis-angle representation - vectorized
    axis_angles = R.from_matrix(rot_matrices).as_rotvec()  # [num_perturbations * bs * horizon, 3]
    
    # Add perturbation in axis-angle space - vectorized
    rotation_perturbation = np.random.randn(*axis_angles.shape) * camera_margin_perturbation_rotation_scale  # [num_perturbations * bs * horizon, 3]
    axis_angles_perturbed = axis_angles + rotation_perturbation
    
    # Convert back to rotation matrices - vectorized
    rot_matrices_perturbed = R.from_rotvec(axis_angles_perturbed).as_matrix()  # [num_perturbations * bs * horizon, 3, 3]
    
    # Convert back to ortho6d - vectorized
    ortho6d_perturbed = compute_ortho6d_from_rotation_matrix(rot_matrices_perturbed)  # [num_perturbations * bs * horizon, 6]
    x_expanded[:, :, :, 3:9] = ortho6d_perturbed.reshape(camera_margin_num_perturbations, batch_size, horizon, 6)
    
    # Reshape to [num_perturbations * bs, horizon, 9] for batch processing
    x_perturbed_all = x_expanded.reshape(camera_margin_num_perturbations * batch_size, horizon, 9)  # [num_perturbations * bs, horizon, 9]
    
    # Compute visibility for all perturbations at once (fully vectorized, no for loop)
    vis_reward_perturbed_all, _, _, _, _, vis_reward_wo_tracking_lost_perturbed_all, _, _ = compute_visibility(x_perturbed_all, task_kwargs)
    
    # Reshape back to [num_perturbations, bs]
    if visibility_reward_type == 'wo_consider_tracking_lost':
        perturbed_visibility_rewards = vis_reward_wo_tracking_lost_perturbed_all.reshape(camera_margin_num_perturbations, batch_size)  # [num_perturbations, bs]
    elif visibility_reward_type == 'w_consider_tracking_lost':
        perturbed_visibility_rewards = vis_reward_perturbed_all.reshape(camera_margin_num_perturbations, batch_size)  # [num_perturbations, bs]
    else:
        perturbed_visibility_rewards = vis_reward_perturbed_all.reshape(camera_margin_num_perturbations, batch_size)  # [num_perturbations, bs]
    
    # Hybrid approach: Minimum visibility (worst-case) + Variance penalty (stability)
    min_visibility = np.min(perturbed_visibility_rewards, axis=0)  # [bs] - worst-case visibility, already in [0, 1]
    visibility_variance = np.var(perturbed_visibility_rewards, axis=0)  # [bs] - variance across perturbations, in [0, 0.25]
    
    camera_margin_reward = min_visibility * (1.0 - camera_margin_variance_penalty_weight * visibility_variance)  # [bs] - in [0, 1]
    camera_margin_reward = torch.from_numpy(camera_margin_reward).float().to(device)  # [bs]
    
    return camera_margin_reward


def compute_close_to_query_points_reward(x, task_kwargs, args, device):
    """
    Compute close_to_query_points reward (camera closer to query points gets higher reward).
    
    Args:
        x: [bs, horizon, 9] - viewpoint predictions (position + ortho6d)
        task_kwargs: dict containing query_point, query_point_indices, etc.
        args: args object with svdd config
        device: torch device
    
    Returns:
        close_to_query_points_reward: [bs] torch.Tensor - reward for each batch
    """
    batch_size, horizon, D = x.shape
    assert D == 9, f"Expected 9D input (position + ortho6d), got {D}"
    
    # Get query_points and selected_horizons from task_kwargs
    query_points = task_kwargs.get('query_point')
    query_point_indices = task_kwargs.get('query_point_indices', None)
    selected_horizons = task_kwargs.get('selected_horizons', list(range(horizon)))
    predict_separate_flow = task_kwargs.get('predict_separate_flow', False)
    
    # Convert x to numpy if it's a tensor
    x_np = x.copy()
    
    # Extract camera center positions from x (viewpoint predictions)
    camera_centers = x_np[:, :, :3]  # [bs, horizon, 3] - camera center positions in marker coordinate
    
    # Handle different query point shapes based on predict_separate_flow flag
    if predict_separate_flow and query_points.ndim == 3:
        if query_points.shape[2] == 4:
            query_points_xyz = query_points[:, :, :3]  # [horizon, N, 3]
        else:
            query_points_xyz = query_points  # [horizon, N, 3]
        
        if query_point_indices is not None:
            query_points_xyz = query_points_xyz[:, query_point_indices, :]  # [horizon, M, 3]
    else:
        if query_points.ndim == 1:
            query_points = query_points.reshape(1, -1)
        
        if query_points.shape[1] == 4:
            query_points_xyz = query_points[:, :3]  # [N, 3]
        else:
            query_points_xyz = query_points  # [N, 3]
        
        if query_point_indices is not None:
            query_points_xyz = query_points_xyz[query_point_indices, :]  # [M, 3]
        
        # Expand to [horizon, M, 3] for consistent processing
        query_points_xyz = np.tile(query_points_xyz[None, :, :], (horizon, 1, 1))  # [horizon, M, 3]
    
    # Compute distances only for selected_horizons: [bs, len(selected_horizons)]
    camera_centers_selected = camera_centers[:, selected_horizons, :]  # [bs, len(selected_horizons), 3]
    query_points_selected = query_points_xyz[selected_horizons, :, :]  # [len(selected_horizons), N, 3]
    
    # Compute distances using broadcasting
    camera_centers_expanded = camera_centers_selected[:, :, None, :]  # [bs, len(selected_horizons), 1, 3]
    query_points_expanded = query_points_selected[None, :, :, :]  # [1, len(selected_horizons), N, 3]
    
    # Compute Euclidean distances: [bs, len(selected_horizons), N]
    distances = np.linalg.norm(query_points_expanded - camera_centers_expanded, axis=-1)  # [bs, len(selected_horizons), N]
    
    # Convert distances to rewards with peak at 0.3m
    min_distance_threshold = 0.3  # meters - optimal distance for maximum reward
    distance_scale = 1.0  # Scale factor for distance
    distance_deviation = np.abs(distances - min_distance_threshold)  # [bs, len(selected_horizons), N]
    point_rewards = np.exp(-distance_deviation / distance_scale)  # [bs, len(selected_horizons), N]
    
    # Mean over all query points: [bs, len(selected_horizons)]
    close_to_query_points_reward_selected = point_rewards.mean(axis=-1)  # [bs, len(selected_horizons)]
    
    # Mean over selected_horizons and convert to tensor
    close_to_query_points_reward = torch.from_numpy(close_to_query_points_reward_selected.mean(axis=1)).float().to(device)  # [bs]
    
    return close_to_query_points_reward


def compute_far_from_query_points_reward(x, task_kwargs, args, device):
    """
    Compute far_from_query_points reward (camera farther from query points gets higher reward).
    
    Args:
        x: [bs, horizon, 9] - viewpoint predictions (position + ortho6d)
        task_kwargs: dict containing query_point, query_point_indices, etc.
        args: args object with svdd config
        device: torch device
    
    Returns:
        far_from_query_points_reward: [bs] torch.Tensor - reward for each batch
    """
    batch_size, horizon, D = x.shape
    assert D == 9, f"Expected 9D input (position + ortho6d), got {D}"
    
    # Get query_points and selected_horizons from task_kwargs
    query_points = task_kwargs.get('query_point')
    query_point_indices = task_kwargs.get('query_point_indices', None)
    selected_horizons = task_kwargs.get('selected_horizons', list(range(horizon)))
    predict_separate_flow = task_kwargs.get('predict_separate_flow', False)
    
    # Convert x to numpy if it's a tensor
    x_np = x.copy()
    
    # Extract camera center positions from x (viewpoint predictions)
    camera_centers = x_np[:, :, :3]  # [bs, horizon, 3] - camera center positions in marker coordinate
    
    # Handle different query point shapes based on predict_separate_flow flag
    if predict_separate_flow and query_points.ndim == 3:
        if query_points.shape[2] == 4:
            query_points_xyz = query_points[:, :, :3]  # [horizon, N, 3]
        else:
            query_points_xyz = query_points  # [horizon, N, 3]
        
        if query_point_indices is not None:
            query_points_xyz = query_points_xyz[:, query_point_indices, :]  # [horizon, M, 3]
    else:
        if query_points.ndim == 1:
            query_points = query_points.reshape(1, -1)
        
        if query_points.shape[1] == 4:
            query_points_xyz = query_points[:, :3]  # [N, 3]
        else:
            query_points_xyz = query_points  # [N, 3]
        
        if query_point_indices is not None:
            query_points_xyz = query_points_xyz[query_point_indices, :]  # [M, 3]
        
        # Expand to [horizon, M, 3] for consistent processing
        query_points_xyz = np.tile(query_points_xyz[None, :, :], (horizon, 1, 1))  # [horizon, M, 3]
    
    # Compute distances only for selected_horizons: [bs, len(selected_horizons)]
    camera_centers_selected = camera_centers[:, selected_horizons, :]  # [bs, len(selected_horizons), 3]
    query_points_selected = query_points_xyz[selected_horizons, :, :]  # [len(selected_horizons), N, 3]
    
    # Compute distances using broadcasting
    camera_centers_expanded = camera_centers_selected[:, :, None, :]  # [bs, len(selected_horizons), 1, 3]
    query_points_expanded = query_points_selected[None, :, :, :]  # [1, len(selected_horizons), N, 3]
    
    # Compute Euclidean distances: [bs, len(selected_horizons), N]
    distances = np.linalg.norm(query_points_expanded - camera_centers_expanded, axis=-1)  # [bs, len(selected_horizons), N]
    
    # Convert distances to rewards (farther = higher reward)
    distance_scale = 1.0  # Scale factor for distance
    point_rewards = 1.0 - np.exp(-distances / distance_scale)  # [bs, len(selected_horizons), N]
    
    # Mean over all query points: [bs, len(selected_horizons)]
    far_from_query_points_reward_selected = point_rewards.mean(axis=-1)  # [bs, len(selected_horizons)]
    
    # Mean over selected_horizons and convert to tensor
    far_from_query_points_reward = torch.from_numpy(far_from_query_points_reward_selected.mean(axis=1)).float().to(device)  # [bs]
    
    return far_from_query_points_reward


def compute_close_to_nominal_reward(x, task_kwargs, args, device):
    """
    Compute close_to_nominal reward (camera closer to nominal trajectory gets higher reward).
    
    Args:
        x: [bs, horizon, 9] - viewpoint predictions (position + ortho6d)
        task_kwargs: dict containing nominal_trajectory, selected_horizons
        args: args object with svdd config
        device: torch device
    
    Returns:
        close_to_nominal_reward: [bs] torch.Tensor - reward for each batch
    """
    batch_size, horizon, D = x.shape
    assert D == 9, f"Expected 9D input (position + ortho6d), got {D}"
    
    # Get nominal trajectory and selected_horizons from task_kwargs
    nominal_trajectory = task_kwargs.get('nominal_trajectory')  # [h, 4, 4] SE(3) in marker coordinate
    selected_horizons = task_kwargs.get('selected_horizons', list(range(horizon)))
    
    x_np = x.copy()  # [bs, horizon, 9]
    
    # Convert x (viewpoint predictions) to SE(3) matrices
    x_se3 = convert_9d_to_se3(x_np)  # [bs, horizon, 4, 4]
    
    # Compute SE(3) distances to nominal trajectory (fully vectorized)
    x_se3_selected = x_se3[:, selected_horizons, :, :]  # [bs, len(selected_horizons), 4, 4]
    nominal_trajectory_selected = nominal_trajectory[selected_horizons, :, :]  # [len(selected_horizons), 4, 4]
    
    # Expand nominal_trajectory_selected to match batch dimension: [bs, len(selected_horizons), 4, 4]
    nominal_trajectory_expanded = np.tile(nominal_trajectory_selected[None, :, :, :], (batch_size, 1, 1, 1))  # [bs, len(selected_horizons), 4, 4]
    
    # Reshape to [bs * len(selected_horizons), 4, 4] for vectorized computation
    x_se3_reshaped = x_se3_selected.reshape(batch_size * len(selected_horizons), 4, 4)  # [bs * len(selected_horizons), 4, 4]
    nominal_reshaped = nominal_trajectory_expanded.reshape(batch_size * len(selected_horizons), 4, 4)  # [bs * len(selected_horizons), 4, 4]
    
    # Compute all distances at once (fully vectorized, no for loop)
    se3_distances_flat = compute_se3_distance(x_se3_reshaped, nominal_reshaped)  # [bs * len(selected_horizons)]
    se3_distances_selected = se3_distances_flat.reshape(batch_size, len(selected_horizons))  # [bs, len(selected_horizons)]
    
    # Convert distances to rewards (closer = higher reward)
    distance_scale = 1.0  # Scale factor for distance
    close_to_nominal_reward = np.exp(-se3_distances_selected / distance_scale)  # [bs, len(selected_horizons)]
    
    # Mean over selected_horizons and convert to tensor
    close_to_nominal_reward = torch.from_numpy(close_to_nominal_reward.mean(axis=1)).float().to(device)  # [bs]
    
    return close_to_nominal_reward


def compute_flow_margin_reward(x, task_kwargs, args, device, visibility_reward_type):
    """
    Compute flow margin reward (visibility robustness to flow perturbations).
    
    Args:
        x: [bs, horizon, 9] - viewpoint predictions (position + ortho6d)
        task_kwargs: dict containing query_point, etc.
        args: args object with svdd config
        device: torch device
        visibility_reward_type: 'wo_consider_tracking_lost' or 'w_consider_tracking_lost'
    
    Returns:
        flow_margin_reward: [bs] torch.Tensor - reward for each batch
    """
    batch_size, horizon, D = x.shape
    assert D == 9, f"Expected 9D input (position + ortho6d), got {D}"
    
    # Get parameters for flow margin computation
    flow_margin_num_perturbations = getattr(args.svdd, 'flow_margin_num_perturbations', 5)
    flow_margin_perturbation_scale = getattr(args.svdd, 'flow_margin_perturbation_scale', 0.01)
    flow_margin_variance_penalty_weight = getattr(args.svdd, 'flow_margin_variance_penalty_weight', 0.1)
    
    selected_horizons = task_kwargs.get('selected_horizons', list(range(horizon)))
    
    # Get original query points (flow) from task_kwargs
    query_points_original = task_kwargs.get('query_point')
    predict_separate_flow = task_kwargs.get('predict_separate_flow', False)
    
    # Generate all flow perturbations at once
    if predict_separate_flow and query_points_original.ndim == 3:
        horizon_flow, N_flow, D_flow = query_points_original.shape
        query_points_expanded_flow = np.tile(query_points_original[None, :, :, :], (flow_margin_num_perturbations, 1, 1, 1))  # [num_perturbations, horizon, N, D]
        
        if D_flow == 4:
            query_points_xyz_flow = query_points_expanded_flow[:, :, :, :3]  # [num_perturbations, horizon, N, 3]
        else:
            query_points_xyz_flow = query_points_expanded_flow  # [num_perturbations, horizon, N, 3]
        
        # Perturb flow points: [num_perturbations, horizon, N, 3]
        flow_perturbation = np.random.randn(*query_points_xyz_flow.shape) * flow_margin_perturbation_scale
        query_points_perturbed_flow = query_points_xyz_flow + flow_perturbation  # [num_perturbations, horizon, N, 3]
        
        # Reshape to [num_perturbations, horizon, N, D] for task_kwargs
        if D_flow == 4:
            query_points_perturbed_flow_full = np.concatenate([
                query_points_perturbed_flow,
                query_points_expanded_flow[:, :, :, 3:4]  # Keep original 4th dimension
            ], axis=-1)  # [num_perturbations, horizon, N, 4]
        else:
            query_points_perturbed_flow_full = query_points_perturbed_flow  # [num_perturbations, horizon, N, 3]
    else:
        # Original behavior: query_points: [N, 3] or [N, 4]
        if query_points_original.ndim == 1:
            query_points_original = query_points_original.reshape(1, -1)
        
        N_flow = query_points_original.shape[0]
        D_flow = query_points_original.shape[1]
        
        if D_flow == 4:
            query_points_xyz_flow = query_points_original[:, :3]  # [N, 3]
        else:
            query_points_xyz_flow = query_points_original  # [N, 3]
        
        # Expand to [num_perturbations, N, 3]
        query_points_expanded_flow = np.tile(query_points_xyz_flow[None, :, :], (flow_margin_num_perturbations, 1, 1))  # [num_perturbations, N, 3]
        
        # Perturb flow points: [num_perturbations, N, 3]
        flow_perturbation = np.random.randn(flow_margin_num_perturbations, N_flow, 3) * flow_margin_perturbation_scale
        query_points_perturbed_flow = query_points_expanded_flow + flow_perturbation  # [num_perturbations, N, 3]
        
        # Reshape to [num_perturbations, N, D] for task_kwargs
        if D_flow == 4:
            query_points_perturbed_flow_full = np.concatenate([
                query_points_perturbed_flow,
                np.tile(query_points_original[:, 3:4][None, :, :], (flow_margin_num_perturbations, 1, 1))  # Keep original 4th dimension
            ], axis=-1)  # [num_perturbations, N, 4]
        else:
            query_points_perturbed_flow_full = query_points_perturbed_flow  # [num_perturbations, N, 3]
    
    # Compute visibility for all flow perturbations
    vis_reward_perturbed_all_flow = []
    
    for p in range(flow_margin_num_perturbations):
        # Create modified task_kwargs with perturbed query points
        task_kwargs_perturbed = task_kwargs.copy()
        task_kwargs_perturbed['query_point'] = query_points_perturbed_flow_full[p]  # [horizon, N, D] or [N, D]
        
        # Compute visibility for this perturbation
        vis_reward_perturbed, _, _, _, _, vis_reward_wo_tracking_lost_perturbed, _, _ = compute_visibility(x, task_kwargs_perturbed)
        
        if visibility_reward_type == 'wo_consider_tracking_lost':
            vis_reward_perturbed_all_flow.append(vis_reward_wo_tracking_lost_perturbed)
        elif visibility_reward_type == 'w_consider_tracking_lost':
            vis_reward_perturbed_all_flow.append(vis_reward_perturbed)
        else:
            vis_reward_perturbed_all_flow.append(vis_reward_perturbed)
    
    # Stack to [num_perturbations, bs]
    perturbed_visibility_rewards_flow = np.stack(vis_reward_perturbed_all_flow, axis=0)  # [num_perturbations, bs]
    
    # Hybrid approach for flow margin with [0, 1] normalization
    min_visibility_flow = np.min(perturbed_visibility_rewards_flow, axis=0)  # [bs] - in [0, 1]
    visibility_variance_flow = np.var(perturbed_visibility_rewards_flow, axis=0)  # [bs] - in [0, 0.25]
    
    # Combined flow margin reward: [0, 1] range
    flow_margin_reward = min_visibility_flow * (1.0 - flow_margin_variance_penalty_weight * visibility_variance_flow)  # [bs] - in [0, 1]
    flow_margin_reward = torch.from_numpy(flow_margin_reward).float().to(device)  # [bs]
    
    return flow_margin_reward


def compute_reward(args, x, task_kwargs=None, device=None):
    # # x : [b*dup, horizon, N, D], tensor, 9D representation
    
    reward_type = task_kwargs.get('reward_type') if task_kwargs.get('reward_type', None) is not None else args.svdd.reward_option
    smoothness_scale = 1.0
    visibility_reward_type = task_kwargs.get('visibility_reward_type') if task_kwargs.get('visibility_reward_type', None) is not None else args.svdd.visibility_reward_option
    
    # Compute proprioception penalty (common for all reward types)
    robot_future_action = task_kwargs.get('robot_future_action')
    if robot_future_action is not None:
        # Convert x to numpy for penalty computation
        x_np = x.copy() if isinstance(x, np.ndarray) else x.cpu().numpy()
        proprioception_penalty = torch.from_numpy(compute_proprioception_penalty(x_np, robot_future_action)).float().to(device)
    else:
        # If robot_future_action is not available, set penalty to zero
        batch_size = x.shape[0] if isinstance(x, np.ndarray) else x.shape[0]
        proprioception_penalty = torch.zeros(batch_size, device=device, dtype=torch.float32)
    
    if reward_type == 'visibility':
        visibility_reward, visibility_results, hit_distances, visibility_rewards_among_tracking_points, visibility_results_among_tracking, visibility_rewards_wo_tracking_lost, visibility_results_wo_tracking_lost, visibility_info = compute_visibility(x, task_kwargs)
        smoothness_penalty = torch.from_numpy(compute_smoothness(convert_9d_to_se3(x), scale=smoothness_scale)).float().to(device)
        
        visibility_reward = torch.from_numpy(visibility_reward).float().to(device)
        visibility_rewards_among_tracking_points = torch.from_numpy(visibility_rewards_among_tracking_points).float().to(device) #[b*dup]
        visibility_rewards_wo_tracking_lost = torch.from_numpy(visibility_rewards_wo_tracking_lost).float().to(device) #[b*dup]

        if visibility_reward_type == 'wo_consider_tracking_lost':
            reward = visibility_rewards_wo_tracking_lost
        elif visibility_reward_type == 'w_consider_tracking_lost':
            reward = visibility_reward
        

        reward = reward - smoothness_penalty - proprioception_penalty

        return {
            'total_reward': reward,
            'visibility_reward': visibility_reward,
            'smoothness_reward': smoothness_penalty,
            'proprioception_penalty': proprioception_penalty,
            'visibility_reward_among_tracking_points': visibility_rewards_among_tracking_points,
            'visibility_reward_wo_tracking_lost': visibility_rewards_wo_tracking_lost,
            'visibility_results': visibility_results,
            'hit_distances': hit_distances,
            'visibility_results_among_tracking': visibility_results_among_tracking,
            'visibility_results_wo_tracking_lost': visibility_results_wo_tracking_lost,
            'visibility_info': visibility_info,
        }
        
    elif reward_type == 'visibility+close_to_nominal':
        # Compute visibility reward
        visibility_reward, visibility_results, hit_distances, visibility_rewards_among_tracking_points, visibility_results_among_tracking, visibility_rewards_wo_tracking_lost, visibility_results_wo_tracking_lost, visibility_info = compute_visibility(x, task_kwargs)
        smoothness_penalty = torch.from_numpy(compute_smoothness(convert_9d_to_se3(x), scale=smoothness_scale)).float().to(device)
        
            
        visibility_reward = torch.from_numpy(visibility_reward).float().to(device) #[b*dup]
        visibility_rewards_among_tracking_points = torch.from_numpy(visibility_rewards_among_tracking_points).float().to(device) #[b*dup]
        visibility_rewards_wo_tracking_lost = torch.from_numpy(visibility_rewards_wo_tracking_lost).float().to(device) #[b*dup]
        

        # Compute close_to_nominal reward using function
        close_to_nominal_weight = args.svdd.close_to_nominal_weight
        close_to_nominal_reward = compute_close_to_nominal_reward(x, task_kwargs, args, device)


        if visibility_reward_type == 'wo_consider_tracking_lost':
            reward = visibility_rewards_wo_tracking_lost
        elif visibility_reward_type == 'w_consider_tracking_lost':
            reward = visibility_reward


        # Combine rewards: visibility_reward + close_to_nominal_weight * close_to_nominal_reward
        reward = reward - smoothness_penalty - proprioception_penalty + close_to_nominal_weight * close_to_nominal_reward
        
        return {
            'total_reward': reward,
            'visibility_reward': visibility_reward,
            'smoothness_reward': smoothness_penalty,
            'proprioception_penalty': proprioception_penalty,
            'close_to_nominal_reward': close_to_nominal_reward,
            'visibility_reward_among_tracking_points': visibility_rewards_among_tracking_points,
            'visibility_reward_wo_tracking_lost': visibility_rewards_wo_tracking_lost,
            'visibility_results': visibility_results,
            'hit_distances': hit_distances,
            'visibility_results_among_tracking': visibility_results_among_tracking,
            'visibility_results_wo_tracking_lost': visibility_results_wo_tracking_lost,
            'visibility_info': visibility_info,
        }
    
    # Handle special case: close_to_goal (doesn't use visibility)
    elif reward_type == 'close_to_goal':
        # Compute visibility reward
        visibility_reward, visibility_results, hit_distances, visibility_rewards_among_tracking_points, visibility_results_among_tracking, visibility_rewards_wo_tracking_lost, visibility_results_wo_tracking_lost, visibility_info = compute_visibility(x, task_kwargs)
        smoothness_penalty = torch.from_numpy(compute_smoothness(convert_9d_to_se3(x), scale=smoothness_scale)).float().to(device)
        
            
        visibility_reward = torch.from_numpy(visibility_reward).float().to(device) #[b*dup]
        visibility_rewards_among_tracking_points = torch.from_numpy(visibility_rewards_among_tracking_points).float().to(device) #[b*dup]
        visibility_rewards_wo_tracking_lost = torch.from_numpy(visibility_rewards_wo_tracking_lost).float().to(device) #[b*dup]
        

        # Compute close_to_nominal reward
        batch_size, horizon, D = x.shape
        assert D == 9, f"Expected 9D input (position + ortho6d), got {D}"
        
        # Get nominal trajectory and selected_horizons from task_kwargs
        nominal_trajectory = task_kwargs.get('nominal_trajectory')  # [h, 4, 4] SE(3) in marker coordinate
        
        close_to_nominal_weight = args.svdd.close_to_nominal_weight
        selected_horizons = task_kwargs.get('selected_horizons', list(range(horizon)))  # Default to all horizons if not specified
        
    
        x_np = x.copy()  # [bs, horizon, 9]
        
        # Convert x (viewpoint predictions) to SE(3) matrices
        # x: [bs, horizon, 9] (position + ortho6d)
        # Convert to SE(3) matrices
        x_se3 = convert_9d_to_se3(x_np)  # [bs, horizon, 4, 4]
        
        # Compute SE(3) distances to nominal trajectory (fully vectorized)
        # Select x_se3 for selected_horizons: [bs, len(selected_horizons), 4, 4]
        x_se3_selected = x_se3[:, selected_horizons, :, :]  # [bs, len(selected_horizons), 4, 4]
        
        # Select nominal_trajectory for selected_horizons: [len(selected_horizons), 4, 4]
        nominal_trajectory_selected = nominal_trajectory[selected_horizons, :, :]  # [len(selected_horizons), 4, 4]
        
        # Expand nominal_trajectory_selected to match batch dimension: [bs, len(selected_horizons), 4, 4]
        nominal_trajectory_expanded = np.tile(nominal_trajectory_selected[None, :, :, :], (batch_size, 1, 1, 1))  # [bs, len(selected_horizons), 4, 4]
        
        # Reshape to [bs * len(selected_horizons), 4, 4] for vectorized computation
        x_se3_reshaped = x_se3_selected.reshape(batch_size * len(selected_horizons), 4, 4)  # [bs * len(selected_horizons), 4, 4]
        nominal_reshaped = nominal_trajectory_expanded.reshape(batch_size * len(selected_horizons), 4, 4)  # [bs * len(selected_horizons), 4, 4]
        
        # Compute all distances at once (fully vectorized, no for loop)
        se3_distances_flat = compute_se3_distance(x_se3_reshaped, nominal_reshaped)  # [bs * len(selected_horizons)]
        se3_distances_selected = se3_distances_flat.reshape(batch_size, len(selected_horizons))  # [bs, len(selected_horizons)]
        
        # Convert distances to rewards (closer = higher reward)
        # Use negative exponential: exp(-distance) so that closer distances give higher rewards
        # Normalize by a scale factor to make it comparable to visibility reward
        distance_scale = 1.0  # Scale factor for distance (adjust as needed)
        close_to_nominal_reward = np.exp(-se3_distances_selected / distance_scale)  # [bs, len(selected_horizons)]
        
        # Mean over selected_horizons and convert to tensor
        close_to_nominal_reward = torch.from_numpy(close_to_nominal_reward.mean(axis=1)).float().to(device)  # [bs]
        

        if visibility_reward_type == 'wo_consider_tracking_lost':
            reward = visibility_rewards_wo_tracking_lost
        elif visibility_reward_type == 'w_consider_tracking_lost':
            reward = visibility_reward


        # Combine rewards: visibility_reward + close_to_nominal_weight * close_to_nominal_reward
        reward = reward - smoothness_penalty - proprioception_penalty + close_to_nominal_weight * close_to_nominal_reward
        
        return {
            'total_reward': reward,
            'visibility_reward': visibility_reward,
            'smoothness_reward': smoothness_penalty,
            'proprioception_penalty': proprioception_penalty,
            'close_to_nominal_reward': close_to_nominal_reward,
            'visibility_reward_among_tracking_points': visibility_rewards_among_tracking_points,
            'visibility_reward_wo_tracking_lost': visibility_rewards_wo_tracking_lost,
            'visibility_results': visibility_results,
            'hit_distances': hit_distances,
            'visibility_results_among_tracking': visibility_results_among_tracking,
            'visibility_results_wo_tracking_lost': visibility_results_wo_tracking_lost,
            'visibility_info': visibility_info,
        }

    elif reward_type == 'visibility+close_to_query_points':
        # Compute visibility reward
        visibility_reward, visibility_results, hit_distances, visibility_rewards_among_tracking_points, visibility_results_among_tracking, visibility_rewards_wo_tracking_lost, visibility_results_wo_tracking_lost, visibility_info = compute_visibility(x, task_kwargs)
        smoothness_penalty = torch.from_numpy(compute_smoothness(convert_9d_to_se3(x), scale=smoothness_scale)).float().to(device)
        
        
        visibility_reward = torch.from_numpy(visibility_reward).float().to(device)
        visibility_rewards_among_tracking_points = torch.from_numpy(visibility_rewards_among_tracking_points).float().to(device)
        visibility_rewards_wo_tracking_lost = torch.from_numpy(visibility_rewards_wo_tracking_lost).float().to(device)
        

        # Compute close_to_query_points reward using function
        close_to_query_points_weight = args.svdd.close_to_query_points_weight
        close_to_query_points_reward = compute_close_to_query_points_reward(x, task_kwargs, args, device)
        

        if visibility_reward_type == 'wo_consider_tracking_lost':
            reward = visibility_rewards_wo_tracking_lost
        elif visibility_reward_type == 'w_consider_tracking_lost':
            reward = visibility_reward

        # Combine rewards: visibility_reward + close_to_query_points_weight * close_to_query_points_reward
        reward = reward - smoothness_penalty - proprioception_penalty + close_to_query_points_weight * close_to_query_points_reward
        
        return {
            'total_reward': reward,
            'visibility_reward': visibility_reward,
            'smoothness_reward': smoothness_penalty,
            'proprioception_penalty': proprioception_penalty,
            'close_to_query_points_reward': close_to_query_points_reward,
            'visibility_reward_among_tracking_points': visibility_rewards_among_tracking_points,
            'visibility_reward_wo_tracking_lost': visibility_rewards_wo_tracking_lost,
            'visibility_results': visibility_results,
            'hit_distances': hit_distances,
            'visibility_results_among_tracking': visibility_results_among_tracking,
            'visibility_results_wo_tracking_lost': visibility_results_wo_tracking_lost,
            'visibility_info': visibility_info,
        }

    elif reward_type == 'visibility+far_from_query_points':
        # Compute visibility reward
        visibility_reward, visibility_results, hit_distances, visibility_rewards_among_tracking_points, visibility_results_among_tracking, visibility_rewards_wo_tracking_lost, visibility_results_wo_tracking_lost, visibility_info = compute_visibility(x, task_kwargs)
        smoothness_penalty = torch.from_numpy(compute_smoothness(convert_9d_to_se3(x), scale=smoothness_scale)).float().to(device)
        
        
        visibility_reward = torch.from_numpy(visibility_reward).float().to(device)
        visibility_rewards_among_tracking_points = torch.from_numpy(visibility_rewards_among_tracking_points).float().to(device)
        visibility_rewards_wo_tracking_lost = torch.from_numpy(visibility_rewards_wo_tracking_lost).float().to(device)

        # Compute far_from_query_points reward using function
        far_from_query_points_weight = getattr(args.svdd, 'far_from_query_points_weight', 0.5)  # Default to 0.5 if not set
        far_from_query_points_reward = compute_far_from_query_points_reward(x, task_kwargs, args, device)


        if visibility_reward_type == 'wo_consider_tracking_lost':
            reward = visibility_rewards_wo_tracking_lost
        elif visibility_reward_type == 'w_consider_tracking_lost':
            reward = visibility_reward

        # Combine rewards: visibility_reward + far_from_query_points_weight * far_from_query_points_reward
        reward = reward - smoothness_penalty - proprioception_penalty + far_from_query_points_weight * far_from_query_points_reward
        
        return {
            'total_reward': reward,
            'visibility_reward': visibility_reward,
            'smoothness_reward': smoothness_penalty,
            'proprioception_penalty': proprioception_penalty,
            'far_from_query_points_reward': far_from_query_points_reward,
            'visibility_reward_among_tracking_points': visibility_rewards_among_tracking_points,
            'visibility_reward_wo_tracking_lost': visibility_rewards_wo_tracking_lost,
            'visibility_results': visibility_results,
            'hit_distances': hit_distances,
            'visibility_results_among_tracking': visibility_results_among_tracking,
            'visibility_results_wo_tracking_lost': visibility_results_wo_tracking_lost,
            'visibility_info': visibility_info,
        }

    elif reward_type == 'visibility+camera_margin':
        # Compute visibility reward for original viewpoints
        visibility_reward, visibility_results, hit_distances, visibility_rewards_among_tracking_points, visibility_results_among_tracking, visibility_rewards_wo_tracking_lost, visibility_results_wo_tracking_lost, visibility_info = compute_visibility(x, task_kwargs)
        smoothness_penalty = torch.from_numpy(compute_smoothness(convert_9d_to_se3(x), scale=smoothness_scale)).float().to(device)
        
        visibility_reward = torch.from_numpy(visibility_reward).float().to(device)
        visibility_rewards_among_tracking_points = torch.from_numpy(visibility_rewards_among_tracking_points).float().to(device)
        visibility_rewards_wo_tracking_lost = torch.from_numpy(visibility_rewards_wo_tracking_lost).float().to(device)

        # Compute camera margin reward using function
        camera_margin_weight = getattr(args.svdd, 'camera_margin_weight', 0.1)  # Default to 0.1 if not set
        camera_margin_reward = compute_camera_margin_reward(x, task_kwargs, args, device, visibility_reward_type)

        if visibility_reward_type == 'wo_consider_tracking_lost':
            reward = visibility_rewards_wo_tracking_lost
        elif visibility_reward_type == 'w_consider_tracking_lost':
            reward = visibility_reward

        # Combine rewards: visibility_reward + camera_margin_weight * camera_margin_reward
        # The camera margin reward encourages viewpoints that maintain good visibility even with small camera perturbations
        reward = reward - smoothness_penalty - proprioception_penalty + camera_margin_weight * camera_margin_reward
        
        return {
            'total_reward': reward,
            'visibility_reward': visibility_reward,
            'smoothness_reward': smoothness_penalty,
            'proprioception_penalty': proprioception_penalty,
            'camera_margin_reward': camera_margin_reward,
            'visibility_reward_among_tracking_points': visibility_rewards_among_tracking_points,
            'visibility_reward_wo_tracking_lost': visibility_rewards_wo_tracking_lost,
            'visibility_results': visibility_results,
            'hit_distances': hit_distances,
            'visibility_results_among_tracking': visibility_results_among_tracking,
            'visibility_results_wo_tracking_lost': visibility_results_wo_tracking_lost,
            'visibility_info': visibility_info,
        }

    elif reward_type == 'visibility+camera_margin+flow_margin':
        # Compute visibility reward for original viewpoints and flows
        visibility_reward, visibility_results, hit_distances, visibility_rewards_among_tracking_points, visibility_results_among_tracking, visibility_rewards_wo_tracking_lost, visibility_results_wo_tracking_lost, visibility_info = compute_visibility(x, task_kwargs)
        smoothness_penalty = torch.from_numpy(compute_smoothness(convert_9d_to_se3(x), scale=smoothness_scale)).float().to(device)
        
        visibility_reward = torch.from_numpy(visibility_reward).float().to(device)
        visibility_rewards_among_tracking_points = torch.from_numpy(visibility_rewards_among_tracking_points).float().to(device)
        visibility_rewards_wo_tracking_lost = torch.from_numpy(visibility_rewards_wo_tracking_lost).float().to(device)

        # Compute camera margin reward using function
        camera_margin_weight = getattr(args.svdd, 'camera_margin_weight', 0.5)  # Default to 0.5 if not set
        camera_margin_reward = compute_camera_margin_reward(x, task_kwargs, args, device, visibility_reward_type)
        
        # Compute flow margin reward using function
        flow_margin_weight = getattr(args.svdd, 'flow_margin_weight', 0.5)  # Default to 0.5 if not set
        flow_margin_reward = compute_flow_margin_reward(x, task_kwargs, args, device, visibility_reward_type)

        if visibility_reward_type == 'wo_consider_tracking_lost':
            reward = visibility_rewards_wo_tracking_lost
        elif visibility_reward_type == 'w_consider_tracking_lost':
            reward = visibility_reward

        # Combine rewards: visibility_reward + camera_margin_weight * camera_margin_reward + flow_margin_weight * flow_margin_reward
        # Both margins encourage robustness to perturbations in camera viewpoints and flow predictions
        reward = reward - smoothness_penalty - proprioception_penalty + camera_margin_weight * camera_margin_reward + flow_margin_weight * flow_margin_reward
        
        return {
            'total_reward': reward,
            'visibility_reward': visibility_reward,
            'smoothness_reward': smoothness_penalty,
            'proprioception_penalty': proprioception_penalty,
            'camera_margin_reward': camera_margin_reward,
            'flow_margin_reward': flow_margin_reward,
            'visibility_reward_among_tracking_points': visibility_rewards_among_tracking_points,
            'visibility_reward_wo_tracking_lost': visibility_rewards_wo_tracking_lost,
            'visibility_results': visibility_results,
            'hit_distances': hit_distances,
            'visibility_results_among_tracking': visibility_results_among_tracking,
            'visibility_results_wo_tracking_lost': visibility_results_wo_tracking_lost,
            'visibility_info': visibility_info,
        }

    elif reward_type == 'visibility+camera_center':
        # Compute visibility reward
        visibility_reward, visibility_results, hit_distances, visibility_rewards_among_tracking_points, visibility_results_among_tracking, visibility_rewards_wo_tracking_lost, visibility_results_wo_tracking_lost, visibility_info = compute_visibility(x, task_kwargs)
        smoothness_penalty = torch.from_numpy(compute_smoothness(convert_9d_to_se3(x), scale=smoothness_scale)).float().to(device)
        
        visibility_reward = torch.from_numpy(visibility_reward).float().to(device)
        visibility_rewards_among_tracking_points = torch.from_numpy(visibility_rewards_among_tracking_points).float().to(device)
        visibility_rewards_wo_tracking_lost = torch.from_numpy(visibility_rewards_wo_tracking_lost).float().to(device)

        # Compute camera_center reward using function
        camera_center_weight = getattr(args.svdd, 'camera_center_weight', 0.5)  # Default to 0.5 if not set
        camera_center_reward = compute_camera_center_reward(x, task_kwargs, args, device, visibility_results)
        
        if visibility_reward_type == 'wo_consider_tracking_lost':
            reward = visibility_rewards_wo_tracking_lost
        elif visibility_reward_type == 'w_consider_tracking_lost':
            reward = visibility_reward
        
        # Combine rewards: visibility_reward + camera_center_weight * camera_center_reward
        reward = reward - smoothness_penalty - proprioception_penalty + camera_center_weight * camera_center_reward
        
        return {
            'total_reward': reward,
            'visibility_reward': visibility_reward,
            'smoothness_reward': smoothness_penalty,
            'proprioception_penalty': proprioception_penalty,
            'camera_center_reward': camera_center_reward,
            'visibility_reward_among_tracking_points': visibility_rewards_among_tracking_points,
            'visibility_reward_wo_tracking_lost': visibility_rewards_wo_tracking_lost,
            'visibility_results': visibility_results,
            'hit_distances': hit_distances,
            'visibility_results_among_tracking': visibility_results_among_tracking,
            'visibility_results_wo_tracking_lost': visibility_results_wo_tracking_lost,
            'visibility_info': visibility_info,
        }

    elif reward_type == 'visibility+camera_center+close_to_query_points':
        # Compute visibility reward
        visibility_reward, visibility_results, hit_distances, visibility_rewards_among_tracking_points, visibility_results_among_tracking, visibility_rewards_wo_tracking_lost, visibility_results_wo_tracking_lost, visibility_info = compute_visibility(x, task_kwargs)
        smoothness_penalty = torch.from_numpy(compute_smoothness(convert_9d_to_se3(x), scale=smoothness_scale)).float().to(device)
        
        visibility_reward = torch.from_numpy(visibility_reward).float().to(device)
        visibility_rewards_among_tracking_points = torch.from_numpy(visibility_rewards_among_tracking_points).float().to(device)
        visibility_rewards_wo_tracking_lost = torch.from_numpy(visibility_rewards_wo_tracking_lost).float().to(device)

        # Compute camera_center reward using function
        camera_center_weight = getattr(args.svdd, 'camera_center_weight', 0.5)  # Default to 0.5 if not set
        camera_center_reward = compute_camera_center_reward(x, task_kwargs, args, device, visibility_results)
        
        # Compute close_to_query_points reward using function
        close_to_query_points_weight = args.svdd.close_to_query_points_weight
        close_to_query_points_reward = compute_close_to_query_points_reward(x, task_kwargs, args, device)
        
        if visibility_reward_type == 'wo_consider_tracking_lost':
            reward = visibility_rewards_wo_tracking_lost
        elif visibility_reward_type == 'w_consider_tracking_lost':
            reward = visibility_reward
        
        # Combine rewards: visibility_reward + camera_center_weight * camera_center_reward + close_to_query_points_weight * close_to_query_points_reward
        reward = reward - smoothness_penalty - proprioception_penalty + camera_center_weight * camera_center_reward + close_to_query_points_weight * close_to_query_points_reward
        
        return {
            'total_reward': reward,
            'visibility_reward': visibility_reward,
            'smoothness_reward': smoothness_penalty,
            'proprioception_penalty': proprioception_penalty,
            'camera_center_reward': camera_center_reward,
            'close_to_query_points_reward': close_to_query_points_reward,
            'visibility_reward_among_tracking_points': visibility_rewards_among_tracking_points,
            'visibility_reward_wo_tracking_lost': visibility_rewards_wo_tracking_lost,
            'visibility_results': visibility_results,
            'hit_distances': hit_distances,
            'visibility_results_among_tracking': visibility_results_among_tracking,
            'visibility_results_wo_tracking_lost': visibility_results_wo_tracking_lost,
            'visibility_info': visibility_info,
        }

    elif reward_type == 'visibility+camera_center+camera_margin+close_to_query_points':
        # Compute visibility reward
        visibility_reward, visibility_results, hit_distances, visibility_rewards_among_tracking_points, visibility_results_among_tracking, visibility_rewards_wo_tracking_lost, visibility_results_wo_tracking_lost, visibility_info = compute_visibility(x, task_kwargs)
        smoothness_penalty = torch.from_numpy(compute_smoothness(convert_9d_to_se3(x), scale=smoothness_scale)).float().to(device)
        
        visibility_reward = torch.from_numpy(visibility_reward).float().to(device)
        visibility_rewards_among_tracking_points = torch.from_numpy(visibility_rewards_among_tracking_points).float().to(device)
        visibility_rewards_wo_tracking_lost = torch.from_numpy(visibility_rewards_wo_tracking_lost).float().to(device)

        # Compute camera_center reward using function
        camera_center_weight = getattr(args.svdd, 'camera_center_weight', 0.5)  # Default to 0.5 if not set
        camera_center_reward = compute_camera_center_reward(x, task_kwargs, args, device, visibility_results)
        
        # Compute camera margin reward using function
        camera_margin_weight = getattr(args.svdd, 'camera_margin_weight', 0.1)  # Default to 0.1 if not set
        camera_margin_reward = compute_camera_margin_reward(x, task_kwargs, args, device, visibility_reward_type)
        
        # Compute close_to_query_points reward using function
        close_to_query_points_weight = args.svdd.close_to_query_points_weight
        close_to_query_points_reward = compute_close_to_query_points_reward(x, task_kwargs, args, device)
        
        if visibility_reward_type == 'wo_consider_tracking_lost':
            reward = visibility_rewards_wo_tracking_lost
        elif visibility_reward_type == 'w_consider_tracking_lost':
            reward = visibility_reward
        
        # Combine rewards: visibility_reward + camera_center_weight * camera_center_reward + camera_margin_weight * camera_margin_reward + close_to_query_points_weight * close_to_query_points_reward
        if getattr(args.svdd, 'reward_ablation', False):
            reward = reward - smoothness_penalty
        else:
            reward = reward - smoothness_penalty - proprioception_penalty + camera_center_weight * camera_center_reward + camera_margin_weight * camera_margin_reward + close_to_query_points_weight * close_to_query_points_reward
        
        return {
            'total_reward': reward,
            'visibility_reward': visibility_reward,
            'smoothness_reward': smoothness_penalty,
            'proprioception_penalty': proprioception_penalty,
            'camera_center_reward': camera_center_reward,
            'camera_margin_reward': camera_margin_reward,
            'close_to_query_points_reward': close_to_query_points_reward,
            'visibility_reward_among_tracking_points': visibility_rewards_among_tracking_points,
            'visibility_reward_wo_tracking_lost': visibility_rewards_wo_tracking_lost,
            'visibility_results': visibility_results,
            'hit_distances': hit_distances,
            'visibility_results_among_tracking': visibility_results_among_tracking,
            'visibility_results_wo_tracking_lost': visibility_results_wo_tracking_lost,
            'visibility_info': visibility_info,
        }

    elif reward_type == 'close_to_goal':
        # Compute close_to_goal reward
        batch_size, horizon, D = x.shape
        assert D == 9, f"Expected 9D input (position + ortho6d), got {D}"
        
        # Get current_viewpoint from task_kwargs (unnormalized, marker coordinate)
        current_viewpoint = task_kwargs.get('current_viewpoint')  # [9] - unnormalized, marker coordinate
        if current_viewpoint is None:
            raise ValueError("current_viewpoint must be provided in task_kwargs for 'close_to_goal' reward_type")
        
        # Get T_mc_transformation from task_kwargs to compute camera axis direction
        T_mc_transformation = task_kwargs.get('T_mc_transformation')  # [4, 4] - marker to camera transformation
        if T_mc_transformation is None:
            raise ValueError("T_mc_transformation must be provided in task_kwargs for 'close_to_goal' reward_type")
        
        # Goal position: current_viewpoint position + offset along camera axis
        # T_mc_transformation is marker to camera transformation
        camera_z_axis_in_marker = T_mc_transformation[:3, 2]  # [3] - camera z-axis direction in marker coordinate
        goal_position = current_viewpoint[:3].copy() + camera_z_axis_in_marker * (-0.3)  # [3] - z-axis -0.3m
        
        # Get selected_horizons from task_kwargs
        selected_horizons = task_kwargs.get('selected_horizons', list(range(horizon)))  # Default to all horizons if not specified
        
        # Convert x to numpy if it's a tensor
        x_np = x.copy()  # [bs, horizon, 9]
        
        # Extract camera center positions from x (viewpoint predictions)
        # x: [bs, horizon, 9] (position + ortho6d)
        # First 3 dimensions are translation (camera center position in marker coordinate)
        camera_centers = x_np[:, :, :3]  # [bs, horizon, 3] - camera center positions in marker coordinate
        
        # Select camera centers for selected_horizons: [bs, len(selected_horizons), 3]
        camera_centers_selected = camera_centers[:, selected_horizons, :]  # [bs, len(selected_horizons), 3]
        
        # Expand goal_position to match batch and horizon dimensions: [bs, len(selected_horizons), 3]
        goal_position_expanded = np.tile(goal_position[None, None, :], (batch_size, len(selected_horizons), 1))  # [bs, len(selected_horizons), 3]
        
        # Compute Euclidean distances to goal: [bs, len(selected_horizons)]
        distances = np.linalg.norm(camera_centers_selected - goal_position_expanded, axis=-1)  # [bs, len(selected_horizons)]
        
        # Convert distances to rewards (closer = higher reward)
        # Use negative exponential: exp(-distance) so that closer distances give higher rewards
        # Normalize by a scale factor
        distance_scale = 1.0  # Scale factor for distance (adjust as needed)
        close_to_goal_reward_selected = np.exp(-distances / distance_scale)  # [bs, len(selected_horizons)]
        
        # Mean over selected_horizons and convert to tensor
        close_to_goal_reward = torch.from_numpy(close_to_goal_reward_selected.mean(axis=1)).float().to(device)  # [bs]
        
        # Optionally add smoothness penalty
        reward = close_to_goal_reward - proprioception_penalty # - smoothness_penalty
        
        return {
            'total_reward': reward,
            'close_to_goal_reward': close_to_goal_reward,
            'proprioception_penalty': proprioception_penalty,
            # 'smoothness_reward': smoothness_penalty,
        }

    else:
        raise NotImplementedError(f"reward_type '{reward_type}' not implemented")
    

def compute_visibility(viewpoint_pred, task_kwargs):
    """
    Compute visibility reward for active vision task (optimized version using precomputed scenes)
    
    Args:
        viewpoint_pred: [bs, horizon, D] tensor where D=9 (position + ortho6d)
        task_kwargs: dict containing necessary data for visibility computation
            - query_point: [N, 3] or [N, 4] query points in marker coordinates
                         If [N, 4], the 4th dimension is cotracker visibility (1.0 = visible, 0.0 = invisible)
            - precomputed_scenes: list of precomputed scenes for each horizon step
            
    
    Returns:
        rewards: [bs] array of visibility rewards (mean of visible query points across all horizons)
        visibility_results: [bs, horizon, N] array of visibility results for all query points
        hit_distances: [bs, horizon, N] array of hit distances for all query points
        rewards_among_tracking_points: [bs] array of visibility rewards computed only for tracking points (where cotracker_visible == 1)
        visibility_results_among_tracking: [bs, horizon, N] array of visibility results for tracking points only
    """
    timing_enabled = task_kwargs.get('print_visibility_timing', True)
    timing_total_start = time.time()
    timing_sections = {}

    def add_timing(name, start_time):
        timing_sections[name] = timing_sections.get(name, 0.0) + (time.time() - start_time)

    section_start = time.time()
    batch_size, horizon, D = viewpoint_pred.shape
    assert horizon == task_kwargs['robot_future_action'].shape[0], 'horizon should be the same'
    assert D == 9, 'Expected 9D input (position + ortho6d)'
    
    # Extract necessary data from task_kwargs
    query_points = task_kwargs['query_point']  # [N, 3] or [N, 4] or [horizon, N, 3] or [horizon, N, 4] query points in marker coordinates
    predict_separate_flow = task_kwargs.get('predict_separate_flow', False)  # Flag to indicate if using predicted flow as query points
    
    # Get query_point_indices if available (None means use all query points)
    query_point_indices = task_kwargs.get('query_point_indices', None)  # [M] - indices of query points to use, or None
    
    # Use pre-computed combined scenes from task_kwargs
    precomputed_scenes = task_kwargs.get('precomputed_scenes')
    
    selected_horizons = task_kwargs['selected_horizons']
    
    
    T_mw = task_kwargs.get('T_mw')  # [4, 4]
    T_wm = np.linalg.inv(T_mw)
    add_timing('setup/extract_inputs', section_start)

    section_start = time.time()
    viewpoint_pred_se3_marker = convert_9d_to_se3(viewpoint_pred)  # [bs, horizon, 4, 4] in marker coordinate
    viewpoint_pred_se3_world = np.einsum('ij,bhjk->bhik', T_wm, viewpoint_pred_se3_marker)  # [bs, horizon, 4, 4] in world coordinate
    # Convert back to 9D format
    viewpoint_pred = convert_se3_to_9d(viewpoint_pred_se3_world)  # [bs, horizon, 9] in world coordinate
    add_timing('viewpoint_marker_to_world', section_start)
    
    section_start = time.time()
    # Get mask_filtered_point_indices if available (points filtered by use_mask_depth_filter or use_mask_filter)
    # mask_filtered_point_indices: [K] - indices of filtered points in original query points, or None
    mask_filtered_point_indices_original = task_kwargs.get('mask_filtered_point_indices')  # [K] - indices of filtered points in original query points, or None
    
    # Handle different query point shapes based on predict_separate_flow flag
    if predict_separate_flow and query_points.ndim == 3:
        # query_points: [horizon, N, 3] or [horizon, N, 4] - predicted flow for each horizon
        # Extract xyz coordinates and cotracker visibility for each horizon
        if query_points.shape[2] == 4:
            # [horizon, N, 4] - includes visibility from cotracker
            query_points_xyz_per_horizon = query_points[:, :, :3]  # [horizon, N, 3]
            cotracker_visibility_per_horizon = query_points[:, :, 3]  # [horizon, N] - 1.0 = visible, 0.0 = invisible
            cotracker_visible_per_horizon = cotracker_visibility_per_horizon > 0  # [horizon, N] - boolean array
        else:
            # [horizon, N, 3] - no visibility, assume all visible
            query_points_xyz_per_horizon = query_points  # [horizon, N, 3]
            cotracker_visible_per_horizon = np.ones(query_points.shape[:2], dtype=bool)  # [horizon, N] - all True
        
        # Filter query points by query_point_indices if specified
        if query_point_indices is not None:
            query_points_xyz_per_horizon = query_points_xyz_per_horizon[:, query_point_indices, :]  # [horizon, M, 3]
            cotracker_visible_per_horizon = cotracker_visible_per_horizon[:, query_point_indices]  # [horizon, M]
        
        N = query_points_xyz_per_horizon.shape[1]  # Number of query points (after filtering by query_point_indices)
        
        # Remap mask_filtered_point_indices from original query points to filtered query points
        # mask_filtered_point_indices_original: [K] - indices in original query points
        # query_point_indices: [M] - indices in original query points that are selected
        # mask_filtered_point_indices: [L] - indices in filtered query points (where L <= K)
        mask_filtered_point_indices = None
        if mask_filtered_point_indices_original is not None and len(mask_filtered_point_indices_original) > 0:
            if query_point_indices is not None:
                # Create a mapping from original indices to filtered indices
                # query_point_indices[i] is the original index, i is the filtered index
                original_to_filtered = {orig_idx: filtered_idx for filtered_idx, orig_idx in enumerate(query_point_indices)}
                # Find which mask_filtered_point_indices are in query_point_indices
                mask_filtered_point_indices = []
                for orig_idx in mask_filtered_point_indices_original:
                    if orig_idx in original_to_filtered:
                        mask_filtered_point_indices.append(original_to_filtered[orig_idx])
                mask_filtered_point_indices = np.array(mask_filtered_point_indices) if len(mask_filtered_point_indices) > 0 else None
            else:
                # No query_point_indices filtering, so mask_filtered_point_indices_original can be used directly
                mask_filtered_point_indices = np.array(mask_filtered_point_indices_original)
    else:
        # Original behavior: query_points: [N, 3] or [N, 4]
        # Ensure query_points is 2D array [N, 3] or [N, 4]
        if query_points.ndim == 1:
            query_points = query_points.reshape(1, -1)
        
        # Extract xyz coordinates and cotracker visibility
        if query_points.shape[1] == 4:
            # [N, 4] - includes visibility from cotracker
            query_points_xyz = query_points[:, :3]  # [N, 3]
            cotracker_visibility = query_points[:, 3]  # [N] - 1.0 = visible, 0.0 = invisible
            # Convert to boolean: > 0 means visible
            cotracker_visible = cotracker_visibility > 0  # [N] - boolean array
        else:
            # [N, 3] - no visibility, assume all visible
            query_points_xyz = query_points  # [N, 3]
            cotracker_visible = np.ones(query_points.shape[0], dtype=bool)  # [N] - all True
        
        # Filter query points by query_point_indices if specified
        if query_point_indices is not None:
            query_points_xyz = query_points_xyz[query_point_indices, :]  # [M, 3]
            cotracker_visible = cotracker_visible[query_point_indices]  # [M]
        
        N = query_points_xyz.shape[0]  # Number of query points (after filtering by query_point_indices)
        query_points_xyz_per_horizon = None  # Will use same query points for all horizons
        cotracker_visible_per_horizon = None
        
        # Remap mask_filtered_point_indices from original query points to filtered query points
        # mask_filtered_point_indices_original: [K] - indices in original query points
        # query_point_indices: [M] - indices in original query points that are selected
        # mask_filtered_point_indices: [L] - indices in filtered query points (where L <= K)
        mask_filtered_point_indices = None
        if mask_filtered_point_indices_original is not None and len(mask_filtered_point_indices_original) > 0:
            if query_point_indices is not None:
                # Create a mapping from original indices to filtered indices
                # query_point_indices[i] is the original index, i is the filtered index
                original_to_filtered = {orig_idx: filtered_idx for filtered_idx, orig_idx in enumerate(query_point_indices)}
                # Find which mask_filtered_point_indices are in query_point_indices
                mask_filtered_point_indices = []
                for orig_idx in mask_filtered_point_indices_original:
                    if orig_idx in original_to_filtered:
                        mask_filtered_point_indices.append(original_to_filtered[orig_idx])
                mask_filtered_point_indices = np.array(mask_filtered_point_indices) if len(mask_filtered_point_indices) > 0 else None
            else:
                # No query_point_indices filtering, so mask_filtered_point_indices_original can be used directly
                mask_filtered_point_indices = np.array(mask_filtered_point_indices_original)
    add_timing('query_point_prepare', section_start)
    
    
    section_start = time.time()
    if predict_separate_flow and query_points_xyz_per_horizon is not None:
        # query_points_xyz_per_horizon: [horizon, N, 3] in marker coordinate
        query_points_homo = np.concatenate([query_points_xyz_per_horizon, np.ones((*query_points_xyz_per_horizon.shape[:2], 1))], axis=-1)  # [horizon, N, 4]
        query_points_world_homo = np.einsum('ij,hnj->hni', T_wm, query_points_homo)  # [horizon, N, 4]
        query_points_xyz_per_horizon = query_points_world_homo[:, :, :3]  # [horizon, N, 3] in world coordinate
    elif query_points_xyz is not None:
        # query_points_xyz: [N, 3] in marker coordinate
        query_points_homo = np.hstack([query_points_xyz, np.ones((query_points_xyz.shape[0], 1))])  # [N, 4]
        query_points_world_homo = (T_wm @ query_points_homo.T).T  # [N, 4]
        query_points_xyz = query_points_world_homo[:, :3]  # [N, 3] in world coordinate
    add_timing('query_marker_to_world', section_start)

    
    section_start = time.time()
    # store visibility results with shape [b(M), h(L), N]
    visibility_results = np.zeros((batch_size, horizon, N), dtype=bool)  # True if visible, False if occluded
    hit_distances = np.zeros((batch_size, horizon, N), dtype=np.float64)  # Hit distances for each viewpoint
    
    # store visibility info with shape [b(M), h(L), N] (batch_visible, frustum_visible)
    batch_visible_info = np.zeros((batch_size, horizon, N), dtype=bool)  # [bs, horizon, N] - raycasting visibility
    frustum_visible_info = np.zeros((batch_size, horizon, N), dtype=bool)  # [bs, horizon, N] - frustum visibility
    
    # store visibility results with shape [b(M), h(L), N_tracking] (tracking points only)
    visibility_results_among_tracking = np.zeros((batch_size, horizon, N), dtype=bool)  # True if visible, False if occluded
    hit_distances_among_tracking = np.zeros((batch_size, horizon, N), dtype=np.float64)  # Hit distances for each viewpoint
    
    # store visibility results with shape [b(M), h(L), N] (ignoring tracking loss)
    visibility_results_wo_tracking_lost = np.zeros((batch_size, horizon, N), dtype=bool)  # True if visible, False if occluded
    
    # For computing rewards_among_tracking_points
    rewards_among_tracking_list = []
    add_timing('allocate_outputs', section_start)
    
    # NOTE: currently, we assume that visibility is maintained during future horizon (we should consider future point flow later)
    for h in selected_horizons:
        section_start = time.time()
        current_scene = precomputed_scenes[h]  # Use pre-computed scene (world coordinate)
        
        viewpoints_position_for_this_robot = viewpoint_pred[:, h, :3].copy() # [b(M), 3] in world coordinate

        # build rays for batch raycasting - all N x M combinations
        origins = viewpoints_position_for_this_robot  # [M, 3] in world coordinate
        
        # create a ray for every (query point, viewpoint) pair
        # origins: [M, 3] -> [M, 1, 3] -> [M, N, 3]
        origins_expanded = origins[:, np.newaxis, :].repeat(N, axis=1)  # [M, N, 3]
        
        # Get query points for this horizon
        if predict_separate_flow and query_points_xyz_per_horizon is not None:
            # Use horizon-specific query points: [horizon, N, 3] -> [N, 3] for this horizon
            query_points_xyz_for_h = query_points_xyz_per_horizon[h]  # [N, 3] in world coordinate
            cotracker_visible_for_h = cotracker_visible_per_horizon[h]  # [N]
        else:
            # Use same query points for all horizons
            query_points_xyz_for_h = query_points_xyz  # [N, 3] in world coordinate
            cotracker_visible_for_h = cotracker_visible  # [N]
        
        # query_points_xyz_for_h: [N, 3] -> [1, N, 3] -> [M, N, 3]
        query_points_expanded = query_points_xyz_for_h[np.newaxis, :, :].repeat(batch_size, axis=0)  # [M, N, 3] in world coordinate
        
        # directions: [M, N, 3] - direction from each viewpoint to each query point
        directions = query_points_expanded - origins_expanded  # [M, N, 3]
        distances = np.linalg.norm(directions, axis=2)  # [M, N] - length of each ray
        directions = directions / distances[:, :, np.newaxis]  # normalized direction vectors [M, N, 3]
        
        # Flatten for batch raycasting: [M*N, 3]
        origins_flat = origins_expanded.reshape(-1, 3)  # [M*N, 3]
        directions_flat = directions.reshape(-1, 3)  # [M*N, 3]
        distances_flat = distances.reshape(-1)  # [M*N]
        add_timing('per_horizon/ray_prepare', section_start)
        
        # batch raycasting against the precomputed scene
        section_start = time.time()
        batch_visible, batch_hit_distances = batch_raycasting_with_scene(
            current_scene, origins_flat, directions_flat, distances_flat - OFFSET_DISTANCE
        )
        add_timing('per_horizon/raycasting', section_start)

        
        # frustum check
        # check that camera information is available in task_kwargs
        section_start = time.time()
    
        camera_intrinsics = task_kwargs['camera_intrinsics']
        image_size = task_kwargs['image_size']
        
        # camera rotations (euler angles extracted from viewpoint_pred)
        
        viewpoints_rotation_ortho6d = viewpoint_pred[:, h, 3:9].copy() # [M, 6] - ortho6d
        
        
        # flatten query points (same order as origins_flat)
        query_points_flat = query_points_expanded.reshape(-1, 3)  # [M*N, 3]
        
        # batch frustum check
        frustum_visible = is_point_in_camera_frustum_batch(
            query_points_flat,  # [M*N, 3]
            viewpoints_position_for_this_robot,  # [M, 3]
            viewpoints_rotation_ortho6d,  # [M, 6]
            camera_intrinsics,
            image_size,
            pixel_bias=3,
        )  # [M*N]
        add_timing('per_horizon/frustum_check', section_start)
        
        # CoTracker visibility check (3rd condition)
        # cotracker_visible_for_h: [N] -> expand to [M*N] to match batch_visible and frustum_visible
        section_start = time.time()
        cotracker_visible_expanded = np.tile(cotracker_visible_for_h, batch_size)  # [M*N]
        
        # Combine all three conditions: ray casting, frustum, and cotracker visibility
        # All three must be satisfied for a point to be visible
        final_visible = batch_visible & frustum_visible & cotracker_visible_expanded  # [M*N]
        
        # Visible without considering tracking lost (only raycasting & frustum, no cotracker)
        visible_wo_consider_tracking_lost = batch_visible & frustum_visible  # [M*N]

        # reshape results to [M, N]
        final_visible = final_visible.reshape(batch_size, N)  # [M, N]
        visible_wo_consider_tracking_lost = visible_wo_consider_tracking_lost.reshape(batch_size, N)  # [M, N]
        batch_hit_distances = batch_hit_distances.reshape(batch_size, N)  # [M, N]
        batch_visible_reshaped = batch_visible.reshape(batch_size, N)  # [M, N]
        frustum_visible_reshaped = frustum_visible.reshape(batch_size, N)  # [M, N]
        
        # Exclude mask-filtered points from visibility_results_wo_tracking_lost if mask_filtered_point_indices is provided
        if mask_filtered_point_indices is not None and len(mask_filtered_point_indices) > 0:
            # Set visibility to False for mask-filtered points (across all batches and this horizon)
            # mask_filtered_point_indices: [M] - indices of points filtered by mask
            # visible_wo_consider_tracking_lost: [M, N] - visibility results for all batches and points
            # Set all batches' visibility to False for filtered point indices
            visible_wo_consider_tracking_lost[:, mask_filtered_point_indices] = False
        
        # store results
        visibility_results[:, h, :] = final_visible
        visibility_results_wo_tracking_lost[:, h, :] = visible_wo_consider_tracking_lost
        hit_distances[:, h, :] = batch_hit_distances
        batch_visible_info[:, h, :] = batch_visible_reshaped  # [M, N]
        frustum_visible_info[:, h, :] = frustum_visible_reshaped  # [M, N]
        add_timing('per_horizon/combine_store', section_start)
        

        section_start = time.time()
        # ========== Additional computation for tracking points only ==========
        # Select only points where cotracker_visible_for_h == 1
        # Use existing computation results instead of recomputing
        tracking_indices = np.where(cotracker_visible_for_h == 1)[0]  # [N_tracking] - indices where cotracker_visible == 1
        N_tracking = len(tracking_indices)
        
        if N_tracking > 0:
            # Extract visibility results for tracking points only from existing computation
            # final_visible already includes raycasting & frustum & cotracker checks
            # For tracking points, we want only raycasting & frustum (without cotracker condition)
            # So we need to compute: batch_visible & frustum_visible (without cotracker)
            
            # For tracking points: combine raycasting and frustum (without cotracker condition)
            # batch_visible_reshaped and frustum_visible_reshaped are already computed above
            visible_among_tracking_points = batch_visible_reshaped[:, tracking_indices] & frustum_visible_reshaped[:, tracking_indices]  # [M, N_tracking]
            
            # Store results in full-size arrays (only at tracking indices)
            visibility_results_among_tracking[:, h, tracking_indices] = visible_among_tracking_points  # [M, N] but only tracking indices filled
            
            # Compute rewards for tracking points for this horizon (average over tracking points)
            rewards_tracking_h = visible_among_tracking_points.mean(axis=1)  # [M]
        else:
            # No tracking points, set reward to 0
            rewards_tracking_h = np.zeros(batch_size, dtype=np.float64)
        
        rewards_among_tracking_list.append(rewards_tracking_h)
        add_timing('per_horizon/tracking_reward', section_start)
        
    
    section_start = time.time()
    # reward per batch element = visibility averaged over all query points and horizon steps
    rewards = visibility_results[:, selected_horizons].mean(axis=-1).mean(axis=-1) # [b, selected_horizons, N] -> [b] [0(bad), 1(good)]
    
    assert rewards.shape == (batch_size,), 'rewards should be [batch_size,]'
    
    # Average over selected horizons for tracking points rewards
    rewards_among_tracking_points = np.array(rewards_among_tracking_list).mean(axis=0)  # [M]
    
    assert rewards_among_tracking_points.shape == (batch_size,), 'rewards_among_tracking_points should be [batch_size,]'
    
    # Average over selected horizons for wo_tracking_lost rewards
    rewards_wo_tracking_lost = visibility_results_wo_tracking_lost[:, selected_horizons].mean(axis=-1).mean(axis=-1) # [b, selected_horizons, N] -> [b] [0(bad), 1(good)]
    
    assert rewards_wo_tracking_lost.shape == (batch_size,), 'rewards_wo_tracking_lost should be [batch_size,]'

    # Create visibility_info dictionary for each horizon
    # visibility_info: {h: {'batch_visible': [bs, N], 'frustum_visible': [bs, N], 'mask_filtered_point_indices': [K] or None}}
    visibility_info = {}
    for h in selected_horizons:
        visibility_info[h] = {
            'batch_visible': batch_visible_info[:, h, :].copy(),  # [bs, N]
            'frustum_visible': frustum_visible_info[:, h, :].copy(),  # [bs, N]
            'mask_filtered_point_indices': mask_filtered_point_indices.copy() if mask_filtered_point_indices is not None and len(mask_filtered_point_indices) > 0 else None,  # [K] or None
        }
    add_timing('reward_and_info', section_start)

    if timing_enabled:
        total_time = time.time() - timing_total_start
        timing_parts = ', '.join(
            f'{name}={elapsed:.4f}s' for name, elapsed in timing_sections.items()
        )
        print(
            f'compute_visibility timing: total={total_time:.4f}s, '
            f'batch={batch_size}, horizon={horizon}, selected={list(selected_horizons)}, N={N}; '
            f'{timing_parts}'
        )
    
    return rewards, visibility_results, hit_distances, rewards_among_tracking_points, visibility_results_among_tracking, rewards_wo_tracking_lost, visibility_results_wo_tracking_lost, visibility_info

def create_pose_from_9d(pose_9d):
    N = pose_9d.shape[0]
    position = pose_9d[:, :3]
    ortho6d = pose_9d[:, 3:9]
    rotation_matrix = compute_rotation_matrix_from_ortho6d(ortho6d)
    pose_4x4 = np.tile(np.eye(4)[None, :, :], (N, 1, 1))
    pose_4x4[:, :3, :3] = rotation_matrix
    pose_4x4[:, :3, 3] = position
    return pose_4x4


def compute_robot_meshes_for_poses(robot_viz, end_effector_poses_marker, gripper_angle, T_B_M, T_W_B, horizon, selected_horizons, 
robot_joint_positions = None, robot_gripper_positions = None):
    """
    Compute robot meshes for given end-effector poses (vectorized version)
    
    Args:
        robot_viz: Z1RobotVisualizer instance
        end_effector_poses_marker: [horizon, 6] end-effector poses in marker coordinates
        T_B_M: Robot base to marker transformation
        horizon: prediction horizon
    
    Returns:
        robot_meshes_list: List of robot meshes for each horizon step
        robot_joint_angles_list: List of joint angles for each horizon step
        end_effector_poses_base: List of end-effector poses in robot base coordinates
    """
    if robot_joint_positions is None:
        # Convert end-effector poses from marker coordinates to robot base coordinates (vectorized)
        poses_marker_4x4 = create_pose_from_9d(end_effector_poses_marker)  # [horizon, 4, 4]
        end_effector_poses_base = T_B_M[None] @ poses_marker_4x4  # [1, 4, 4] @ [horizon, 4, 4] -> [horizon, 4, 4]


    # Solve inverse kinematics for each horizon step
    robot_meshes_list = [{}] * horizon  # Initialize with empty dicts for all horizon steps
    
    for h in selected_horizons:
        if robot_joint_positions is None:
            ik_start = time.time()
            ik_success = robot_viz.solve_inverse_kinematics(end_effector_poses_base[h])
            ik_time = time.time() - ik_start
            gripper = gripper_angle[h]
        elif robot_joint_positions is not None and robot_gripper_positions is not None:
            robot_viz.set_joint_angles(robot_joint_positions[h], robot_gripper_positions[h])
            gripper = robot_gripper_positions[h]
        else:
            raise ValueError('Either robot_joint_positions or current_joint_position must be provided')

        
        robot_meshes = robot_viz.get_robot_meshes_in_specified_transform(
            gripper_angle=gripper, T_target=T_W_B,
        )
        robot_meshes_list[h] = robot_meshes
        
    
    return robot_meshes_list


def precompute_combined_scenes(mesh_original, robot_meshes_list, horizon, selected_horizons):
    """
    Pre-compute combined scenes with robot meshes for all horizon steps
    
    Args:
        mesh_original: Original scene mesh
        robot_meshes_list: List of robot meshes for each horizon step
        horizon: prediction horizon
    
    Returns:
        combined_scenes_list: List of pre-computed combined scenes for each horizon step
    """
    combined_scenes_list = [None] * horizon  # Initialize with None for all horizon steps
    

    for h in selected_horizons:
        scene = create_combined_scene_with_robot(mesh_original, robot_meshes_list[h])
        combined_scenes_list[h] = scene
    
    return combined_scenes_list


def compute_precomputed_scenes_for_task_kwargs(task_kwargs, stats=None, args=None):
    """
    Compute precomputed scenes for visibility computation.
    This function extracts the scene computation logic from run_svdd so it can be reused.
    
    Args:
        task_kwargs: dict containing necessary data for scene computation:
            - selected_horizons: list of horizon indices to compute scenes for
            - robot_viz: Robot visualizer instance
            - T_B_M: Transformation from base to marker [4, 4]
            - robot_future_action: Robot future action [horizon, act_dim] in marker coordinate
            - mesh_original: Original scene mesh (Open3D TriangleMesh) OR
            - mesh_vertices, mesh_triangles, mesh_vertex_colors: Pickleable mesh data
            - T_mw: Transformation from marker to world [4, 4]
        stats: Normalization stats (optional, only needed if args is provided)
        args: Configuration args (optional, only needed for normalization)
    
    Returns:
        precomputed_scenes: List of precomputed scenes for each horizon step, or None if computation fails
    """
    if task_kwargs is None:
        return None
    
    selected_horizons = task_kwargs.get('selected_horizons')
    if selected_horizons is None:
        return None
    
    robot_viz = task_kwargs.get('robot_viz')
    T_B_M = task_kwargs.get('T_B_M')
    robot_future_action = task_kwargs.get('robot_future_action')
    
    if robot_future_action is not None:
        robot_future_action = robot_future_action.copy()
    
    
    mesh_original = task_kwargs.get('mesh_original')
    if mesh_original is None and task_kwargs.get('mesh_vertices') is not None and task_kwargs.get('mesh_triangles') is not None:
        mesh_original = o3d.geometry.TriangleMesh()
        mesh_original.vertices = o3d.utility.Vector3dVector(task_kwargs['mesh_vertices'])
        mesh_original.triangles = o3d.utility.Vector3iVector(task_kwargs['mesh_triangles'])
        mesh_vertex_colors = task_kwargs.get('mesh_vertex_colors')
        if mesh_vertex_colors is not None and len(mesh_vertex_colors) == len(task_kwargs['mesh_vertices']):
            mesh_original.vertex_colors = o3d.utility.Vector3dVector(mesh_vertex_colors)
        mesh_original.compute_vertex_normals()
    
    T_mw = task_kwargs.get('T_mw')
    
    # Check if all required components are available
    if not all([robot_viz is not None, T_B_M is not None, robot_future_action is not None, mesh_original is not None, T_mw is not None]):
        return None
    
    # Compute T_W_B
    T_W_B = np.linalg.inv(T_B_M @ T_mw)
    
    # Get end-effector poses
    end_effector_poses_marker = robot_future_action[:, :9]  # [horizon, 9]
    
    # Process gripper action
    gripper_action = robot_future_action[:, -1]  # [horizon] # marker coordinate, unnormalized
    # NOTE: assume gripper_action is in [0 (open), 1 (close)]
    gripper_action = gripper_action.copy()
    gripper_action *= -1.0  # [-1 (close), 0 (open)]
    gripper_action += 1.0  # [0 (close), 1 (open)]
    gripper_action *= 0.04  # [0, 0.04] m
    gripper_angle = gripper_action.copy()
    
    # Pre-compute robot meshes once
    robot_meshes_list = compute_robot_meshes_for_poses(
        robot_viz, end_effector_poses_marker, gripper_angle, T_B_M, T_W_B, robot_future_action.shape[0], selected_horizons
    )
    
    # Pre-compute combined scenes once
    precomputed_scenes = precompute_combined_scenes(
        mesh_original, robot_meshes_list, robot_future_action.shape[0], selected_horizons
    )
    
    return precomputed_scenes


def convert_to_relative(se3_matrices):
    se3_matrices = se3_matrices.copy()
    # SE(3) matrices [B, H, 4, 4]
    
    # Get inverse of first transformation for each batch [B, 4, 4]
    first_transforms = se3_matrices[:, 0]  # [B, 4, 4]
    first_inv = np.linalg.inv(first_transforms)  # [B, 4, 4]
    
    # Apply relative transformation: T_rel = T_first_inv @ T_current
    # Broadcasting: [B, 4, 4] @ [B, H, 4, 4] -> [B, H, 4, 4]
    relative_se3 = np.einsum('bij,bhjk->bhik', first_inv, se3_matrices)
    
    # Set first index to identity
    relative_se3[:, 0] = np.eye(4)
    
    return relative_se3


def convert_se3_to_9d(se3_matrices):
    se3_matrices = se3_matrices.copy()
    B, H, _ , _ = se3_matrices.shape
    pose_9d = np.zeros((B, H, 9))
    pose_9d[..., :3] = se3_matrices[..., :3, 3]  # positions
    ortho6d = compute_ortho6d_from_rotation_matrix(se3_matrices[..., :3, :3].reshape(-1, 3, 3)).reshape(B, H, 6)
    pose_9d[..., 3:9] = ortho6d
    
    return pose_9d


def convert_9d_to_se3(pose_9d):
    pose_9d = pose_9d.copy()
    B, H, _ = pose_9d.shape
    se3_matrices = np.tile(np.eye(4)[None, None, :, :], (B, H, 1, 1))
    se3_matrices[..., :3, 3] = pose_9d[..., :3]
    ortho6d = pose_9d[..., 3:9]
    rotation_matrices = compute_rotation_matrix_from_ortho6d(ortho6d.reshape(-1, 6)).reshape(B, H, 3, 3)
    se3_matrices[..., :3, :3] = rotation_matrices
    return se3_matrices

def _se3_residual(T_fk: np.ndarray, T_tgt: np.ndarray) -> np.ndarray:
    """
    SE(3) residual as 6D vector:
    - translation error: p_fk - p_tgt   (in meters, assuming)
    - rotation error:    log(R_tgt^T R_fk) as rotvec (in radians)
    """
    assert T_fk.shape == (4, 4)
    assert T_tgt.shape == (4, 4)

    p_fk = T_fk[:3, 3]
    p_tgt = T_tgt[:3, 3]

    R_fk = T_fk[:3, :3]
    R_tgt = T_tgt[:3, :3]

    # rotation error: identity when fk == tgt
    R_err = R_tgt.T @ R_fk
    rotvec = R.from_matrix(R_err).as_rotvec()  # 3-vector (axis * angle)

    return np.concatenate([p_fk - p_tgt, rotvec], axis=0)

def retarget_se3_trajectory(
    robot_viz,
    target_T: np.ndarray,          # [T,4,4]
    q0: np.ndarray,                # [DoF]
    *,
    w_pos: float = 1.0,            # weight for position residual
    w_rot: float = 1.0,            # weight for rotation residual
    lambda_smooth: float = 1e-2,   # smoothness weight (q_t - q_{t-1})
    bounds=None,                   # (lower, upper) each can be scalar or [DoF]
    max_nfev: int = 50,
    tol: float = 1e-6,
    verbose: int = 0,
):
    """
    Retarget SE(3) target trajectory into joint trajectory via per-timestep IK.
    
    Args:
        robot_viz: Robot visualizer with compute_forward_kinematics method (Z1RobotVisualizer or WidowXVisualizer)
        target_T: [T, 4, 4] target SE(3) trajectory
        q0: [DoF] initial joint angles
        w_pos: weight for position residual
        w_rot: weight for rotation residual
        lambda_smooth: smoothness weight
        bounds: joint limits (lower, upper)
        max_nfev: maximum function evaluations per timestep
        tol: tolerance
        verbose: verbosity level
    
    Returns:
        q_traj: [T, DoF] retargeted joint trajectory
        T_traj: [T, 4, 4] forward kinematics of retargeted trajectory
        info: dict with per-step cost and solver status
    """
    from scipy.optimize import least_squares
    
    target_T = np.asarray(target_T)
    q0 = np.asarray(q0).astype(float)

    assert target_T.ndim == 3 and target_T.shape[1:] == (4, 4), "target_T must be [T,4,4]"
    T = target_T.shape[0]
    dof = q0.shape[0]

    if bounds is None:
        # Use robot_viz's joint_limits if available
        if hasattr(robot_viz, 'joint_limits'):
            lb = np.array([limit[0] for limit in robot_viz.joint_limits])
            ub = np.array([limit[1] for limit in robot_viz.joint_limits])
        else:
            lb = -np.inf * np.ones(dof)
            ub = +np.inf * np.ones(dof)
    else:
        lb, ub = bounds
        lb = np.broadcast_to(np.asarray(lb, dtype=float), (dof,)).copy()
        ub = np.broadcast_to(np.asarray(ub, dtype=float), (dof,)).copy()

    # sqrt weights for least_squares residual scaling
    s_pos = np.sqrt(w_pos)
    s_rot = np.sqrt(w_rot)
    s_sm  = np.sqrt(lambda_smooth)

    q_prev = q0.copy()
    q_traj = np.zeros((T, dof), dtype=float)
    T_traj = np.zeros((T, 4, 4), dtype=float)

    costs = []
    statuses = []
    nfev_list = []  # Track number of function evaluations per timestep

    for t in range(T):
        T_tgt = target_T[t]

        def residual(q):
            T_fk = robot_viz.compute_forward_kinematics(q)  # must return (4,4)
            e6 = _se3_residual(T_fk, T_tgt)   # [6] = [pos(3), rot(3)]
            # scale residuals
            e6_scaled = np.concatenate([s_pos * e6[:3], s_rot * e6[3:]], axis=0)
            smooth = s_sm * (q - q_prev)      # [DoF]
            return np.concatenate([e6_scaled, smooth], axis=0)

        # First timestep may need more iterations if starting from poor initial guess
        # Use more iterations for first timestep, fewer for subsequent (warm start)
        current_max_nfev = max_nfev * 2 if t == 0 else max_nfev

        res = least_squares(
            residual,
            x0=q_prev,             # warm start
            bounds=(lb, ub),
            max_nfev=current_max_nfev,
            xtol=tol,
            ftol=tol,
            gtol=tol,
            verbose=verbose,
        )

        q_prev = res.x
        q_traj[t] = q_prev
        T_traj[t] = robot_viz.compute_forward_kinematics(q_prev)

        # (optional) store diagnostics
        costs.append(res.cost)      # 0.5 * sum(residual^2)
        statuses.append({"success": bool(res.success), "status": int(res.status), "message": res.message})
        nfev_list.append(res.nfev)  # Number of function evaluations for this timestep

    info = {
        "costs": np.asarray(costs, dtype=float),
        "statuses": statuses,
        "weights": {"w_pos": w_pos, "w_rot": w_rot, "lambda_smooth": lambda_smooth},
        "nfev_list": np.asarray(nfev_list, dtype=int),  # Function evaluations per timestep
        "total_nfev": sum(nfev_list),  # Total function evaluations
    }
    return q_traj, T_traj, info

def retarget_se3_trajectory_visibility(
    robot_viz,
    target_T: np.ndarray,          # [T,4,4]
    q0: np.ndarray,                # [DoF]
    task_kwargs,                   # dict for compute_visibility
    T_B_M_view: np.ndarray,        # [4,4] transformation from marker to base coordinate
    *,
    w_pos: float = 1.0,            # weight for position residual
    w_rot: float = 1.0,            # weight for rotation residual
    w_visibility: float = 1.0,    # weight for visibility residual
    lambda_smooth: float = 1e-2,   # smoothness weight (q_t - q_{t-1})
    bounds=None,                   # (lower, upper) each can be scalar or [DoF]
    max_nfev: int = 50,
    tol: float = 1e-6,
    verbose: int = 0,
):
    """
    Retarget SE(3) target trajectory into joint trajectory via per-timestep IK with visibility consideration.
    Similar to retarget_se3_trajectory but also considers visibility reward in the residual.
    
    Args:
        robot_viz: Robot visualizer with compute_forward_kinematics method (Z1RobotVisualizer or WidowXVisualizer)
        target_T: [T, 4, 4] target SE(3) trajectory in base coordinate
        q0: [DoF] initial joint angles
        task_kwargs: dict containing necessary data for visibility computation (query_point, precomputed_scenes, etc.)
        T_B_M_view: [4, 4] transformation from marker to base coordinate
        w_pos: weight for position residual
        w_rot: weight for rotation residual
        w_visibility: weight for visibility residual (negative visibility reward)
        lambda_smooth: smoothness weight
        bounds: joint limits (lower, upper)
        max_nfev: maximum function evaluations per timestep
        tol: tolerance
        verbose: verbosity level
    
    Returns:
        q_traj: [T, DoF] retargeted joint trajectory
        T_traj: [T, 4, 4] forward kinematics of retargeted trajectory
        info: dict with per-step cost and solver status
    """
    from scipy.optimize import least_squares
    
    target_T = np.asarray(target_T)
    q0 = np.asarray(q0).astype(float)
    T_B_M_view = np.asarray(T_B_M_view)

    assert target_T.ndim == 3 and target_T.shape[1:] == (4, 4), "target_T must be [T,4,4]"
    assert T_B_M_view.shape == (4, 4), "T_B_M_view must be [4,4]"
    T = target_T.shape[0]
    dof = q0.shape[0]
    
    # Get T_E_C_view for converting end-effector to camera pose
    T_E_C_view = robot_viz.T_E_C.copy()  # [4, 4]

    if bounds is None:
        # Use robot_viz's joint_limits if available
        if hasattr(robot_viz, 'joint_limits'):
            lb = np.array([limit[0] for limit in robot_viz.joint_limits])
            ub = np.array([limit[1] for limit in robot_viz.joint_limits])
        else:
            lb = -np.inf * np.ones(dof)
            ub = +np.inf * np.ones(dof)
    else:
        lb, ub = bounds
        lb = np.broadcast_to(np.asarray(lb, dtype=float), (dof,)).copy()
        ub = np.broadcast_to(np.asarray(ub, dtype=float), (dof,)).copy()

    # sqrt weights for least_squares residual scaling
    s_pos = np.sqrt(w_pos)
    s_rot = np.sqrt(w_rot)
    s_sm  = np.sqrt(lambda_smooth)
    s_vis = np.sqrt(w_visibility)

    q_prev = q0.copy()
    q_traj = np.zeros((T, dof), dtype=float)
    T_traj = np.zeros((T, 4, 4), dtype=float)

    costs = []
    statuses = []
    nfev_list = []  # Track number of function evaluations per timestep

    for t in range(T):
        T_tgt = target_T[t]

        def residual(q):
            # Compute forward kinematics for current timestep
            T_B_E_fk = robot_viz.compute_forward_kinematics(q)  # [4,4] base coordinate, end-effector
            
            # SE(3) residual
            e6 = _se3_residual(T_B_E_fk, T_tgt)   # [6] = [pos(3), rot(3)]
            e6_scaled = np.concatenate([s_pos * e6[:3], s_rot * e6[3:]], axis=0)
            
            # Smoothness residual
            smooth = s_sm * (q - q_prev)      # [DoF]
            
            # Visibility residual: construct full trajectory and compute visibility
            # Build trajectory: current timestep uses FK result, previous timesteps use T_traj, future timesteps use target_T
            T_B_E_traj = np.zeros((T, 4, 4), dtype=float)
            T_B_E_traj[:t] = T_traj[:t]  # Previous timesteps (already computed)
            T_B_E_traj[t] = T_B_E_fk  # Current timestep (from FK)
            T_B_E_traj[t+1:] = target_T[t+1:]  # Future timesteps (use target)
            
            # Convert end-effector poses to camera poses: T_B_C = T_B_E @ T_E_C
            T_B_C_traj = T_B_E_traj @ T_E_C_view  # [T, 4, 4] base coordinate, camera
            
            # Convert base coordinate to marker coordinate: T_M_C = T_M_B @ T_B_C
            T_M_B_view = np.linalg.inv(T_B_M_view)
            T_M_C_traj = np.einsum('ij,hjk->hik', T_M_B_view, T_B_C_traj)  # [T, 4, 4] marker coordinate, camera
            
            # Convert SE(3) to 9D format: [T, 4, 4] -> [1, T, 9]
            T_M_C_traj_expanded = T_M_C_traj[None, :, :, :]  # [1, T, 4, 4]
            viewpoint_9d = convert_se3_to_9d(T_M_C_traj_expanded)  # [1, T, 9] marker coordinate
            
            # Compute visibility reward
            # Note: compute_visibility expects [bs, horizon, 9] and returns rewards [bs]
            visibility_reward, _, _, _, _, _, _ = compute_visibility(viewpoint_9d, task_kwargs)
            visibility_reward_scalar = visibility_reward[0]  # scalar
            
            # Convert visibility reward to residual (negative because we want to maximize visibility)
            # Higher visibility -> lower residual
            visibility_residual = s_vis * (1.0 - visibility_reward_scalar)  # scalar, convert to residual
            
            # Combine all residuals
            return np.concatenate([e6_scaled, smooth, np.array([visibility_residual])], axis=0)

        # First timestep may need more iterations if starting from poor initial guess
        # Use more iterations for first timestep, fewer for subsequent (warm start)
        current_max_nfev = max_nfev * 2 if t == 0 else max_nfev

        res = least_squares(
            residual,
            x0=q_prev,             # warm start
            bounds=(lb, ub),
            max_nfev=current_max_nfev,
            xtol=tol,
            ftol=tol,
            gtol=tol,
            verbose=verbose,
        )

        q_prev = res.x
        q_traj[t] = q_prev
        T_traj[t] = robot_viz.compute_forward_kinematics(q_prev)

        # (optional) store diagnostics
        costs.append(res.cost)      # 0.5 * sum(residual^2)
        statuses.append({"success": bool(res.success), "status": int(res.status), "message": res.message})
        nfev_list.append(res.nfev)  # Number of function evaluations for this timestep

    info = {
        "costs": np.asarray(costs, dtype=float),
        "statuses": statuses,
        "weights": {"w_pos": w_pos, "w_rot": w_rot, "w_visibility": w_visibility, "lambda_smooth": lambda_smooth},
        "nfev_list": np.asarray(nfev_list, dtype=int),  # Function evaluations per timestep
        "total_nfev": sum(nfev_list),  # Total function evaluations
    }
    return q_traj, T_traj, info

def retargeting(robot_viz, x0_svdd_9d, T_B_M_view, T_M_C, use_visibility_aware_retargeting=False, task_kwargs=None):
    """
    Retarget viewpoint trajectory (9D format) using IK and retargeting if needed.
    
    Args:
        robot_viz: Robot visualizer with methods:
            - compute_forward_kinematics(joint_angles, gripper_angle)
            - solve_ik_null_space(target_T, initial_guess, ...)
            - joint_limits: joint limits
        x0_svdd_9d: [B, H, 9] array of viewpoint poses (position: 3, ortho6d: 6) in marker coordinate
        T_B_M_view: [4, 4] transformation from marker to base coordinate (optional)
        T_M_C: [4, 4] transformation from camera to marker coordinate (optional, for initial joint guess)
    
    Returns:
        x0_svdd_9d_retargeted: [B, H, 9] array of retargeted viewpoint poses in marker coordinate
    """
    retarget_start = time.time()
    B, H, _ = x0_svdd_9d.shape
    T_E_C_view = robot_viz.T_E_C.copy()
    # Convert 9D to SE(3) in marker coordinate
    x0_svdd_se3_marker = convert_9d_to_se3(x0_svdd_9d)  # [B, H, 4, 4] marker coordinate
    
    # Convert to base coordinate if T_B_M_view is provided
    
    # Convert marker to base: T_B_C = T_B_M_view @ T_M_C
    x0_svdd_se3_base = np.einsum('ij,bhjk->bhik', T_B_M_view, x0_svdd_se3_marker)  # [H, 4, 4] base coordinate
    
    
    # For each batch (assuming B=1 for now, can extend if needed)
    x0_svdd_9d_retargeted_list = []
    
    for b in range(B):
        T_B_C_target = x0_svdd_se3_base[b]  # [H, 4, 4] base coordinate
        T_B_E_target = T_B_C_target @ np.linalg.inv(T_E_C_view) # [H, 4, 4] 
        
        
        # Convert current camera pose to base coordinate
        T_M_C_current = T_M_C.copy()  # [4, 4] marker coordinate
        T_B_C_current = T_B_M_view @ T_M_C_current  # [4, 4] base coordinate
        T_B_E_current = T_B_C_current @ np.linalg.inv(T_E_C_view) # [4, 4] end-effector coordinate
        
        # Use IK to get initial joint guess from current camera pose
        # Note: This assumes robot_viz has solve_ik_null_space method
        
        # Get current joint angles from task_kwargs if available, otherwise use robot_viz.joint_angles
        if task_kwargs is not None and 'current_joint_pos' in task_kwargs:
            q0 = task_kwargs['current_joint_pos'].copy()
        else:
            # Fallback to robot_viz.joint_angles if current_joint_pos is not available
            q0 = robot_viz.joint_angles.copy()
        
        
        # Perform retargeting
        joint_limits = robot_viz.joint_limits
        q_min = np.array([limit[0] for limit in joint_limits])
        q_max = np.array([limit[1] for limit in joint_limits])
        if use_visibility_aware_retargeting:
            raise NotImplementedError("Do not use this. convergence is super slow and not guaranteed.")
            q_traj, T_traj, retarget_info = retarget_se3_trajectory_visibility(
                robot_viz,
                T_B_E_target,
                q0,
                task_kwargs,
                T_B_M_view,
                w_pos=1.0,
                w_rot=0.5,
                w_visibility=10.0,
                lambda_smooth=1e-3,
                bounds=(q_min, q_max),
                max_nfev=100,
                tol=1e-3,
                verbose=0,
            )
        else:
            q_traj, T_traj, retarget_info = retarget_se3_trajectory(
                robot_viz,
                T_B_E_target,
                q0,
                w_pos=1.0,
                w_rot=0.5,
                lambda_smooth=1e-3,
                bounds=(q_min, q_max),
                max_nfev=100,
            )
            
        # Convert retargeted trajectory back to marker coordinate
        T_M_B_view = np.linalg.inv(T_B_M_view)
        T_M_E_retargeted = np.einsum('ij,hjk->hik', T_M_B_view, T_traj)  # [H, 4, 4] marker coordinate
        T_M_C_retargeted = np.einsum('hij,jk->hik', T_M_E_retargeted, T_E_C_view) # [H, 4, 4] camera coordinate
        
        
        # Convert SE(3) back to 9D
        x0_svdd_9d_retargeted_b = convert_se3_to_9d(T_M_C_retargeted[None])[0]  # [H, 9] marker coordinate
        x0_svdd_9d_retargeted_list.append(x0_svdd_9d_retargeted_b)
        
    
    print(f"View env retargeting completed. Batch: {B}, time taken: {time.time() - retarget_start:.2f}s. fev: {retarget_info['total_nfev']}. Final cost: {retarget_info['costs'][-1]:.6f}")
    
    x0_svdd_9d_retargeted = np.array(x0_svdd_9d_retargeted_list)  # [B, H, 9]
    
    return x0_svdd_9d_retargeted

@torch.no_grad()
def calculate_weight(args, noise_scheduler, latents, new_noise_pred, t, dataset, pt_max, pt_min, pt_mean, pt_std, flow_norm_type='min_max'): # t = 981, 961, 941 ..
    
    pred_original_sample = noise_scheduler.step(
                    new_noise_pred, t, latents
                ).pred_original_sample


    B, H, ND = pred_original_sample.shape
    N, D = args.dataset.num_dift_points, args.dataset.flow_dim
    pred_original_sample = pred_original_sample.reshape(B,H,N,D).detach().cpu().numpy() # [bs, h, N, D]
    if args.use_relative:
        raise NotImplementedError
    if flow_norm_type == 'min_max':
        # roughly [-1, 1] ->  [0, 1]
        pred_original_sample = dataset.unnormalize_transform(pred_original_sample)

        # roughly [0, 1] ->  raw values
        pred_original_sample = pred_original_sample * (pt_max[None, None, None, :] - pt_min[None, None, None, :]) + pt_min[None, None, None, :]
    elif flow_norm_type == 'mean_std':
        raise NotImplementedError
        
    pred_original_sample = torch.from_numpy(pred_original_sample).to(latents.device)
    
    reward_dict = compute_reward(args, pred_original_sample)
    weights = reward_dict['total_reward']  # Extract total reward for backward compatibility
    
    return weights


@torch.no_grad()
def calculate_weight_active_vision(args, noise_scheduler, latents, new_noise_pred, t, dataset, task_kwargs, fix_mask, prior, stats, current_viewpoint): # t = 981, 961, 941 ..
    
    pred_original_sample = noise_scheduler.step(
                    new_noise_pred, t, latents
                ).pred_original_sample

    device = pred_original_sample.device
    if args.use_fix_mask:
        pred_original_sample = (1. - fix_mask) * pred_original_sample + fix_mask * prior

    pred_original_sample_np = pred_original_sample.detach().cpu().numpy() # [B*DUP, h, dim]
    
    if args.training.use_ptp:
        # only consider future actions to compute the reward
        pred_original_sample_np = pred_original_sample_np[:, args.dataset.obs_horizon-1:, :].copy()
    
    B, H, D = pred_original_sample_np.shape
    pred_original_sample_np = pred_original_sample_np.reshape(-1, D) # [B*DUP*h, dim]

    if not args.use_relative:
        pred_original_sample_np = unnormalize_data(pred_original_sample_np, stats=stats["view_actions"], type=args.dataset.norm_type)
        pred_original_sample_np = pred_original_sample_np.reshape(B, H, -1) # [B*DUP, h, dim]
    else: # relative
        pred_original_sample_np = unnormalize_data(pred_original_sample_np, stats=stats["delta_view_action_for_uncondition"], type=args.dataset.norm_type)
        pred_original_sample_np = pred_original_sample_np.reshape(B, H, -1) # [B*DUP, h, dim(7)]
        
        relative_pos = pred_original_sample_np[..., :3].copy() # [b*dup, h, 3]
        relative_rotvec = pred_original_sample_np[..., 3:6].copy() # [b*dup, h, 3] # axis angle
        relative_rot = R.from_rotvec(relative_rotvec.reshape(-1, 3)).as_matrix().reshape(B, H, 3, 3) # [b*dup, h, 3, 3]

        # unnormalized
        current_viewpoint = np.tile(current_viewpoint[None, :], (B, 1)) # [9] -> [b*dup, 9]
        current_viewpoint_pos = current_viewpoint[..., :3].copy() # [b*dup, 3]
        current_viewpoint_ortho6d = current_viewpoint[..., 3:9].copy() # [b*dup, 6]
        current_viewpoint_rot = compute_rotation_matrix_from_ortho6d(current_viewpoint_ortho6d) # [b*dup, 3, 3]
        
        absolute_pos = current_viewpoint_pos[:, None, :] + np.cumsum(relative_pos, axis=1) # [b*dup, h, 3], {t+1}, {t+2}, {t+3}, ...
        # cumulative product of rotation matrices (itertools.accumulate)
        # process each batch element
        cumulative_rot_matrices = np.zeros_like(relative_rot)  # [b*dup, h, 3, 3]
        for b in range(B):
            rot_objects = [R.from_matrix(relative_rot[b, i]) for i in range(H)]
            cumulative_rots = list(accumulate(rot_objects, lambda acc, r: acc * r, initial=R.identity()))[1:]
            cumulative_rot_matrices[b] = np.array([r.as_matrix() for r in cumulative_rots])
        absolute_rot = np.einsum('bij,btjk->btik', current_viewpoint_rot, cumulative_rot_matrices) # [b*dup, h, 3, 3]
        absolute_ortho6d = compute_ortho6d_from_rotation_matrix(absolute_rot.reshape(-1, 3, 3)).reshape(B, H, 6) # [b*dup, h, 6]
        pred_original_sample_np = np.concatenate([absolute_pos, absolute_ortho6d], axis=-1) # [b*dup, h, 9]


    if not args.use_relative:
        assert D == 9, 'assume position, ortho6d'
    else:
        assert D == 6, 'assume relative position, axis angle'
    
    
    # Convert [B*DUP, H, 6] (position, euler) to [B*DUP, H, 4, 4] SE(3) matrices
    # pred_original_sample is in marker coordinate system
    
    assert task_kwargs['use_marker_coordinate'], 'currently, only support marker coordinate'
    
    
    # # Convert euler angles to rotation matrices using scipy
    # # 'xyz' corresponds to roll, pitch, yaw order
    # # rotations = R.from_euler('xyz', euler_flat)
    # # rotation_matrices = rotations.as_matrix()  # [B*H, 3, 3]

    
    start = time.time()
    reward_dict = compute_reward(args, pred_original_sample_np, task_kwargs, device=device)
    weights = reward_dict['total_reward']  # Extract total reward for backward compatibility
    return weights, reward_dict  # Return both weights and reward_dict for detailed analysis

@torch.no_grad()
def run_svdd(args, noise_scheduler, num_inference_steps, model, point_flow, proprioception, dataset=None, stats=None, task=None,  task_kwargs=None,  
            num_fix_steps=1, num_fresh_steps=1, future_action_sequence=None, return_task_kwargs=False, visibility=None, viewpoint=None, use_condition=False, use_retargeting=False):
    '''
    NOTE: assume
    point_flow: [bs, h, N, D]
    proprioception: [bs, h, dim]
    viewpoint: [bs, h, dim] - viewpoint history (normalized, assumed to be tensor, not None)
    viewpoint_mask: [bs, h] - viewpoint mask (bool tensor, None if not provided)
    
    stats: {
        point_tracking_data: {
            min: [dim], max: [dim]
    }
    '''
    pt_max = dataset.pt_max.copy()
    pt_min = dataset.pt_min.copy()
    pt_mean = dataset.pt_mean.copy()
    pt_std = dataset.pt_std.copy()
    flow_norm_type = args.dataset.flow_norm_type

    do_classifier_free_guidance = False # args.guidance_scale > 1.0
    if task == 'active_vision':
        if args.use_relative:
            ND = 6 # position, axis angle

        else:
            ND = 9 # position, ortho6d
    else:
        ND = args.dataset.num_dift_points*args.dataset.flow_dim
    
    # Store original ND (action_dim) for later extraction
    ND_action = ND
    
    # If predict_flow is True, ND should include flow_output_dim
    # Note: predict_separate_flow doesn't need to change ND since flow is reconstructed separately
    if model.predict_flow:
        raise NotImplementedError('since it is view policy, have not considered predict flow yet')
        flow_output_dim = model.flow_output_dim
        ND = ND + flow_output_dim  # action_dim + flow_output_dim
    
    noise_scheduler.set_timesteps(num_inference_steps)


    timesteps = noise_scheduler.timesteps.clone()

    temp_batch_size = point_flow.shape[0] # number of samples to generate
    input_horizon = args.dataset.pred_horizon if not args.training.use_ptp else args.dataset.pred_horizon + args.dataset.obs_horizon - 1
    shape = (temp_batch_size, input_horizon, ND)
    
    
    latents = torch.randn(shape).to(point_flow.device) * args.svdd.temperature
    

    B = latents.shape[0]  # temp_batch_size
    DUP = args.svdd.duplicate
    

    # Pre-compute robot meshes and combined scenes for active vision task (only once before timesteps loop)
    precomputed_robot_data = None
    precomputed_scenes = None
    
    
    if task == 'active_vision' and task_kwargs is not None:
        # import here beacuse of multiprocessing
        from egoavflow.utils import compute_ortho6d_from_rotation_matrix, compute_rotation_matrix_from_ortho6d 
        from egoavflow.diffusion_policy.dataloader.diffusion_bc_dataset import normalize_data, unnormalize_data
        selected_horizons = task_kwargs['selected_horizons']
        T_M_C = task_kwargs.get('T_mc_transformation') # [4, 4] (from time t)
        
        # position, euler
        mc_position = T_M_C[:3, 3] # [3]
        mc_ortho6d = compute_ortho6d_from_rotation_matrix(T_M_C[:3, :3][None])[0] # [6]
        mc_viewpoints_unnorm = np.concatenate([mc_position, mc_ortho6d], axis=0) # [9]

        mc_viewpoints = normalize_data(mc_viewpoints_unnorm[None], stats=stats["view_actions"], type=args.dataset.norm_type)[0] # [9]
        mc_viewpoints = torch.from_numpy(mc_viewpoints).float().to(point_flow.device)
        
        # Compute precomputed scenes using the extracted function
        start = time.time()
        precomputed_scenes = compute_precomputed_scenes_for_task_kwargs(task_kwargs, stats=stats, args=args)
        if precomputed_scenes is not None:
            print(f" Precompute combined scenes time: {time.time() - start:.4f} seconds")
            task_kwargs['precomputed_scenes'] = precomputed_scenes
            

    # precompute the expaneded conditions
    point_flow_expanded = point_flow.unsqueeze(0).expand(DUP, *point_flow.shape) # [DUP, b, h, N, D]
    point_flow_expanded = point_flow_expanded.reshape(B * DUP, *point_flow.shape[-3:]) # [B*DUP, h, N, D]
    visibility_expanded = visibility.unsqueeze(0).expand(DUP, *visibility.shape) # [DUP, B(1), N]
    visibility_expanded = visibility_expanded.reshape(B * DUP, *visibility.shape[-1:]) # [B*DUP, N]
    
    proprioception_expanded = proprioception.unsqueeze(0).expand(DUP, *proprioception.shape) # [DUP, b, h, dim]
    proprioception_expanded = proprioception_expanded.reshape(B * DUP, *proprioception.shape[-2:]) # [B*DUP, h, dim]

    # Expand viewpoint
    viewpoint_expanded = viewpoint.unsqueeze(0).expand(DUP, *viewpoint.shape) # [DUP, b, h, dim]
    viewpoint_expanded = viewpoint_expanded.reshape(B * DUP, *viewpoint.shape[-2:]) # [B*DUP, h, dim]
    

    # generate sample via SVDD (reward maximization)
    start = time.time()
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents # [bs, h, dim] -> [bs*2, h, dim] # NOTE: the first bs entries correspond to the unconditional branch
        
        # predict the noise residual

        assert args.noise_scheduler.prediction_type == 'epsilon'
        

        if args.use_fix_mask:
            # For flow generation, we need to reshape to match the training format
            prior = torch.zeros_like(latent_model_input, device=latent_model_input.device)
            fix_mask = torch.zeros_like(latent_model_input, device=latent_model_input.device)
            pred_horizon = args.dataset.pred_horizon
            
            if future_action_sequence is None:
                if not args.use_relative:
                    # First inference: fix only the first timestep
                    fix_mask[:, 0, :] = 1.0
                    # Set prior for the first timestep
                    if task == 'active_vision':
                        # mc_viewpoints: [D(9)], broadcasting to [B, 1, D]
                        if model.predict_flow:
                            # Only set action part, flow part remains 0
                            prior[:, 0, :ND_action] = mc_viewpoints[None, :]  # [1, D] -> [B, 1, D] (broadcasting)
                        else:
                            prior[:, 0, :] = mc_viewpoints[None, :]  # [1, D] -> [B, 1, D] (broadcasting)
                    else:
                        # point_flow[:, -1]: [B, N, D] -> reshape to [B, N*D]
                        prior_data = point_flow[:, -1].reshape(B, -1)  # [B, N*D]
                        if model.predict_flow:
                            
                            # Only set action part, flow part remains 0
                            prior[:, 0, :ND_action] = prior_data  # [B, N*D] -> [B, 1, N*D] (broadcasting)
                        else:
                            prior[:, 0, :] = prior_data  # [B, N*D] -> [B, 1, N*D] (broadcasting)
            else:
                if args.training.use_ptp:
                    raise NotImplementedError('have not validated this yet. maybe you future_action_sequence may include past actions?, should use input_horizon instead of pred_horizon?')
                fix_mask = compute_fix_mask_weights(pred_horizon, num_fix_steps, num_fresh_steps).to(latent_model_input.device)
                fix_mask = torch.tile(fix_mask[None, :, None], (latent_model_input.shape[0], 1, latent_model_input.shape[-1])) # [B, h, dim]
                
                if future_action_sequence.shape[0] != pred_horizon: # tiled part will be addressed through fix_mask zero (last H-fix-soft steps)
                    future_action_sequence = np.concatenate([future_action_sequence, future_action_sequence[-1:].repeat(pred_horizon - future_action_sequence.shape[0], axis=0)]) # [pred_horizon(seq_len), 7]
                assert future_action_sequence.shape[0] == pred_horizon
                
                if task == 'active_vision':
                    from scipy.spatial.transform import Rotation as R
                    from egoavflow.utils import compute_ortho6d_from_rotation_matrix
                    if not args.use_relative:
                        # future_action_sequence: [horizon, 8] (unnormalized env format)
                        # Need to convert to view action format and normalize
                        # Convert env format (pos, quat, gripper) to view action format (pos, ortho6d)
                        
                        # Extract positions and quaternions for all steps at once (batch processing)
                        pos = future_action_sequence[:, :3]  # [seq_len, 3]
                        quat = future_action_sequence[:, 3:7]  # [seq_len, 4]
                        # Convert quaternions to rotation matrices (batch)
                        rot_matrix = R.from_quat(quat).as_matrix()  # [seq_len, 3, 3]
                        # Convert rotation matrices to ortho6d (batch)
                        ortho6d = compute_ortho6d_from_rotation_matrix(rot_matrix)  # [seq_len, 6]
                        # Concatenate positions and ortho6d
                        future_view_actions = np.concatenate([pos, ortho6d], axis=-1)  # [seq_len, 9]
                        # Normalize
                        from egoavflow.diffusion_policy.dataloader.diffusion_bc_dataset import normalize_data
                        future_view_actions_normalized = normalize_data(future_view_actions, stats=stats["view_actions"], type=args.dataset.norm_type)  # [seq_len, 9]
                        prior_action = torch.from_numpy(future_view_actions_normalized).float().to(point_flow.device)  # [B, pred_horizon, 9]
                        if model.predict_flow:
                            # Expand prior to include flow part (zeros)
                            prior = torch.zeros(B, pred_horizon, ND, device=point_flow.device, dtype=prior_action.dtype)
                            prior[:, :, :ND_action] = prior_action
                        else:
                            prior = prior_action
                    else:
                        future_action_pos = future_action_sequence[:, :3] # [pred_horizon, 3]
                        future_action_quat = future_action_sequence[:, 3:7]  # [pred_horizon, 4]
                        future_action_rot_matrix = R.from_quat(future_action_quat).as_matrix()  # [pred_horizon, 3, 3]

                        relative_pos = future_action_pos[1:] - future_action_pos[:-1] # [pred_horizon-1, 3]
                        relative_pos = np.concatenate([relative_pos, np.zeros((1, 3))], axis=0) # [pred_horizon, 3]
                        relative_rot = np.einsum('tij,tjk->tik', future_action_rot_matrix[:-1].transpose(0, 2, 1), future_action_rot_matrix[1:]) # [pred_horizon-1, 3, 3]
                        relative_rot = np.concatenate([relative_rot, np.eye(3)[None]], axis=0) # [pred_horizon, 3, 3]
                        relative_rotvec = R.from_matrix(relative_rot).as_rotvec() # [pred_horizon, 3]                        

                        future_view_actions = np.concatenate([relative_pos, relative_rotvec], axis=-1)  # [pred_horizon, 6]
                        future_view_actions_normalized = normalize_data(future_view_actions, stats=stats["delta_view_action_for_uncondition"], type=args.dataset.norm_type)  # [pred_horizon, 6]

                        prior_action = torch.from_numpy(future_view_actions_normalized).float().to(point_flow.device)  # [B, pred_horizon, 6]
                        if model.predict_flow:
                            # Expand prior to include flow part (zeros)
                            prior = torch.zeros(B, pred_horizon, ND, device=point_flow.device, dtype=prior_action.dtype)
                            prior[:, :, :ND_action] = prior_action
                        else:
                            prior = prior_action

                else:
                    raise NotImplementedError('have not validated this yet')
            # In evaluation, we replace the first timestep(s) with the current flow/viewpoint or future action sequence
            latent_model_input = (1. - fix_mask) * latent_model_input + fix_mask * prior


        # Convert inputs for CustomRDTRunner's model.forward
        # latent_model_input: [B, horizon, action_dim] or [B, horizon, action_dim + flow_output_dim] -> adapt to hidden_size
        action_input = model.action_adaptor(latent_model_input)  # [B, horizon, hidden_size]
        
        # Convert point_flow to flow_tokens: [B, h, N, D] -> [B, N, h*D] for point-wise positional embedding
        B_flow, H_flow, N_flow, D_flow = point_flow.shape
        
        # Create flow_mask based on visibility (last dimension of point_flow)
        # If using 4D 3D-flow or pixel-flow conditions, visibility is passed separately.
        # flow_mask: (B, N) where True means visible (masking=true), False means invisible (masking=false)
        if D_flow == 4 or use_pixel_flow_for_robot_policy(args):
            if args.training.use_masking: # flow: 3dim, but masking is applied
                flow_mask = visibility.bool()  # [B, N] - True if visible in the most recent timestep
                if not torch.any(flow_mask, dim=1).all():
                    print(f'WARNING: at least one should be visible(true) along N axis. We temporarily set arbitrary one to True')
                    flow_mask[:, torch.randint(0, N_flow, (1,), device=point_flow.device)] = True
                flow_mask = apply_random_flow_mask_count(flow_mask, get_random_flow_mask_count(args))
                assert torch.any(flow_mask, dim=1).all(), f"at least one should be visible(true) along N axis, flow_mask : {flow_mask}"
            else:
                flow_mask = None
        else:
            flow_mask = None
        
        
        if args.dataset.view_obs_horizon == 1:
            # Extract only the last timestep for view_model
            # Flow tokens: extract last timestep before reshaping
            input_flow_dim = get_condition_input_flow_dim(args)
            assert input_flow_dim <= D_flow, f"input_flow_dim ({input_flow_dim}) must be <= flow dim ({D_flow})"
            flow_tokens = point_flow[:, -1:, :, :input_flow_dim].permute(0, 2, 1, 3).reshape(B_flow, N_flow, -1)
            
            # Proprioception tokens: extract last timestep
            proprio_tokens_for_model = proprioception[:, -1:, :]  # [B, 1, dim]
            
            # Viewpoint tokens: extract last timestep
            viewpoint_tokens = viewpoint[:, -1:, :]  # [B, 1, dim]
            
            # All masks should be None when input is a single timestep
            flow_mask_for_model = None
            
        else:
            # Use all history
            input_flow_dim = get_condition_input_flow_dim(args)
            assert input_flow_dim <= D_flow, f"input_flow_dim ({input_flow_dim}) must be <= flow dim ({D_flow})"
            flow_tokens = point_flow[:, :, :, :input_flow_dim].permute(0, 2, 1, 3).reshape(B_flow, N_flow, -1)
            
            proprio_tokens_for_model = proprioception
            
            # viewpoint: [bs, h, dim] -> [bs, h, dim]   
            
            viewpoint_tokens = viewpoint
            
            flow_mask_for_model = flow_mask
        
        # Add condition flag to tokens based on use_condition
        # flow_tokens: [B, N, D] -> [B, N, D+1]
        if flow_tokens is not None:
            B_flow_tokens, N_flow_tokens, D_flow_tokens = flow_tokens.shape
            assert B == B_flow_tokens, f"B: {B}, B_flow_tokens: {B_flow_tokens}"
            if use_condition:
                condition_flag = torch.ones(B_flow_tokens, N_flow_tokens, 1, dtype=flow_tokens.dtype, device=flow_tokens.device)
            else:
                # Unconditional: use zero tokens and append 0 flag
                flow_tokens = torch.zeros(B_flow_tokens, N_flow_tokens, D_flow_tokens, dtype=flow_tokens.dtype, device=flow_tokens.device)
                condition_flag = torch.zeros(B_flow_tokens, N_flow_tokens, 1, dtype=flow_tokens.dtype, device=flow_tokens.device)
            flow_tokens = torch.cat([flow_tokens, condition_flag], dim=-1)
        
        # proprio_tokens_for_model: [B, H, D] -> [B, H, D+1]
        if proprio_tokens_for_model is not None:
            B_proprio, H_proprio, D_proprio = proprio_tokens_for_model.shape
            assert B == B_proprio, f"B: {B}, B_proprio: {B_proprio}"
            if use_condition:
                condition_flag = torch.ones(B_proprio, H_proprio, 1, dtype=proprio_tokens_for_model.dtype, device=proprio_tokens_for_model.device)
            else:
                # Unconditional: use zero tokens and append 0 flag
                proprio_tokens_for_model = torch.zeros(B_proprio, H_proprio, D_proprio, dtype=proprio_tokens_for_model.dtype, device=proprio_tokens_for_model.device)
                condition_flag = torch.zeros(B_proprio, H_proprio, 1, dtype=proprio_tokens_for_model.dtype, device=proprio_tokens_for_model.device)
            proprio_tokens_for_model = torch.cat([proprio_tokens_for_model, condition_flag], dim=-1)
        
        # viewpoint_tokens: [B, H, D] -> [B, H, D+1]
        if viewpoint_tokens is not None:
            B_viewpoint, H_viewpoint, D_viewpoint = viewpoint_tokens.shape
            assert B == B_viewpoint, f"B: {B}, B_viewpoint: {B_viewpoint}"
            if use_condition:
                condition_flag = torch.ones(B_viewpoint, H_viewpoint, 1, dtype=viewpoint_tokens.dtype, device=viewpoint_tokens.device)
            else:
                # Unconditional: use zero tokens and append 0 flag
                viewpoint_tokens = torch.zeros(B_viewpoint, H_viewpoint, D_viewpoint, dtype=viewpoint_tokens.dtype, device=viewpoint_tokens.device)
                condition_flag = torch.zeros(B_viewpoint, H_viewpoint, 1, dtype=viewpoint_tokens.dtype, device=viewpoint_tokens.device)
            viewpoint_tokens = torch.cat([viewpoint_tokens, condition_flag], dim=-1)
            
        
        flow_cond, proprio_cond, viewpoint_cond, future_action_cond, _ = model.adapt_conditions(
            flow_tokens=flow_tokens,
            proprio_tokens=proprio_tokens_for_model,
            viewpoint_tokens=viewpoint_tokens,  # [B, h, dim] or [B, 1, dim]
            future_action_tokens=None,
            action_tokens=None
        )  # flow_cond: [B, N, hidden_size], proprio_cond: [B, h, hidden_size] or [B, 1, hidden_size], viewpoint_cond: [B, h, hidden_size] or [B, 1, hidden_size], future_action_cond: None
        
        # Call model.forward (CustomRDT)
        model_output = model.model(
            x=action_input,  # [B, horizon, hidden_size]
            t=torch.full((temp_batch_size,), t, dtype=torch.long, device=point_flow.device),  # [B,]
            flow_c=flow_cond,  # [B, N, hidden_size]
            proprio_c=proprio_cond,  # [B, h, hidden_size] or [B, 1, hidden_size]
            viewpoint_c=viewpoint_cond,  # [B, h, hidden_size] or [B, 1, hidden_size]
            future_action_c=future_action_cond,
            flow_mask=flow_mask_for_model,
            proprio_mask=None,
            viewpoint_mask=None,
            future_action_mask=None,
        )
        
        # Handle output based on predict_flow or predict_separate_flow flag
        if model.predict_flow:
            pred_action, pred_flow = model_output
            # Concat for noise_pred (to match latents dimension)
            noise_pred = torch.cat([pred_action, pred_flow], dim=-1)  # [B, horizon, action_dim + flow_output_dim]
        elif model.predict_separate_flow:
            # For predict_separate_flow, model returns only action (flow is reconstructed separately)
            noise_pred = model_output  # [B, horizon, action_dim]
        else:
            noise_pred = model_output
        # noise_pred: [B, horizon, action_dim] or [B, horizon, action_dim + flow_output_dim]


        # parallelized version
        if i < len(timesteps) - 1:
            # ----------------------------------------------------------
            # === (1) expand the batch to create all duplicate candidates at once ===
            # original latents: shape (temp_batch_size, horizon, obs_dim)
            

            # latents_expanded: [B * DUP, horizon, obs_dim]
            latents_expanded = latents.unsqueeze(0).expand(DUP, *latents.shape)
            latents_expanded = latents_expanded.reshape(B * DUP, input_horizon, ND)

            # noise_pred_expanded: [B * DUP, horizon, obs_dim]
            noise_pred_expanded = noise_pred.unsqueeze(0).expand(DUP, *noise_pred.shape)
            noise_pred_expanded = noise_pred_expanded.reshape(B * DUP, input_horizon, ND)


            with torch.no_grad():
                latents_candidate = custom_ddim_step(
                    args,
                    noise_scheduler,
                    num_inference_steps,
                    noise_pred_expanded,   # shape [B*DUP, horizon, obs_dim]
                    t,
                    latents_expanded,      # same shape
                    eta=args.svdd.eta,
                )

                
            # === (2) recompute noise_pred for the next step (t+1) ===
            next_t = timesteps[i+1]
            
            fix_mask_expanded = fix_mask.unsqueeze(0).expand(DUP, *fix_mask.shape)  # [DUP, B,  H, dim]
            fix_mask_expanded = fix_mask_expanded.reshape(B * DUP, input_horizon, ND) # [B*DUP, H, dim]
            prior_expanded = prior.unsqueeze(0).expand(DUP, *prior.shape) # [DUP, B, H, dim]
            prior_expanded = prior_expanded.reshape(B * DUP, input_horizon, ND) # [B*DUP, H, dim]

            if args.use_fix_mask and fix_mask_expanded is not None and prior_expanded is not None:
                latents_candidate = latents_candidate * (1. - fix_mask_expanded) + prior_expanded * fix_mask_expanded


            # Convert inputs for CustomRDTRunner's model.forward (expanded batch)
            # latents_candidate: [B*DUP, horizon, action_dim] or [B*DUP, horizon, action_dim + flow_output_dim] -> adapt to hidden_size
            action_input_candidate = model.action_adaptor(latents_candidate)  # [B*DUP, horizon, hidden_size]
            
            # Convert point_flow_expanded to flow_tokens: [B*DUP, h, N, D] -> [B*DUP, N, h*D] for point-wise positional embedding
            B_flow_cand, H_flow_cand, N_flow_cand, D_flow_cand = point_flow_expanded.shape
            
            # Prepare tokens for candidate (same logic as main tokens)
            if args.dataset.view_obs_horizon == 1:
                # Extract only the last timestep for view_model
                input_flow_dim = get_condition_input_flow_dim(args)
                assert input_flow_dim <= D_flow_cand, f"input_flow_dim ({input_flow_dim}) must be <= flow dim ({D_flow_cand})"
                flow_tokens_candidate = point_flow_expanded[:, -1:, :, :input_flow_dim].permute(0, 2, 1, 3).reshape(B_flow_cand, N_flow_cand, -1)
                
                # Proprioception tokens: extract last timestep
                proprio_tokens_candidate = proprioception_expanded[:, -1:, :]  # [B*DUP, 1, dim]
                
                # Viewpoint tokens: extract last timestep
                viewpoint_tokens_candidate = viewpoint_expanded[:, -1:, :]  # [B*DUP, 1, dim]
                
                # All masks should be None when input is a single timestep
                flow_mask_candidate = None
            else:
                # Use all history
                input_flow_dim = get_condition_input_flow_dim(args)
                assert input_flow_dim <= D_flow_cand, f"input_flow_dim ({input_flow_dim}) must be <= flow dim ({D_flow_cand})"
                flow_tokens_candidate = point_flow_expanded[:, :, :, :input_flow_dim].permute(0, 2, 1, 3).reshape(B_flow_cand, N_flow_cand, -1)
                
                # Create flow_mask based on visibility (last dimension of point_flow_expanded)
                # If D_flow_cand >= 4, the last dimension is visibility (0 > visible, <= 0 invisible)
                # flow_mask: (B*DUP, N) where True means visible (masking=true), False means invisible (masking=false)
                if D_flow_cand == 4 or use_pixel_flow_for_robot_policy(args):
                    if args.training.use_masking: # flow: 3dim, but masking is applied
                        flow_mask_candidate = visibility_expanded.bool()  # [B*DUP, N] - True if visible in the most recent timestep
                        if not torch.any(flow_mask_candidate, dim=1).all():
                            print(f'WARNING: at least one should be visible(true) along N axis. We temporarily set arbitrary one to True')
                            flow_mask_candidate[:, torch.randint(0, N_flow_cand, (1,), device=point_flow.device)] = True
                        flow_mask_candidate = apply_random_flow_mask_count(
                            flow_mask_candidate,
                            get_random_flow_mask_count(args),
                        )
                        assert torch.any(flow_mask_candidate, dim=1).all(), f"at least one should be visible(true) along N axis, flow_mask_candidate : {flow_mask_candidate}"
                    else:
                        flow_mask_candidate = None
                else:
                    flow_mask_candidate = None
                
                proprio_tokens_candidate = proprioception_expanded
                
                # viewpoint: [B*DUP, h, dim] -> [B*DUP, h, dim]
                viewpoint_tokens_candidate = viewpoint_expanded
            
            # Add condition flag to candidate tokens based on use_condition
            # flow_tokens_candidate: [B*DUP, N, D] -> [B*DUP, N, D+1]
            if flow_tokens_candidate is not None:
                B_flow_cand, N_flow_cand, D_flow_cand = flow_tokens_candidate.shape
                if use_condition:
                    condition_flag = torch.ones(B_flow_cand, N_flow_cand, 1, dtype=flow_tokens_candidate.dtype, device=flow_tokens_candidate.device)
                else:
                    # Unconditional: use zero tokens and append 0 flag
                    flow_tokens_candidate = torch.zeros(B_flow_cand, N_flow_cand, D_flow_cand, dtype=flow_tokens_candidate.dtype, device=flow_tokens_candidate.device)
                    condition_flag = torch.zeros(B_flow_cand, N_flow_cand, 1, dtype=flow_tokens_candidate.dtype, device=flow_tokens_candidate.device)
                flow_tokens_candidate = torch.cat([flow_tokens_candidate, condition_flag], dim=-1)
            
            # proprio_tokens_candidate: [B*DUP, H, D] -> [B*DUP, H, D+1]
            if proprio_tokens_candidate is not None:
                B_proprio_cand, H_proprio_cand, D_proprio_cand = proprio_tokens_candidate.shape
                if use_condition:
                    condition_flag = torch.ones(B_proprio_cand, H_proprio_cand, 1, dtype=proprio_tokens_candidate.dtype, device=proprio_tokens_candidate.device)
                else:
                    # Unconditional: use zero tokens and append 0 flag
                    proprio_tokens_candidate = torch.zeros(B_proprio_cand, H_proprio_cand, D_proprio_cand, dtype=proprio_tokens_candidate.dtype, device=proprio_tokens_candidate.device)
                    condition_flag = torch.zeros(B_proprio_cand, H_proprio_cand, 1, dtype=proprio_tokens_candidate.dtype, device=proprio_tokens_candidate.device)
                proprio_tokens_candidate = torch.cat([proprio_tokens_candidate, condition_flag], dim=-1)
            
            # viewpoint_tokens_candidate: [B*DUP, H, D] -> [B*DUP, H, D+1]
            if viewpoint_tokens_candidate is not None:
                B_viewpoint_cand, H_viewpoint_cand, D_viewpoint_cand = viewpoint_tokens_candidate.shape
                if use_condition:
                    condition_flag = torch.ones(B_viewpoint_cand, H_viewpoint_cand, 1, dtype=viewpoint_tokens_candidate.dtype, device=viewpoint_tokens_candidate.device)
                else:
                    # Unconditional: use zero tokens and append 0 flag
                    viewpoint_tokens_candidate = torch.zeros(B_viewpoint_cand, H_viewpoint_cand, D_viewpoint_cand, dtype=viewpoint_tokens_candidate.dtype, device=viewpoint_tokens_candidate.device)
                    condition_flag = torch.zeros(B_viewpoint_cand, H_viewpoint_cand, 1, dtype=viewpoint_tokens_candidate.dtype, device=viewpoint_tokens_candidate.device)
                viewpoint_tokens_candidate = torch.cat([viewpoint_tokens_candidate, condition_flag], dim=-1)

            flow_cond_candidate, proprio_cond_candidate, viewpoint_cond_candidate, future_action_cond_candidate, _ = model.adapt_conditions(
                flow_tokens=flow_tokens_candidate,
                proprio_tokens=proprio_tokens_candidate,
                viewpoint_tokens=viewpoint_tokens_candidate,  # [B*DUP, h, dim] or [B*DUP, 1, dim]
                future_action_tokens=None,
                action_tokens=None
            )  # flow_cond: [B*DUP, N, hidden_size], proprio_cond: [B*DUP, h, hidden_size] or [B*DUP, 1, hidden_size], viewpoint_cond: [B*DUP, h, hidden_size] or [B*DUP, 1, hidden_size], future_action_cond: None
            
            # Call model.forward (CustomRDT)
            model_output_candidate = model.model(
                x=action_input_candidate,  # [B*DUP, horizon, hidden_size]
                t=torch.full((B*DUP,), next_t, dtype=torch.long, device=point_flow.device),  # [B*DUP,]
                flow_c=flow_cond_candidate,  # [B*DUP, N, hidden_size]
                proprio_c=proprio_cond_candidate,  # [B*DUP, h, hidden_size] or [B*DUP, 1, hidden_size]
                viewpoint_c=viewpoint_cond_candidate,  # [B*DUP, h, hidden_size] or [B*DUP, 1, hidden_size]
                future_action_c=future_action_cond_candidate,
                flow_mask=flow_mask_candidate,
                proprio_mask=None,
                viewpoint_mask=None,
                future_action_mask=None,
            )
            
            # Handle output based on predict_flow or predict_separate_flow flag
            if model.predict_flow:
                pred_action_candidate, pred_flow_candidate = model_output_candidate
                # Concat for noise_pred_candidate (to match latents_candidate dimension)
                noise_pred_candidate = torch.cat([pred_action_candidate, pred_flow_candidate], dim=-1)  # [B*DUP, horizon, action_dim + flow_output_dim]
            elif model.predict_separate_flow:
                # For predict_separate_flow, model returns only action (flow is reconstructed separately)
                noise_pred_candidate = model_output_candidate  # [B*DUP, horizon, action_dim]
            else:
                noise_pred_candidate = model_output_candidate
            # noise_pred_candidate: [B*DUP, horizon, action_dim] or [B*DUP, horizon, action_dim + flow_output_dim]

            # === (3) compute weights ===
            reward_dict_candidate = None
            if task == 'active_vision':
                weights_candidate, reward_dict_candidate = calculate_weight_active_vision(
                    args, noise_scheduler,
                    latents_candidate,  # x_t
                    noise_pred_candidate, # [B*DUP, horizon, dim]
                    next_t,
                    dataset,
                    task_kwargs,
                    fix_mask_expanded,
                    prior_expanded,
                    stats,
                    mc_viewpoints_unnorm,
                )
            else:
                weights_candidate = calculate_weight(
                    args, noise_scheduler,
                    latents_candidate,  # x_t
                    noise_pred_candidate,
                    next_t,
                    dataset,
                    pt_max,
                    pt_min,
                )
            # weights_candidate: shape [B*DUP]
            weights_candidate = weights_candidate.view(DUP, B).cpu().numpy()
            
            # Store reward_dict_candidate in task_kwargs for later retrieval
            if reward_dict_candidate is not None:
                if 'reward_dicts' not in task_kwargs:
                    task_kwargs['reward_dicts'] = []
                task_kwargs['reward_dicts'].append(reward_dict_candidate) # list of dict of [B*DUP]

            # latents_candidate: [B*DUP, horizon, obs_dim] -> [DUP, B, horizon, obs_dim]
            latents_candidate = latents_candidate.view(DUP, B, input_horizon, ND).cpu().numpy()

            # === (4) pick the best candidate with argmax ===
            index_chosen = np.argmax(weights_candidate, axis=0)  # shape [B]
            new_latents = []
            for idx_b in range(B):
                best_dup = index_chosen[idx_b]  # which duplicate
                new_latents.append(latents_candidate[best_dup, idx_b])  # shape [horizon, obs_dim]

            new_latents = np.stack(new_latents, axis=0)  # shape [B, horizon, obs_dim]

            latents = torch.from_numpy(new_latents).to(point_flow.device)
            # ----------------------------------------------------------


        else:  #If we are in the last step 
            with torch.no_grad():
                latents = custom_ddim_step(
                                        args,
                                        noise_scheduler,
                                        num_inference_steps,
                                        noise_pred,
                                        t,
                                        latents,
                                        eta=args.svdd.eta,
                                    )
                
                
                if args.use_fix_mask:
                    latents = (1. - fix_mask) * latents + fix_mask * prior

        
    print(f'SVDD M candidate : {DUP}, K step denosing time taken : {time.time() - start} s')
    

    B, H, ND_full = latents.shape
    # If predict_flow is True, extract only action part from latents
    # Note: predict_separate_flow doesn't need extraction since latents are already action_dim only
    if model.predict_flow:
        latents = latents[:, :, :ND_action]  # [B, H, action_dim]
        ND = ND_action
    else:
        ND = ND_full
    from scipy.spatial.transform import Rotation as R
    if task == 'active_vision':
        # NOTE: currently, assume that we do not normalize the viewpoint data
        x0_svdd = latents.detach().cpu().numpy() # [bs, h, ND]
        if args.training.use_ptp:
            x0_svdd = x0_svdd[:, args.dataset.obs_horizon-1:, :]
        assert x0_svdd.shape == (temp_batch_size, args.dataset.pred_horizon, ND)
        
        # Compute final reward_dict for x0_svdd [B, horizon, ND]
        
        # Convert x0_svdd to 9D representation for retargeting and reward computation
        # x0_svdd is already in normalized form, need to unnormalize and convert to 9D
        B_final, H_final, ND_final = x0_svdd.shape
        
        # Unnormalize x0_svdd
        x0_svdd_unnorm = x0_svdd.copy()
        if not args.use_relative:
            x0_svdd_unnorm = unnormalize_data(x0_svdd_unnorm.reshape(-1, ND_final), stats=stats["view_actions"], type=args.dataset.norm_type)
            x0_svdd_unnorm = x0_svdd_unnorm.reshape(B_final, H_final, -1) # [B, h, dim]
        else:  # relative
            x0_svdd_unnorm = unnormalize_data(x0_svdd_unnorm.reshape(-1, ND_final), stats=stats["delta_view_action_for_uncondition"], type=args.dataset.norm_type)
            x0_svdd_unnorm = x0_svdd_unnorm.reshape(B_final, H_final, -1) # [B, h, dim(6)]
            
            # Convert relative to absolute
            relative_pos = x0_svdd_unnorm[..., :3].copy() # [B, h, 3]
            relative_rotvec = x0_svdd_unnorm[..., 3:6].copy() # [B, h, 3] # axis angle
            relative_rot = R.from_rotvec(relative_rotvec.reshape(-1, 3)).as_matrix().reshape(B_final, H_final, 3, 3) # [B, h, 3, 3]
            
            # Get current_viewpoint from task_kwargs (last viewpoint in history)
            current_viewpoint = np.tile(mc_viewpoints_unnorm[None], (B_final, 1)) # [B, 9]
            
            current_viewpoint_pos = current_viewpoint[..., :3].copy() # [B, 3]
            current_viewpoint_ortho6d = current_viewpoint[..., 3:9].copy() # [B, 6]
            current_viewpoint_rot = compute_rotation_matrix_from_ortho6d(current_viewpoint_ortho6d) # [B, 3, 3]
            
            absolute_pos = current_viewpoint_pos[:, None, :] + np.cumsum(relative_pos, axis=1) # [B, h, 3]
            # Cumulative rotation
            cumulative_rot_matrices = np.zeros_like(relative_rot)  # [B, h, 3, 3]
            for b in range(B_final):
                rot_objects = [R.from_matrix(relative_rot[b, i]) for i in range(H_final)]
                cumulative_rots = list(accumulate(rot_objects, lambda acc, r: acc * r, initial=R.identity()))[1:]
                cumulative_rot_matrices[b] = np.array([r.as_matrix() for r in cumulative_rots])
            absolute_rot = np.einsum('bij,btjk->btik', current_viewpoint_rot, cumulative_rot_matrices) # [B, h, 3, 3]
            absolute_ortho6d = compute_ortho6d_from_rotation_matrix(absolute_rot.reshape(-1, 3, 3)).reshape(B_final, H_final, 6) # [B, h, 6]
            x0_svdd_unnorm = np.concatenate([absolute_pos, absolute_ortho6d], axis=-1) # [B, h, 9]
        
        x0_svdd_9d = x0_svdd_unnorm
        
        if return_task_kwargs and task_kwargs is not None:
            # Compute reward for final x0_svdd [B, H, 9]
            device = latents.device
            final_reward_dict = compute_reward(args, x0_svdd_9d.copy(), task_kwargs, device=device)
            # final_reward_dict contains rewards with shape [B]
            task_kwargs['x0_svdd'] = x0_svdd_9d.copy()
            task_kwargs['final_reward_dict'] = final_reward_dict

        
        # Apply retargeting if enabled
        if use_retargeting and task_kwargs is not None: # only used for reward computation
            # Use view_robot_viz (episode_view_robot_viz) instead of robot_viz
            view_robot_viz = task_kwargs.get('view_robot_viz')
            T_B_M_view = task_kwargs.get('T_B_M_view')
            T_M_C = task_kwargs.get('T_mc_transformation')
            use_visibility_aware_retargeting = task_kwargs.get('use_visibility_aware_retargeting', False)
            x0_svdd_9d_retargeted = retargeting(view_robot_viz, x0_svdd_9d, T_B_M_view=T_B_M_view, T_M_C=T_M_C, \
                use_visibility_aware_retargeting=use_visibility_aware_retargeting, task_kwargs=task_kwargs)
            task_kwargs['x0_svdd_retargeted'] = x0_svdd_9d_retargeted.copy() # marker coordinate, unnormalized (T_MC)
            final_reward_dict_retargeted = compute_reward(args, x0_svdd_9d_retargeted.copy(), task_kwargs, device=device)
            # final_reward_dict contains rewards with shape [B]
            task_kwargs['final_reward_dict_retargeted'] = final_reward_dict_retargeted
            
    else:
        N, D = args.dataset.num_dift_points, args.dataset.flow_dim
        x0_svdd_norm = latents.reshape(B, H, N, D).detach().cpu().numpy() # [bs, h, N, D]

        if flow_norm_type == 'min_max':
    
            # roughly [-1, 1] ->  [0, 1]
            x0_svdd = dataset.unnormalize_transform(x0_svdd_norm)

            # roughly [0, 1] ->  raw values
            x0_svdd = x0_svdd * (pt_max[None, None, None, :] - pt_min[None, None, None, :]) + pt_min[None, None, None, :]
        elif flow_norm_type == 'mean_std':
            raise NotImplementedError
        
        assert x0_svdd.shape == (temp_batch_size, args.dataset.pred_horizon, N, D)

    if return_task_kwargs:
        return x0_svdd, task_kwargs
    else:
        return x0_svdd, None
