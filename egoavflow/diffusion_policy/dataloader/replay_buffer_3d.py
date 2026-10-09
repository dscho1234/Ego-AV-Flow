import concurrent.futures
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
import zarr
from tqdm import tqdm

from egoavflow.common.imagecodecs_numcodecs import register_codecs
from scipy.spatial.transform import Rotation as R
from egoavflow.utils import compute_ortho6d_from_rotation_matrix, compute_rotation_matrix_from_ortho6d
register_codecs()


# --------- Coordinate transformation helpers ----------
def inv_T(T: np.ndarray) -> np.ndarray:
    """Inverse of SE(3) transformation matrix"""
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3,3]  = -R.T@t
    return Ti

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

def convert_to_relative(se3_matrices, ref_idx):
    se3_matrices = se3_matrices.copy()
    # SE(3) matrices [B, H, 4, 4]
    
    # Get inverse of ref_idx transformation for each batch [B, 4, 4]
    ref_transforms = se3_matrices[:, ref_idx]  # [B, 4, 4]
    ref_inv = np.linalg.inv(ref_transforms)  # [B, 4, 4]
    
    # Apply relative transformation: T_rel = T_ref_inv @ T_current
    # Broadcasting: [B, 4, 4] @ [B, H, 4, 4] -> [B, H, 4, 4]
    relative_se3 = np.einsum('bij,bhjk->bhik', ref_inv, se3_matrices)
    
    # Set ref_idx index to identity
    relative_se3[:, ref_idx] = np.eye(4)
    
    return relative_se3


class ReplayBuffer:
    def __init__(
        self,
        data_path,
        load_camera_ids=[],
        camera_resize_shape=[],
        max_episode=None,
        max_workers=32,
        backward_episode_len = 10000000,
    ) -> None:
        self.data_path = data_path
        self.load_camera_ids = load_camera_ids
        self.camera_resize_shape = camera_resize_shape
        self.max_workers = max_workers
        self.max_episode = max_episode
        self.backward_episode_len = backward_episode_len
        self.initiate_memory_buffer()
        self.load_data_to_memory()

    def initiate_memory_buffer(self):
        self.memory_buffer = defaultdict(list)

    def load_data_to_memory(self):
        load_episode_num = 0
        for path in self.data_path:
            root = zarr.open(path, mode="a")
            episodes = list(root.group_keys())
            for episode in episodes:
                self.memory_buffer["action"].append(
                    self.load_low_dim_data(root, osp.join(episode, "action"))
                )
                self.memory_buffer["proprioception"].append(
                    self.load_low_dim_data(root, osp.join(episode, "proprioception"))
                )
                for camera_ids in self.load_camera_ids: # [0]
                    cam_name = f"camera_{camera_ids}"
                    self.memory_buffer[cam_name].append(
                        self.load_visual_data(root, osp.join(episode, cam_name, "rgb"))
                    )
                load_episode_num += 1
                if (
                    self.max_episode is not None
                    and load_episode_num >= self.max_episode
                ):
                    break

        self.eps_end = np.cumsum([len(x) for x in self.memory_buffer["action"]])
        for k, v in self.memory_buffer.items():
            self.memory_buffer[k] = np.concatenate(v)

    def load_low_dim_data(self, root, low_dim_path, padding_size = None):
        if padding_size is not None:
            data_sequence = root[low_dim_path][-self.backward_episode_len:].astype(np.float32) # [T, dim, dim...]
            last_time_step = data_sequence[-1, :]
            padding = np.repeat(
                last_time_step[np.newaxis, :],
                padding_size - data_sequence.shape[0],
                axis=0,
            )
            return np.concatenate([data_sequence, padding], axis=0)[
                None, :
            ]  # (1,padding_size, dim)
        else:
            return root[low_dim_path][-self.backward_episode_len:].astype(np.float32)

    def load_visual_data(self, root, visual_path, dim=3):
        visual_shape = root[visual_path].shape
        np_arr_shape = (
            (visual_shape[0], *self.camera_resize_shape, dim)
            if self.camera_resize_shape
            else visual_shape
        )
        np_arr = np.zeros(np_arr_shape, dtype=np.uint8)

        def load_img(zarr_arr, visual_path, index, np_arr):
            try:
                if self.camera_resize_shape:
                    np_arr[index] = cv2.resize(
                        zarr_arr[visual_path][index], self.camera_resize_shape
                    )
                else:
                    np_arr[index] = zarr_arr[visual_path][index]
                return True
            except Exception as e:
                print(e)
                return False

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = set()
            for i in range(visual_shape[0]):
                futures.add(executor.submit(load_img, root, visual_path, i, np_arr))

            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError("Failed to load image!")

        return np_arr[-self.backward_episode_len:]

    def __repr__(self) -> str:
        rep = ""
        for k, v in self.memory_buffer.items():
            rep += f"{k}, {v.shape}\n"
        rep += f"eps_end, {self.eps_end}\n"
        return rep

    def __getitem__(self, key):
        return self.memory_buffer[key]

    def remove_key(self, key):
        del self.memory_buffer[key]


class QuestPointTrackingReplayBuffer3D(ReplayBuffer):
    def __init__(
        self,
        data_path,
        point_tracking_img_size,
        load_camera_ids=[],
        point_tracking_camera_id=0,
        camera_resize_shape=[],
        max_episode=None,
        downsample_rate=1,
        max_workers=32,
        max_episode_len=None,
        padding_size=160,
        is_sam=False,
        
        flow_dim = 3,
        use_dift = False,
        use_dift_point_tracking = False,
        use_estimated_extrinsics = False,
        use_marker_coordinate = False,
        backward_episode_len = 10000000,
        droid = False, 
        use_relative = False,
        # augmentation parameters
        augment_se3 = False,
        num_aug_per_episode = 1,
        se3_translation_range = 0.5,
        se3_rotation_range = 30, 
        num_points_for_object = None, 
    ) -> None:
        self.point_tracking_camera_id = point_tracking_camera_id
        self.point_tracking_img_size = point_tracking_img_size
        self.downsample_rate = downsample_rate
        self.max_episode_len = max_episode_len
        self.padding_size = padding_size
        self.is_sam = is_sam
        self.droid = droid
        
        self.data_path_offset = [0]
        self.flow_dim = flow_dim
        self.use_dift = use_dift
        self.use_dift_point_tracking = use_dift_point_tracking
        self.use_estimated_extrinsics = use_estimated_extrinsics
        self.use_marker_coordinate = use_marker_coordinate
        self.use_relative = use_relative
        self.augment_se3 = augment_se3
        self.num_aug_per_episode = num_aug_per_episode
        self.se3_translation_range = se3_translation_range
        self.se3_rotation_range = se3_rotation_range
        self.num_points_for_object = num_points_for_object
        

        super().__init__(
            data_path, load_camera_ids, camera_resize_shape, max_episode, max_workers, backward_episode_len
        )
    

    def transform_points_to_marker_coords(self, points_camera: np.ndarray, T_mc: np.ndarray) -> np.ndarray:
        """
        Transform points from camera coordinates to marker coordinates using vectorized operations
        
        Args:
            points_camera: (N, 3) points in camera coordinates
            T_mc: (N, 4, 4) or (4, 4) marker-to-camera transformation matrix(es)
        
        Returns:
            points_marker: (N, 3) points in marker coordinates
        """
        N = points_camera.shape[0]
        
        # Handle single transformation matrix case
        if T_mc.ndim == 2:
            T_mc = np.tile(T_mc[None, :, :], (N, 1, 1))
        
        # Convert points to homogeneous coordinates (N, 4)
        points_homo = np.column_stack([points_camera, np.ones(N)])
        
        # Apply transformation for each point
        points_marker_homo = np.zeros((N, 4))
        for i in range(N):
            points_marker_homo[i] = T_mc[i] @ points_homo[i]
        
        return points_marker_homo[:, :3]  # (N, 3)


    def transform_ortho6d_to_marker_coords(self, ortho6d_camera: np.ndarray, T_mc: np.ndarray) -> np.ndarray:
        """
        Transform ortho6d from camera coordinates to marker coordinates
        
        Args:
            ortho6d_camera: (N, 6) ortho6d in camera coordinates
            T_mc: (N, 4, 4) marker-to-camera transformation matrices
        
        Returns:
            ortho6d_marker: (N, 6) ortho6d in marker coordinates
        """
        N = ortho6d_camera.shape[0]
        ortho6d_marker = np.zeros((N, 6))
        
        for i in range(N):
            # Convert ortho6d to rotation matrix
            R_camera = compute_rotation_matrix_from_ortho6d(ortho6d_camera[i][None])[0]
            
            # Transform rotation matrix
            R_marker = T_mc[i][:3, :3] @ R_camera
            
            # Convert back to ortho6d
            ortho6d_marker[i] = compute_ortho6d_from_rotation_matrix(R_marker[None])[0]
        
        return ortho6d_marker


    def generate_se3_augmentation_transform(self) -> np.ndarray:
        """
        Generate SE(3) augmentation transform relative to initial object pose.
        Rotation axis: 50% chance for x-axis, 50% chance for spherical coordinate-based axis
        (x-axis as z-axis in spherical coordinate, generating axes away from x-axis).
        
        Args:
            initial_object_pose: (4, 4) SE(3) matrix representing initial object pose
        
        Returns:
            se3_transform: (4, 4) SE(3) transformation matrix
        """
        # Generate random translation: uniform in [-translation_range, translation_range]
        random_translation = np.random.uniform(
            -self.se3_translation_range, 
            self.se3_translation_range, 
            size=(3,)
        )
        
        # 50% chance for x-axis rotation, 50% chance for spherical coordinate-based rotation
        use_x_axis = np.random.choice([True, False], p=[0.5, 0.5])
        
        if use_x_axis:
            # x-axis rotation
            random_axis = np.array([1.0, 0.0, 0.0])
        else:
            # NOTE: hemisphere whose normal axis is x-axis
            phi = np.random.uniform(-np.pi/2, np.pi/2)  # azimuthal angle
            theta = np.random.uniform(0, np.pi) # np.arccos(np.random.uniform(-1, 1))  # polar angle (uniform in cos(theta))
            
            random_axis = np.array([
                np.cos(theta),  # x component (along x-axis)
                np.sin(theta) * np.cos(phi),  # y component
                np.sin(theta) * np.sin(phi)   # z component
            ])
        
        # Generate random angle: uniform in [-rotation_range, rotation_range]
        random_angle = np.random.uniform(np.deg2rad(-self.se3_rotation_range), np.deg2rad(self.se3_rotation_range))
        
        # Convert axis-angle to rotation matrix using Rodrigues' formula
        K = np.array([[0, -random_axis[2], random_axis[1]],
                     [random_axis[2], 0, -random_axis[0]],
                     [-random_axis[1], random_axis[0], 0]])
        
        random_rotation = (np.eye(3) + 
                          np.sin(random_angle) * K + 
                          (1 - np.cos(random_angle)) * (K @ K))
        
        # Construct SE(3) transformation matrix
        se3_transform = np.eye(4)
        se3_transform[:3, :3] = random_rotation
        se3_transform[:3, 3] = random_translation
        
        # Transform is relative to initial_object_pose
        # The transform should be applied in the frame of initial_object_pose
        # So we need to: initial_object_pose @ se3_transform @ inv(initial_object_pose) for points
        # But for SE(3) matrices, we apply: se3_transform @ T
        return se3_transform
        

    def compute_delta_view_action(self, view_actions_se3):
        translation_view_action = view_actions_se3[:, :3, 3].copy() # [T, 3]
        delta_translation_view_action = translation_view_action[1:] - translation_view_action[:-1] #[T-1, 3]
        delta_translation_view_action = np.concatenate([delta_translation_view_action, np.zeros((1, 3))], axis=0) # [T, 3]
        # compute relative rotation matrix and compute delta axis angle
        # Method: Convert 6D to rotation matrix, compute relative rotation, then convert to axis-angle
        # 1. Get rotation matrices (already have from view_actions_se3)
        R_t = view_actions_se3[:, :3, :3].copy()  # [T, 3, 3]
        R_t_1 = R_t[1:]  # [T-1, 3, 3] - R_{t+1}
        R_t_0 = R_t[:-1]  # [T-1, 3, 3] - R_t
        
        # 2. Compute relative rotation: R_rel = R_t^T @ R_{t+1}
        R_rel = np.einsum('tij,tjk->tik', R_t_0.transpose(0, 2, 1), R_t_1)  # [T-1, 3, 3]
        
        # 3. Convert to axis-angle (log map) using scipy
        rotations = R.from_matrix(R_rel)
        axis_angle = rotations.as_rotvec()  # [T-1, 3] - axis-angle representation
        
        # Append zero rotation (0 degrees) for the last timestep to match length T
        delta_rotation_view_action = np.concatenate([axis_angle, np.zeros((1, 3))], axis=0)  # [T, 3]


        delta_view_action = np.concatenate([delta_translation_view_action, delta_rotation_view_action], axis=-1) # [T, 6]
        return delta_view_action

    def load_data_to_memory(self):
        if self.use_marker_coordinate:
            self.load_data_to_memory_marker_coordinate()
        else:
            self.load_data_to_memory_not_marker_coordinate()

    def load_data_to_memory_marker_coordinate(self):
        load_episode_num = 0
        for path in self.data_path:
            print(">> Loading data from: ", path)
            root = zarr.open(path, mode="a")
            episodes = list(root.group_keys())
            episodes = sorted(episodes, key=lambda x: int(x.split("_")[1]))
            print(f">> {len(episodes)} episodes num from {path} ")
            if len(episodes) == 0:
                raise ValueError("No episodes found!")
            
            path_load_episode_num = 0

            original_num_episodes = len(episodes)
            is_augmentation_episode = False

            # Initialize list to store initial object poses for SE(3) augmentation
            episode_initial_object_pose = []

            if self.augment_se3:
                # copy episodes N times
                episodes = episodes * self.num_aug_per_episode
                print(f">> Duplicated episodes {self.num_aug_per_episode} times. Total episodes: {len(episodes)}")


            for i, episode in tqdm(enumerate(episodes)):
                if i >= original_num_episodes and self.augment_se3:
                    is_augmentation_episode = True
                    
                episode_length = len(
                    self.load_low_dim_data(root, osp.join(episode, "action"))[
                        :: self.downsample_rate
                    ]
                )
                if self.max_episode_len and episode_length > self.max_episode_len:
                    print(f"episode {episode} exceeds max episode length")
                else:
                    print(f"episode {episode} length: {episode_length}")

                # Load transformation matrix based on configuration
                # For now, use T_mc_opt as default (can be extended to support both options)
                T_mc_opt_name = "T_mc_opt" if not self.droid else "T_mc_opt_droid"
                T_mc_transformation = self.load_low_dim_data(root, osp.join(episode, T_mc_opt_name))[
                    :: self.downsample_rate
                ][: self.max_episode_len]  # (T, 4, 4)

                
                view_actions_se3 = T_mc_transformation.copy() # [T, 4, 4]
                translation_view_action = view_actions_se3[:, :3, 3].copy() # [T, 3]
                rotation_view_action = view_actions_se3[:, :3, :3].copy() # [T, 3, 3]
                # convert rotation matrix to euler
                rotation_view_action = compute_ortho6d_from_rotation_matrix(rotation_view_action)
                view_actions = np.concatenate([translation_view_action, rotation_view_action], axis=-1) # [T, 9]
                self.memory_buffer["view_actions"].append(view_actions)

                if self.use_relative: # tile last action
                    
                    
                    delta_view_action = self.compute_delta_view_action(view_actions_se3.copy())

                    self.memory_buffer["delta_view_action"].append(delta_view_action)


                    if self.augment_se3 and is_augmentation_episode: # This is only valid when cfg.use_relative is True
                        ref_idx = np.random.randint(0, len(view_actions_se3))
                        T_m_ref_cam = view_actions_se3[ref_idx].copy() # [4, 4]
                        # fisrt element is identity
                        view_actions_se3 = convert_to_relative(view_actions_se3.copy()[None], ref_idx)[0] # [T, 4, 4], ref cam coordinate
                        
                        # Check if the transformation is approximately equal using SE(3) distance
                        # Convert back to marker coordinate and compare with original
                        view_actions_se3_marker = np.einsum('ij,tjk->tik', T_m_ref_cam, view_actions_se3) # [T, 4, 4], marker coordinate
                        # Compute SE(3) distance for each timestep
                        
                        max_se3_distance = 0.0
                        
                        se3_dist = compute_se3_distance(view_actions_se3_marker, T_mc_transformation)
                        max_se3_distance = np.max(se3_dist)

                        # Use SE(3) distance threshold (typically 1e-4 to 1e-3 for numerical errors)
                        se3_distance_threshold = 1e-4
                        assert max_se3_distance < se3_distance_threshold, \
                            f"SE(3) transformation mismatch: max SE(3) distance = {max_se3_distance:.2e} (threshold = {se3_distance_threshold:.2e})"
                        
                        # Apply spherical coordinate-based rotation augmentation
                        # Generate random rotation axis in spherical coordinates
                        # Spherical coordinates: (theta, phi) where theta is polar angle, phi is azimuthal angle
                        phi = np.random.uniform(0, 2 * np.pi)  # azimuthal angle [0, 2π]
                        theta = np.arccos(np.random.uniform(-1, 1))  # polar angle [0, π] (uniform distribution on sphere)
                        
                        # Convert spherical coordinates to unit vector (rotation axis)
                        random_axis = np.array([
                            np.sin(theta) * np.cos(phi),  # x component
                            np.sin(theta) * np.sin(phi),  # y component
                            np.cos(theta)  # z component
                        ])
                        random_axis = random_axis / np.linalg.norm(random_axis)  # Normalize to unit vector
                        
                        # Generate random rotation angle: uniform in [-rotation_range, rotation_range]
                        random_angle = np.random.uniform(
                            np.deg2rad(-self.se3_rotation_range), 
                            np.deg2rad(self.se3_rotation_range)
                        )
                        
                        # Convert axis-angle to rotation matrix using Rodrigues' formula
                        K = np.array([[0, -random_axis[2], random_axis[1]],
                                    [random_axis[2], 0, -random_axis[0]],
                                    [-random_axis[1], random_axis[0], 0]])
                        
                        rotation_matrix = (np.eye(3) + 
                                        np.sin(random_angle) * K + 
                                        (1 - np.cos(random_angle)) * (K @ K))
                        
                        # Apply rotation to all timesteps in view_actions_se3
                        # For each timestep: R_new = rotation_matrix @ R_old
                        view_actions_se3[:, :3, :3] = np.einsum('ij,tjk->tik', rotation_matrix, view_actions_se3[:, :3, :3])

                        view_actions_se3_for_uncondition = np.einsum('ij,tjk->tik', T_m_ref_cam, view_actions_se3) # [T, 4, 4], marker coordinate

                        delta_view_action= self.compute_delta_view_action(view_actions_se3_for_uncondition.copy())
                        

                    self.memory_buffer["delta_view_action_for_uncondition"].append(delta_view_action.copy())


                episodic_T_mc_transformation = self.load_low_dim_data(root, osp.join(episode, T_mc_opt_name), padding_size = self.padding_size)[
                    :, ::self.downsample_rate
                ][:, :self.max_episode_len][0] # [T(padding), 4, 4]
                
                self.memory_buffer["episodic_T_mc_opt"].append(episodic_T_mc_transformation[None])

                # Load proprioception data and transform to marker coordinates
                proprioception_camera = self.load_low_dim_data(root, osp.join(episode, "proprioception"))[
                    :: self.downsample_rate
                ][: self.max_episode_len]  # (T, dim)
                
                # Extract xyz positions and euler angles from proprioception
                xyz_positions = proprioception_camera[:, :3]  # (T, 3)
                ortho6d = proprioception_camera[:, 3:9]  # (T, 6)
                
                # Transform proprioception positions and euler angles to marker coordinates
                proprioception_marker_xyz = self.transform_points_to_marker_coords(xyz_positions, T_mc_transformation)
                proprioception_marker_ortho6d = self.transform_ortho6d_to_marker_coords(ortho6d, T_mc_transformation)
                
                # Store se3_transform for this augmentation episode (to be reused for action and point tracking)
                se3_transform_aug = None
                initial_object_pose_aug = None
                apply_rotation_proprio = False
                apply_rotation_action = False
                
                if self.augment_se3 and is_augmentation_episode:
                    # Get the original episode index (modulo original_num_episodes)
                    original_episode_idx = (i - original_num_episodes) % original_num_episodes
                    initial_object_pose_aug = episode_initial_object_pose[original_episode_idx]
                    
                    # Generate SE(3) augmentation transform (same for all data in this episode)
                    se3_transform_aug = self.generate_se3_augmentation_transform()
                    
                    # Apply augmentation to proprioception (episode-level, vectorized, always apply both position and rotation)
                    T_proprio = len(proprioception_marker_xyz)
                    # Convert all timesteps to SE(3) representation
                    proprio_R_all = compute_rotation_matrix_from_ortho6d(proprioception_marker_ortho6d)  # [T, 3, 3]
                    proprio_T_all = np.zeros((T_proprio, 4, 4))
                    proprio_T_all[:, :3, :3] = proprio_R_all
                    proprio_T_all[:, :3, 3] = proprioception_marker_xyz
                    proprio_T_all[:, 3, 3] = 1.0
                    
                    # Transform relative to initial object pose (vectorized)
                    inv_initial_pose = inv_T(initial_object_pose_aug)
                    T_rel_all = np.einsum('ij,tjk->tik', inv_initial_pose, proprio_T_all)  # [T, 4, 4]
                    
                    # Apply augmentation (episode-level, same transform for all timesteps, always apply both position and rotation)
                    T_rel_aug_all = np.einsum('ij,tjk->tik', se3_transform_aug, T_rel_all)  # [T, 4, 4]
                    
                    # Transform back to marker coordinate frame (vectorized)
                    proprio_T_aug_all = np.einsum('ij,tjk->tik', initial_object_pose_aug, T_rel_aug_all)  # [T, 4, 4]
                    
                    # Update proprioception_marker_xyz and proprioception_marker_ortho6d
                    proprioception_marker_xyz = proprio_T_aug_all[:, :3, 3]  # [T, 3]
                    proprioception_marker_ortho6d = compute_ortho6d_from_rotation_matrix(proprio_T_aug_all[:, :3, :3])  # [T, 6]


                # Combine transformed xyz and euler angles with rest of data
                proprioception_marker = np.concatenate([
                    proprioception_marker_xyz,  # (T, 3)
                    proprioception_marker_ortho6d,  # (T, 6)
                    proprioception_camera[:, -1:]  # (T, 1) - gripper action
                ], axis=1)
                
                self.memory_buffer["proprioception"].append(proprioception_marker)

                
                # Load and transform action data
                action_camera = self.load_low_dim_data(root, osp.join(episode, "action"))[
                    :: self.downsample_rate
                ][: self.max_episode_len]  # (T, dim)
                
                # Extract xyz positions and euler angles from action
                action_xyz = action_camera[:, :3]  # (T, 3)
                action_ortho6d = action_camera[:, 3:9]  # (T, 6)
                
                # Transform action positions and euler angles to marker coordinates
                action_marker_xyz = self.transform_points_to_marker_coords(action_xyz, T_mc_transformation)
                action_marker_ortho6d = self.transform_ortho6d_to_marker_coords(action_ortho6d, T_mc_transformation)

                if self.augment_se3 and is_augmentation_episode:
                    # Apply augmentation to action (episode-level, vectorized, always apply both position and rotation, reuse se3_transform_aug and initial_object_pose_aug from proprioception)
                    T_action = len(action_marker_xyz)
                    # Convert all timesteps to SE(3) representation
                    action_R_all = compute_rotation_matrix_from_ortho6d(action_marker_ortho6d)  # [T, 3, 3]
                    action_T_all = np.zeros((T_action, 4, 4))
                    action_T_all[:, :3, :3] = action_R_all
                    action_T_all[:, :3, 3] = action_marker_xyz
                    action_T_all[:, 3, 3] = 1.0
                    
                    # Transform relative to initial object pose (vectorized)
                    inv_initial_pose = inv_T(initial_object_pose_aug)
                    T_rel_all = np.einsum('ij,tjk->tik', inv_initial_pose, action_T_all)  # [T, 4, 4]
                    
                    # Apply augmentation (episode-level, same transform for all timesteps, always apply both position and rotation)
                    T_rel_aug_all = np.einsum('ij,tjk->tik', se3_transform_aug, T_rel_all)  # [T, 4, 4]
                    
                    # Transform back to marker coordinate frame (vectorized)
                    action_T_aug_all = np.einsum('ij,tjk->tik', initial_object_pose_aug, T_rel_aug_all)  # [T, 4, 4]
                    
                    # Update action_marker_xyz and action_marker_ortho6d
                    action_marker_xyz = action_T_aug_all[:, :3, 3]  # [T, 3]
                    action_marker_ortho6d = compute_ortho6d_from_rotation_matrix(action_T_aug_all[:, :3, :3])  # [T, 6]


                # Combine transformed xyz and euler angles with rest of data
                action_marker = np.concatenate([
                    action_marker_xyz,  # (T, 3)
                    action_marker_ortho6d,  # (T, 6)
                    action_camera[:, -1:]  # (T, 1) - gripper action
                ], axis=1)
                
                self.memory_buffer["action"].append(action_marker)

                if self.use_relative:
                    delta_action_pos = action_marker_xyz[1:] - action_marker_xyz[:-1] # [T-1, 3]
                    delta_action_pos = np.concatenate([delta_action_pos, np.zeros((1, 3))], axis=0) # [T, 3]
                    action_R = compute_rotation_matrix_from_ortho6d(action_marker_ortho6d) # [T, 3, 3]
                    action_R_1 = action_R[1:] # [T-1, 3, 3]
                    action_R_0 = action_R[:-1] # [T-1, 3, 3]
                    R_rel = np.einsum('tij,tjk->tik', action_R_0.transpose(0, 2, 1), action_R_1)  # [T-1, 3, 3]
                    rotations = R.from_matrix(R_rel)
                    axis_angle = rotations.as_rotvec()  # [T-1, 3] - axis-angle representation
                    delta_action_rot = np.concatenate([axis_angle, np.zeros((1, 3))], axis=0) # [T, 3]
                    delta_action = np.concatenate([delta_action_pos, delta_action_rot, action_camera[:, -1:]], axis=-1) # [T, 7]
                    # NOTE: use gripper as absolute action
                    self.memory_buffer["delta_action"].append(delta_action)


                if self.use_estimated_extrinsics:
                    self.memory_buffer["extrinsics"].append(
                        self.load_low_dim_data(root, osp.join(episode, "extrinsics"))[
                            :: self.downsample_rate
                        ][: self.max_episode_len] # [T, 4, 4]
                    )
                    self.memory_buffer["episodic_extrinsics"].append(
                        self.load_low_dim_data(root, osp.join(episode, "extrinsics"), padding_size = self.padding_size)[
                            :, :: self.downsample_rate
                        ][:, :self.max_episode_len] # [1, T(padding), 4, 4]
                    )
                else:
                    self.memory_buffer["extrinsics"].append(
                        np.tile(np.eye(4)[None], (episode_length, 1, 1))[:self.max_episode_len] # [T, 4, 4]
                    )
                    self.memory_buffer["episodic_extrinsics"].append(
                        np.tile(np.eye(4)[None], (self.padding_size, 1, 1))[
                            ::self.downsample_rate
                        ][:self.max_episode_len][None] # [1, T(padding), 4, 4]
                    )

                # Load other data (unchanged)
                if self.use_dift:
                    self.memory_buffer["dift_points"].append(
                        self.load_low_dim_data(root, osp.join(episode, "dift_points"))[None]  # [1, N, 2]
                    )

                # Load point tracking sequences and transform to marker coordinates
                
                if self.use_dift_point_tracking:
                    point_tracking_sequence_path = osp.join(episode, "dift_point_tracking_sequence")
                    pixel_tracking_sequence_path = osp.join(episode, "dift_pixel_tracking_sequence")
                else:
                    point_tracking_sequence_path = osp.join(episode, "mask_point_tracking_sequence")
                    pixel_tracking_sequence_path = osp.join(episode, "mask_pixel_tracking_sequence")
                    
                # NOTE: tracking_sequence is already downsampled in realworld_3d_point_tracking_depth.py
                
                # Load point tracking sequence and transform to marker coordinates
                point_tracking_camera = self.load_point_tracking_data(
                    root,
                    point_tracking_sequence_path, 
                    sam_mask_data= not self.use_dift_point_tracking,
                )[0][::self.downsample_rate][:self.max_episode_len]  # [T, N, 3 or 4]
                
                # Extract xyz coordinates and visibility (if available)
                if point_tracking_camera.shape[2] == 4:
                    point_tracking_xyz = point_tracking_camera[:, :, :3]  # [T, N, 3]
                    point_tracking_visibility = point_tracking_camera[:, :, 3]  # [T, N]
                else:
                    point_tracking_xyz = point_tracking_camera  # [T, N, 3]
                    point_tracking_visibility = None
                
                # Transform xyz coordinates to marker coordinates
                T, N, _ = point_tracking_xyz.shape
                point_tracking_xyz_marker = np.zeros((T, N, 3))
                
                for t in range(T):
                    point_tracking_xyz_marker[t] = self.transform_points_to_marker_coords(
                        point_tracking_xyz[t], T_mc_transformation[t]
                    )
            

                # Store initial object pose for original episodes (for SE(3) augmentation)
                if self.augment_se3 and not is_augmentation_episode:
                    # Get initial object points (first timestep, first num_points_for_object points)
                    initial_object_points = point_tracking_xyz_marker[0, :self.num_points_for_object]  # [num_points_for_object, 3]
                    # Compute mean position
                    mean_position = np.mean(initial_object_points, axis=0)  # [3]
                    # Create SE(3) matrix with identity rotation
                    initial_object_pose = np.eye(4)
                    initial_object_pose[:3, 3] = mean_position
                    episode_initial_object_pose.append(initial_object_pose)
                
                
                if self.augment_se3 and is_augmentation_episode:
                    # Apply augmentation to point_tracking_xyz_marker (episode-level, vectorized, always apply both position and rotation)
                    # Convert all points to homogeneous coordinates [T, N, 4]
                    points_homo = np.concatenate([
                        point_tracking_xyz_marker,
                        np.ones((T, N, 1))
                    ], axis=2)  # [T, N, 4]
                    
                    # Transform relative to initial object pose (vectorized)
                    inv_initial_pose = inv_T(initial_object_pose_aug)
                    points_rel = np.einsum('ij,tnj->tni', inv_initial_pose, points_homo)  # [T, N, 4]
                    
                    # Apply augmentation (episode-level, same transform for all points)
                    points_rel_aug = np.einsum('ij,tnj->tni', se3_transform_aug, points_rel)  # [T, N, 4]
                    
                    # Transform back to marker coordinate frame (vectorized)
                    points_aug = np.einsum('ij,tnj->tni', initial_object_pose_aug, points_rel_aug)  # [T, N, 4]
                    
                    point_tracking_xyz_marker = points_aug[:, :, :3]  # [T, N, 3]
                

                # Combine xyz and visibility (if available)
                if point_tracking_visibility is not None:
                    point_tracking_marker = np.concatenate([
                        point_tracking_xyz_marker, 
                        point_tracking_visibility[:, :, None]
                    ], axis=2)  # [T, N, 4]
                else:
                    point_tracking_marker = point_tracking_xyz_marker  # [T, N, 3]
                

                self.memory_buffer["point_tracking_sequence"].append(point_tracking_marker)

                
                if self.use_relative:
                    delta_point_tracking_pos = point_tracking_marker[1:, :, :3] - point_tracking_marker[:-1, :, :3] # [T-1, N(+1), 3]
                    delta_point_tracking_pos = np.concatenate([delta_point_tracking_pos, np.zeros((1, N, 3))], axis=0) # [T, N(+1), 3]
                    if point_tracking_visibility is not None:
                        delta_point_tracking_marker = np.concatenate([delta_point_tracking_pos, point_tracking_visibility[:, :, None]], axis=2) # [T, N(+1), 4]
                    else:
                        delta_point_tracking_marker = delta_point_tracking_pos # [T, N(+1), 3]
                    self.memory_buffer["delta_point_tracking_sequence"].append(delta_point_tracking_marker)

                
                self.memory_buffer["pixel_tracking_sequence"].append(
                    self.load_point_tracking_data(
                        root,
                        pixel_tracking_sequence_path,
                        sam_mask_data= not self.use_dift_point_tracking,
                    )[0][::self.downsample_rate][:self.max_episode_len],  # [T, N, 3]
                )

                # NOTE: tracking_sequence is already downsampled in realworld_3d_point_tracking_depth.py
                
                # Load episodic point tracking sequence and transform to marker coordinates
                episodic_point_tracking_camera = self.load_point_tracking_data(
                    root,
                    point_tracking_sequence_path,
                    padding_size=self.padding_size,
                    sam_mask_data= not self.use_dift_point_tracking,
                )[:, ::self.downsample_rate][:, :self.max_episode_len]  # [1, max_episode_len, N, 3 or 4] 
                
                # Extract xyz coordinates and visibility (if available)
                episodic_point_tracking_camera_data = episodic_point_tracking_camera[0]  # [max_episode_len, N, 3 or 4]
                
                if episodic_point_tracking_camera_data.shape[2] == 4:
                    episodic_point_tracking_xyz = episodic_point_tracking_camera_data[:, :, :3]  # [max_episode_len, N, 3]
                    episodic_point_tracking_visibility = episodic_point_tracking_camera_data[:, :, 3]  # [max_episode_len, N]
                else:
                    episodic_point_tracking_xyz = episodic_point_tracking_camera_data  # [max_episode_len, N, 3]
                    episodic_point_tracking_visibility = None
                
                # Transform xyz coordinates to marker coordinates
                T_episodic, N_episodic, _ = episodic_point_tracking_xyz.shape
                episodic_point_tracking_xyz_marker = np.zeros((T_episodic, N_episodic, 3))

                
                for t in range(T_episodic):
                    episodic_point_tracking_xyz_marker[t] = self.transform_points_to_marker_coords(
                        episodic_point_tracking_xyz[t], episodic_T_mc_transformation[t]
                    )
                
                
                if self.augment_se3 and is_augmentation_episode:
                    # Apply augmentation to episodic_point_tracking_xyz_marker (episode-level, vectorized, always apply both position and rotation)
                    # Convert all points to homogeneous coordinates [T_episodic, N_episodic, 4]
                    episodic_points_homo = np.concatenate([
                        episodic_point_tracking_xyz_marker,
                        np.ones((T_episodic, N_episodic, 1))
                    ], axis=2)  # [T_episodic, N_episodic, 4]
                    
                    # Transform relative to initial object pose (vectorized)
                    inv_initial_pose = inv_T(initial_object_pose_aug)
                    episodic_points_rel = np.einsum('ij,tnj->tni', inv_initial_pose, episodic_points_homo)  # [T_episodic, N_episodic, 4]
                    
                    # Apply augmentation (episode-level, same transform for all points)
                    episodic_points_rel_aug = np.einsum('ij,tnj->tni', se3_transform_aug, episodic_points_rel)  # [T_episodic, N_episodic, 4]
                    
                    # Transform back to marker coordinate frame (vectorized)
                    episodic_points_aug = np.einsum('ij,tnj->tni', initial_object_pose_aug, episodic_points_rel_aug)  # [T_episodic, N_episodic, 4]
                    
                    episodic_point_tracking_xyz_marker = episodic_points_aug[:, :, :3]  # [T_episodic, N_episodic, 3]


                # Combine xyz and visibility (if available)
                if episodic_point_tracking_visibility is not None:
                    episodic_point_tracking_marker = np.concatenate([
                        episodic_point_tracking_xyz_marker, 
                        episodic_point_tracking_visibility[:, :, None]
                    ], axis=2)  # [max_episode_len, N, 4]
                else:
                    episodic_point_tracking_marker = episodic_point_tracking_xyz_marker  # [max_episode_len, N, 3]
                
                
                self.memory_buffer["episodic_point_tracking_sequence"].append(episodic_point_tracking_marker[None])  # [1, max_episode_len, N, 3 or 4]
                self.memory_buffer["episodic_pixel_tracking_sequence"].append(
                    self.load_point_tracking_data(
                        root,
                        pixel_tracking_sequence_path,
                        padding_size=self.padding_size,
                        sam_mask_data= not self.use_dift_point_tracking,
                    )[:, ::self.downsample_rate][:, :self.max_episode_len]  # [1, padding, N, 3] -> [1, max_episode_len, N, 3]
                )
                

                self.memory_buffer["text"].append(
                    root[osp.join(episode, "task_description")]  # [1, ]
                )


                episode_length = len(self.memory_buffer["action"][-1])
                self.memory_buffer["episode_idx"].append(
                    np.array([load_episode_num] * episode_length)
                )
                load_episode_num += 1
                path_load_episode_num += 1
                if (
                    self.max_episode is not None
                    and path_load_episode_num >= self.max_episode
                ):
                    break
            print("load_episode_num", load_episode_num)
            self.data_path_offset.append(load_episode_num)
        self.eps_end = np.cumsum([len(x) for x in self.memory_buffer["action"]])

        for k, v in self.memory_buffer.items():
            self.memory_buffer[k] = np.concatenate(v)

        print(f">> Loaded {load_episode_num} episodes!")
        print(f">> self.data_path_offset {self.data_path_offset}")

    def load_data_to_memory_not_marker_coordinate(self):
        load_episode_num = 0
        for path in self.data_path:
            print(">> Loading data from: ", path)
            root = zarr.open(path, mode="a")
            episodes = list(root.group_keys())
            episodes = sorted(episodes, key=lambda x: int(x.split("_")[1]))
            print(f">> {len(episodes)} episodes num from {path} ")
            if len(episodes) == 0:
                raise ValueError("No episodes found!")
            path_load_episode_num = 0

            for i, episode in tqdm(enumerate(episodes)):
                episode_length = len(
                    self.load_low_dim_data(root, osp.join(episode, "action"))[
                        :: self.downsample_rate
                    ]
                )
                if self.max_episode_len and episode_length > self.max_episode_len:
                    print(f"episode {episode} exceeds max episode length")
                else:
                    print(f"episode {episode} length: {episode_length}")

                # Load transformation matrix based on configuration (for view_actions, but not for transformation)
                # For now, use T_mc_opt as default (can be extended to support both options)
                T_mc_opt_name = "T_mc_opt" if not self.droid else "T_mc_opt_droid"
                T_mc_transformation = self.load_low_dim_data(root, osp.join(episode, T_mc_opt_name))[
                    :: self.downsample_rate
                ][: self.max_episode_len]  # (T, 4, 4)

                view_actions_se3 = T_mc_transformation.copy() # [T, 4, 4]
                translation_view_action = view_actions_se3[:, :3, 3].copy() # [T, 3]
                rotation_view_action = view_actions_se3[:, :3, :3].copy() # [T, 3, 3]
                # convert rotation matrix to ortho6d
                rotation_view_action = compute_ortho6d_from_rotation_matrix(rotation_view_action)
                view_actions = np.concatenate([translation_view_action, rotation_view_action], axis=-1) # [T, 9]
                self.memory_buffer["view_actions"].append(view_actions)

                if self.use_relative: # tile last action
                    delta_view_action = self.compute_delta_view_action(view_actions_se3.copy())
                    self.memory_buffer["delta_view_action"].append(delta_view_action)

                
                # Load and transform action_se3 data
                action_se3_camera = self.load_low_dim_data(root, osp.join(episode, "action_se3"))[
                    :: self.downsample_rate
                ][: self.max_episode_len]  # (T, 4, 4)
                
                
                self.memory_buffer["action_se3"].append(action_se3_camera)

                # # Load and transform proprioception_se3 data
                proprioception_se3_camera = self.load_low_dim_data(root, osp.join(episode, "proprioception_se3"))[
                    :: self.downsample_rate
                ][: self.max_episode_len]  # (T, 4, 4)
                
                
                self.memory_buffer["proprioception_se3"].append(proprioception_se3_camera)


                # Load proprioception data (no transformation to marker coordinates)
                proprioception_camera = self.load_low_dim_data(root, osp.join(episode, "proprioception"))[
                    :: self.downsample_rate
                ][: self.max_episode_len]  # (T, dim)
                
                # Extract xyz positions and ortho6d from proprioception
                xyz_positions = proprioception_camera[:, :3]  # (T, 3)
                ortho6d = proprioception_camera[:, 3:9]  # (T, 6)
                
                # No transformation - use camera coordinates directly
                proprioception_camera_xyz = xyz_positions  # (T, 3)
                proprioception_camera_ortho6d = ortho6d  # (T, 6)

                # Combine xyz and ortho6d with rest of data
                proprioception = np.concatenate([
                    proprioception_camera_xyz,  # (T, 3)
                    proprioception_camera_ortho6d,  # (T, 6)
                    proprioception_camera[:, -1:]  # (T, 1) - gripper action
                ], axis=1)
                
                self.memory_buffer["proprioception"].append(proprioception)
                # Load and transform action data (no transformation to marker coordinates)
                action_camera = self.load_low_dim_data(root, osp.join(episode, "action"))[
                    :: self.downsample_rate
                ][: self.max_episode_len]  # (T, dim)
                
                # Extract xyz positions and ortho6d from action
                action_xyz = action_camera[:, :3]  # (T, 3)
                action_ortho6d = action_camera[:, 3:9]  # (T, 6)
                
                # No transformation - use camera coordinates directly
                action_camera_xyz = action_xyz  # (T, 3)
                action_camera_ortho6d = action_ortho6d  # (T, 6)

                # Combine xyz and ortho6d with rest of data
                action = np.concatenate([
                    action_camera_xyz,  # (T, 3)
                    action_camera_ortho6d,  # (T, 6)
                    action_camera[:, -1:]  # (T, 1) - gripper action
                ], axis=1)
                
                self.memory_buffer["action"].append(action)

                if self.use_relative:
                    delta_action_pos = action_camera_xyz[1:] - action_camera_xyz[:-1] # [T-1, 3]
                    delta_action_pos = np.concatenate([delta_action_pos, np.zeros((1, 3))], axis=0) # [T, 3]
                    action_R = compute_rotation_matrix_from_ortho6d(action_camera_ortho6d) # [T, 3, 3]
                    action_R_1 = action_R[1:] # [T-1, 3, 3]
                    action_R_0 = action_R[:-1] # [T-1, 3, 3]
                    R_rel = np.einsum('tij,tjk->tik', action_R_0.transpose(0, 2, 1), action_R_1)  # [T-1, 3, 3]
                    rotations = R.from_matrix(R_rel)
                    axis_angle = rotations.as_rotvec()  # [T-1, 3] - axis-angle representation
                    delta_action_rot = np.concatenate([axis_angle, np.zeros((1, 3))], axis=0) # [T, 3]
                    delta_action = np.concatenate([delta_action_pos, delta_action_rot, action_camera[:, -1:]], axis=-1) # [T, 7]
                    # NOTE: use gripper as absolute action
                    self.memory_buffer["delta_action"].append(delta_action)

                if self.use_estimated_extrinsics:
                    self.memory_buffer["extrinsics"].append(
                        self.load_low_dim_data(root, osp.join(episode, "extrinsics"))[
                            :: self.downsample_rate
                        ][: self.max_episode_len] # [T, 4, 4]
                    )
                    self.memory_buffer["episodic_extrinsics"].append(
                        self.load_low_dim_data(root, osp.join(episode, "extrinsics"), padding_size = self.padding_size)[
                            :, :: self.downsample_rate
                        ][:, :self.max_episode_len] # [1, T(padding), 4, 4]
                    )
                else:
                    self.memory_buffer["extrinsics"].append(
                        np.tile(np.eye(4)[None], (episode_length, 1, 1))[:self.max_episode_len] # [T, 4, 4]
                    )
                    self.memory_buffer["episodic_extrinsics"].append(
                        np.tile(np.eye(4)[None], (self.padding_size, 1, 1))[
                            ::self.downsample_rate
                        ][:self.max_episode_len][None] # [1, T(padding), 4, 4]
                    )

                # Load other data (unchanged)
                if self.use_dift:
                    self.memory_buffer["dift_points"].append(
                        self.load_low_dim_data(root, osp.join(episode, "dift_points"))[None]  # [1, N, 2]
                    )
                
                # Load point tracking sequences (no transformation to marker coordinates)
                
                if self.use_dift_point_tracking:
                    point_tracking_sequence_path = osp.join(episode, "dift_point_tracking_sequence")
                    pixel_tracking_sequence_path = osp.join(episode, "dift_pixel_tracking_sequence")
                else:
                    point_tracking_sequence_path = osp.join(episode, "mask_point_tracking_sequence")
                    pixel_tracking_sequence_path = osp.join(episode, "mask_pixel_tracking_sequence")
                    
                # NOTE: tracking_sequence is already downsampled in realworld_3d_point_tracking_depth.py
                
                # Load point tracking sequence (no transformation)
                point_tracking_camera = self.load_point_tracking_data(
                    root,
                    point_tracking_sequence_path, 
                    sam_mask_data= not self.use_dift_point_tracking,
                )[0][::self.downsample_rate][:self.max_episode_len]  # [T, N, 3 or 4]
                
                # Extract xyz coordinates and visibility (if available)
                if point_tracking_camera.shape[2] == 4:
                    point_tracking_xyz = point_tracking_camera[:, :, :3]  # [T, N, 3]
                    point_tracking_visibility = point_tracking_camera[:, :, 3]  # [T, N]
                else:
                    point_tracking_xyz = point_tracking_camera  # [T, N, 3]
                    point_tracking_visibility = None
                
                # No transformation - use camera coordinates directly
                point_tracking_xyz_camera = point_tracking_xyz  # [T, N, 3]
                T, N, _ = point_tracking_xyz_camera.shape

                # Combine xyz and visibility (if available)
                if point_tracking_visibility is not None:
                    point_tracking = np.concatenate([
                        point_tracking_xyz_camera, 
                        point_tracking_visibility[:, :, None]
                    ], axis=2)  # [T, N, 4]
                else:
                    point_tracking = point_tracking_xyz_camera  # [T, N, 3]

                self.memory_buffer["point_tracking_sequence"].append(point_tracking)

                if self.use_relative:
                    delta_point_tracking_pos = point_tracking[1:, :, :3] - point_tracking[:-1, :, :3] # [T-1, N(+1), 3]
                    delta_point_tracking_pos = np.concatenate([delta_point_tracking_pos, np.zeros((1, N, 3))], axis=0) # [T, N(+1), 3]
                    if point_tracking_visibility is not None:
                        delta_point_tracking = np.concatenate([delta_point_tracking_pos, point_tracking_visibility[:, :, None]], axis=2) # [T, N(+1), 4]
                    else:
                        delta_point_tracking = delta_point_tracking_pos # [T, N(+1), 3]
                    self.memory_buffer["delta_point_tracking_sequence"].append(delta_point_tracking)

                self.memory_buffer["pixel_tracking_sequence"].append(
                    self.load_point_tracking_data(
                        root,
                        pixel_tracking_sequence_path,
                        sam_mask_data= not self.use_dift_point_tracking,
                    )[0][::self.downsample_rate][:self.max_episode_len],  # [T, N, 3]
                )

                # NOTE: tracking_sequence is already downsampled in realworld_3d_point_tracking_depth.py
                
                # Load episodic point tracking sequence (no transformation)
                episodic_point_tracking_camera = self.load_point_tracking_data(
                    root,
                    point_tracking_sequence_path,
                    padding_size=self.padding_size,
                    sam_mask_data= not self.use_dift_point_tracking,
                )[:, ::self.downsample_rate][:, :self.max_episode_len]  # [1, max_episode_len, N, 3 or 4] 
                
                # Extract xyz coordinates and visibility (if available)
                episodic_point_tracking_camera_data = episodic_point_tracking_camera[0]  # [max_episode_len, N, 3 or 4]
                
                if episodic_point_tracking_camera_data.shape[2] == 4:
                    episodic_point_tracking_xyz = episodic_point_tracking_camera_data[:, :, :3]  # [max_episode_len, N, 3]
                    episodic_point_tracking_visibility = episodic_point_tracking_camera_data[:, :, 3]  # [max_episode_len, N]
                else:
                    episodic_point_tracking_xyz = episodic_point_tracking_camera_data  # [max_episode_len, N, 3]
                    episodic_point_tracking_visibility = None
                
                # No transformation - use camera coordinates directly
                episodic_point_tracking_xyz_camera = episodic_point_tracking_xyz  # [max_episode_len, N, 3]
                T_episodic, N_episodic, _ = episodic_point_tracking_xyz_camera.shape

                # Combine xyz and visibility (if available)
                if episodic_point_tracking_visibility is not None:
                    episodic_point_tracking = np.concatenate([
                        episodic_point_tracking_xyz_camera, 
                        episodic_point_tracking_visibility[:, :, None]
                    ], axis=2)  # [max_episode_len, N, 4]
                else:
                    episodic_point_tracking = episodic_point_tracking_xyz_camera  # [max_episode_len, N, 3]
                
                self.memory_buffer["episodic_point_tracking_sequence"].append(episodic_point_tracking[None])  # [1, max_episode_len, N, 3 or 4]
                self.memory_buffer["episodic_pixel_tracking_sequence"].append(
                    self.load_point_tracking_data(
                        root,
                        pixel_tracking_sequence_path,
                        padding_size=self.padding_size,
                        sam_mask_data= not self.use_dift_point_tracking,
                    )[:, ::self.downsample_rate][:, :self.max_episode_len]  # [1, padding, N, 3] -> [1, max_episode_len, N, 3]
                )
                self.memory_buffer["text"].append(
                    root[osp.join(episode, "task_description")] # [1, ]
                )


                episode_length = len(self.memory_buffer["action"][-1])
                self.memory_buffer["episode_idx"].append(
                    np.array([load_episode_num] * episode_length)
                )
                load_episode_num += 1
                path_load_episode_num += 1
                if (
                    self.max_episode is not None
                    and path_load_episode_num >= self.max_episode
                ):
                    break
            print("load_episode_num", load_episode_num)
            self.data_path_offset.append(load_episode_num)
        self.eps_end = np.cumsum([len(x) for x in self.memory_buffer["action"]])

        for k, v in self.memory_buffer.items():
            self.memory_buffer[k] = np.concatenate(v)

        print(f">> Loaded {load_episode_num} episodes!")
        print(f">> self.data_path_offset {self.data_path_offset}")

    def load_point_tracking_data(self, root, point_tracking_path, padding_size = None, sam_mask_data = False):
        if sam_mask_data:
            point_tracking_sequence = np.transpose(
                root[point_tracking_path][:, :, -self.backward_episode_len:], [2, 1, 0, 3]
            )  # (T,N, num_obj, 3 or 4)
            T, N, num_obj, D = point_tracking_sequence.shape
            point_tracking_sequence =point_tracking_sequence.reshape(T, N*num_obj, D) #[T, N*num_obj, 3 or 4]
        else:
            point_tracking_sequence = np.transpose(
                root[point_tracking_path][:, -self.backward_episode_len:], [1, 0, 2]
            )  # (T,N,3 or 4)
        if self.flow_dim == 3:
            point_tracking_sequence = point_tracking_sequence[..., :3]

        if padding_size is not None:
            last_time_step = point_tracking_sequence[-1, :, :]
            padding = np.repeat(
                last_time_step[np.newaxis, :, :],
                padding_size - point_tracking_sequence.shape[0],
                axis=0,
            )
            return np.concatenate([point_tracking_sequence, padding], axis=0)[
                None, :
            ]  # (1,padding_size,N,3 or 4)
        else:
            return point_tracking_sequence[None, :]

    