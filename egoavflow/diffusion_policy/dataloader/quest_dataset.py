import random

import numpy as np
import torch
from einops import rearrange
import copy
from egoavflow.diffusion_policy.dataloader.diffusion_bc_dataset import (
    create_sample_indices,
    get_data_stats,
    normalize_data,
    sample_sequence
)
from egoavflow.diffusion_policy.dataloader.replay_buffer_3d import (
    QuestPointTrackingReplayBuffer3D,
)

from egoavflow.utils import compute_ortho6d_from_rotation_matrix, compute_rotation_matrix_from_ortho6d


def get_data_stats_custom(data):
    data = data.reshape(-1, data.shape[-1])
    
    # Check if data has visibility dimension (shape [..., 4] with [x, y, z, visible])
    assert data.shape[-1] in [3, 4]
    # Filter points where visible == 1
    visible_mask = data[:, -1] == 1
    visible_points = data[visible_mask]
    
    # Calculate stats for x, y, z using only visible points
    if len(visible_points) > 0:
        point_stats = {
            "min": np.min(visible_points[:, :-1], axis=0),
            "max": np.max(visible_points[:, :-1], axis=0),
            "mean": np.mean(visible_points[:, :-1], axis=0),
            "std": np.std(visible_points[:, :-1], axis=0)
        }
        # For visible dimension, calculate stats on all data
        visible_stats = {
            "min": np.min(data[:, -1:], axis=0),
            "max": np.max(data[:, -1:], axis=0),
            "mean": np.mean(data[:, -1:], axis=0),
            "std": np.std(data[:, -1:], axis=0)
        }
        # Combine stats
        stats = {
            "min": np.concatenate([point_stats["min"], visible_stats["min"]]),
            "max": np.concatenate([point_stats["max"], visible_stats["max"]]),
            "mean": np.concatenate([point_stats["mean"], visible_stats["mean"]]),
            "std": np.concatenate([point_stats["std"], visible_stats["std"]])
        }
    
    
    return stats

class QuestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dirs,
        num_points,
        downsample_rate=1,
        pred_horizon=24,
        obs_horizon=1,
        action_horizon=8,
        action_dim=10,
        shuffle_points=False,
        max_episode=None,
        load_camera_ids=[0],
        point_tracking_camera_id=0,
        camera_resize_shape=None,
        point_tracking_img_size=[256, 256],
        pixel_tracking_img_size=[256, 256],
        unnormal_list=[],
        is_sam=True,
        max_episode_len=None,
        padding_size=None,
        
        seed=0,
        flow_dim = 3,
        use_dift = False,
        num_dift_points = 6,
        rgbd_img_size = None,
        use_dift_point_tracking=False,
        augment_se3=False,
        se3_translation_range=0.5,
        se3_rotation_range=30, # +-30 deg
        use_estimated_extrinsics=False,
        stats_from_training=None,
        use_marker_coordinate=False,
        noise_aug_prob = 0.0,
        noise_aug_std = 0.01, 
        backward_episode_len = None,
        droid = False,
        
        use_relative = False,
        flow_norm_type='min_max',
        norm_type = 'min_max',
        use_ptp = False,
        zero_padding_for_ptp_initial = False,
        action_norm_type = 'min_max',
        
        num_aug_per_episode=1,
        num_points_for_object=None,
        num_points_per_mask = None,
        **kwargs
    ):
        if backward_episode_len is None:
            backward_episode_len = 10000000
            
        self.augment_se3 = augment_se3
        se3_translation_range = np.array(se3_translation_range) if type(se3_translation_range) != np.ndarray else se3_translation_range
        self.se3_translation_range = se3_translation_range
        self.se3_rotation_range = se3_rotation_range
        self.num_points_per_mask = num_points_per_mask

        self.buffer = QuestPointTrackingReplayBuffer3D(
            data_path=data_dirs,
            load_camera_ids=load_camera_ids,
            point_tracking_camera_id=point_tracking_camera_id,
            camera_resize_shape=camera_resize_shape,
            point_tracking_img_size=point_tracking_img_size, # not used inside
            downsample_rate=downsample_rate,
            max_episode=max_episode,
            max_episode_len=max_episode_len,
            padding_size=padding_size,
            is_sam=is_sam,
            flow_dim=flow_dim,
            use_dift=use_dift,
            use_dift_point_tracking=use_dift_point_tracking,
            use_estimated_extrinsics=use_estimated_extrinsics,
            use_marker_coordinate=use_marker_coordinate,
            backward_episode_len=backward_episode_len,
            droid = droid,
            use_relative=use_relative,
            augment_se3=augment_se3,
            num_aug_per_episode=num_aug_per_episode,
            se3_translation_range=se3_translation_range,
            se3_rotation_range=se3_rotation_range,
            num_points_for_object=num_points_for_object,
        )
        self.downsample_rate = downsample_rate
        self.data_dirs = data_dirs
        self.point_tracking_camera_id = point_tracking_camera_id
        self.load_camera_ids = load_camera_ids
        self.flow_norm_type = flow_norm_type
        self.norm_type = norm_type
        self.action_norm_type = action_norm_type
        
        self.use_relative = use_relative
        self.camera_resize_shape = camera_resize_shape
        self.point_tracking_img_size = point_tracking_img_size
        self.pixel_tracking_img_size = pixel_tracking_img_size
        self.num_points = num_points
        self.shuffle_points = shuffle_points
        self.seed = seed
        self.set_seed(self.seed)
        self.unnormal_list = unnormal_list
        self.episode_ends = self.buffer.eps_end
        self.buffer.memory_buffer["actual_sample_indices"] = np.arange(
            len(self.buffer["action"])
        )

        self.use_ptp = use_ptp
        self.zero_padding_for_ptp_initial = zero_padding_for_ptp_initial
        
        
        # Calculate padding for history and future
        total_sequence_length = obs_horizon -1 + pred_horizon 
        
        indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=total_sequence_length,  # Changed from pred_horizon to total_sequence_length
            pad_before=obs_horizon - 1,  # Need 2 more timesteps before (t-2, t-1)
            pad_after=action_horizon - 1,
        )
        
        
        self.flow_dim = flow_dim
        
        if flow_dim == 3:
            self._norm_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            self._norm_std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        else:
            self._norm_mean = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            self._norm_std = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)


        # NOTE: should not compute mean or std, since it's padded data!
        self.episodic_point_tracking_data = self.buffer["episodic_point_tracking_sequence"].copy()  # [num_episode, T(padding), N, 3]
        
        self.point_tracking_data = self.buffer["point_tracking_sequence"].copy()  # [num_epi*T (all), N, 3 or 4]
        
        self.extrinsics = self.buffer["episodic_extrinsics"] # [num_episode, T(padding), 4, 4]
        print(f"In QuestDataset, extrinsics: {self.extrinsics.shape}")
        if use_marker_coordinate:
            self.T_mc_opt = self.buffer["episodic_T_mc_opt"].copy() # [num_episode, T(padding), 4, 4]
            print(f"In QuestDataset, T_mc_opt: {self.T_mc_opt.shape}")


        self.pixel_tracking_data = self.buffer["episodic_pixel_tracking_sequence"] # [num_episode, T(padding), N, 3]

        self.initial_frames = self.buffer["initial_frame"] # empty 

        self.dift_points = self.buffer["dift_points"] # [num_episode, N, 2] (not related to time axis), in original camera_0/rgb (640(w), 480(h))
        self.text = self.buffer["text"] # [num_episode, ]

        self.buffer.remove_key("episodic_point_tracking_sequence") 
        self.buffer.remove_key("episodic_pixel_tracking_sequence") 
        self.buffer.remove_key("episodic_extrinsics")
        if use_marker_coordinate:
            self.buffer.remove_key("episodic_T_mc_opt")
        self.buffer.remove_key("initial_frame")
        self.buffer.remove_key("dift_points")
        self.buffer.remove_key("text")
        
        stats = dict()
        for key, data in self.buffer.memory_buffer.items():
            print(key)
            if key in self.unnormal_list:
                pass
            else:
                stats[key] = get_data_stats(data)

            if key in self.unnormal_list or (self.use_ptp and self.use_relative and key in ["delta_action", "delta_view_action", "delta_point_tracking_sequence", "delta_view_action_for_uncondition"]):
                pass
            else:
                
                if key == "delta_point_tracking_sequence":
                    norm_type = self.flow_norm_type
                elif key in ['action', 'view_actions', "delta_action", "delta_view_action", "delta_view_action_for_uncondition"] or 'action' in key:
                    norm_type = self.action_norm_type
                else:
                    norm_type = self.norm_type
                print(f"Normalizing {key} with shape {data.shape} and norm_type: {norm_type}")
                self.buffer.memory_buffer[key] = normalize_data(data, stats[key], type=norm_type)


        if stats_from_training is None:
            stats["point_tracking_data"] = get_data_stats_custom(self.point_tracking_data)
        else:
            stats = copy.deepcopy(stats_from_training)
            print('Ignore previously computed stats. Instead use the loaded stats from training.')
        
        
        self.pt_min = stats["point_tracking_data"]["min"].copy() # [3 or 4]
        self.pt_max = stats["point_tracking_data"]["max"].copy() # [3 or 4]
        self.pt_mean = stats["point_tracking_data"]["mean"].copy() # [3 or 4]    
        self.pt_std = stats["point_tracking_data"]["std"].copy() # [3 or 4]
            
        print(f"In QuestDataset, stats_from_training is None: {stats_from_training is None}, pt_min: {self.pt_min}, pt_max: {self.pt_max}, pt_mean: {self.pt_mean}, pt_std: {self.pt_std}")


        self.indices = indices
        self.stats = stats
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.use_dift = use_dift
        self.rgbd_img_size = rgbd_img_size
        self.num_dift_points = num_dift_points
        self.use_dift_point_tracking = use_dift_point_tracking
        
        self.use_marker_coordinate = use_marker_coordinate
        self.noise_aug_prob = noise_aug_prob
        self.noise_aug_std = noise_aug_std
        
        
    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    
    @staticmethod
    def process_pixel_tracking_data(
        episode_pixel_tracking,
        pixel_tracking_img_size,
    ):
        
        # clip the points to the image size
        episode_pixel_tracking[..., :2] = np.clip(
            episode_pixel_tracking[..., :2], a_min=np.zeros(2), a_max=np.array(pixel_tracking_img_size) - 1
        ) # [T, N, 3]

        random_sample_indices = None


        #         # # Find nearest neighbors
        #         # NOTE: assuming uniformly sampled within the sam mask, select unique closest points to DIFT points
                

        pixel_tracking_data = episode_pixel_tracking.copy()

        # normalize the x and y to [0,1]
        pixel_tracking_data = QuestDataset.normalize_pixels(
            pixel_tracking_data, *pixel_tracking_img_size
        )
        return pixel_tracking_data, random_sample_indices
    
    @staticmethod
    def normalize_pixels(tracking_pixel_sequence, W, H):
        """
        Args:
            tracking_point_sequence: (T,num_points,3) 3: (x,y,visible)
            H: height of image
            W: width of image
        """
        
        tracking_pixel_sequence[:, :, 0] = tracking_pixel_sequence[:, :, 0] / W
        tracking_pixel_sequence[:, :, 1] = tracking_pixel_sequence[:, :, 1] / H
        return tracking_pixel_sequence

    @staticmethod
    def process_point_tracking_data(
        episode_point_tracking,
        pt_min=None,
        pt_max=None,
        flow_transforms=None,
        pt_mean=None,
        pt_std=None,
        norm_type='min_max',
    ):
        

        point_tracking_data = episode_point_tracking.copy()

        # normalize the points to [0,1]
        if norm_type == 'min_max':
            point_tracking_data = (point_tracking_data - pt_min[None,None, :]) / (pt_max[None,None, :] - pt_min[None,None, :])
            # [0, 1] -> [-1, 1] normalize 
            point_tracking_data = rearrange(point_tracking_data, "T N C -> C T N")
            point_tracking_data = flow_transforms(point_tracking_data)
            point_tracking_data = rearrange(point_tracking_data, "C T N -> T N C")
        elif norm_type == 'mean_std':
            point_tracking_data = (point_tracking_data - pt_mean[None,None, :]) / (pt_std[None,None, :] + 1e-10)
        
        
        return point_tracking_data

    
    def flow_transforms(self, x):
        """
        x: numpy array, shape [C, ...]
        mean, std: [C]
        """
        mean = self._norm_mean.astype(x.dtype)
        std = self._norm_std.astype(x.dtype)
        # [C, 1, 1, ...] (x.ndim-1 ones)
        shape = (len(mean),) + (1,) * (x.ndim - 1)
        mean = mean.reshape(shape)
        std = std.reshape(shape)
        return (x - mean) / std

    def unnormalize_transform(self, x):
        """
        x: numpy array, shape [..., C]
        mean, std: [C]
        """
        mean = self._norm_mean.astype(x.dtype)
        std = self._norm_std.astype(x.dtype)
        # [1, 1, ..., C] (x.ndim-1 ones)
        shape = (1,) * (x.ndim - 1) + (len(mean),)
        mean = mean.reshape(shape)
        std = std.reshape(shape)
        return x * std + mean


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.indices[idx]
        # Calculate total sequence length including history
        total_sequence_length = self.obs_horizon -1 + self.pred_horizon 

            
        nsample = sample_sequence(
            train_data=self.buffer.memory_buffer,
            sequence_length=total_sequence_length,  # Changed from pred_horizon to total_sequence_length
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )
        # load episode-level flow and mask
        episode_idx = nsample["episode_idx"][0]
        
        # Split the sequence into history and future
        
        point_tracking_full = nsample["point_tracking_sequence"]  # [total_sequence_length, N(*num_obj), 3 or 4], unnormalized
        pixel_tracking_full = nsample["pixel_tracking_sequence"]  # [total_sequence_length, N, 3 or 4], unnormalized
        proprioception_full = nsample["proprioception"]  # [total_sequence_length, dim], normalized
        action_full = nsample["action"]  # [total_sequence_length, dim], normalized
        extrinsics_full = nsample["extrinsics"]  # [total_sequence_length, 4, 4]
        view_action_full = nsample["view_actions"]  # [total_sequence_length, 6]

        if self.use_relative:
            delta_view_action_full = nsample["delta_view_action"]  # [total_sequence_length, 6]
            delta_view_action_for_uncondition_full = nsample["delta_view_action_for_uncondition"]  # [total_sequence_length, 6]
            delta_action_full = nsample["delta_action"]  # [total_sequence_length, 7], normalized
            delta_point_tracking_full = nsample["delta_point_tracking_sequence"]  # [total_sequence_length, N(*num_obj), 3 or 4]
            if self.use_ptp : # without this, training is not affected. just for test-time verification in the PTP paper.
                if self.zero_padding_for_ptp_initial and sample_start_idx > 0:
                    delta_view_action_full[:self.obs_horizon-1] = np.zeros_like(delta_view_action_full[:self.obs_horizon-1])
                    delta_view_action_for_uncondition_full[:self.obs_horizon-1] = np.zeros_like(delta_view_action_for_uncondition_full[:self.obs_horizon-1])
                    delta_point_tracking_full[:self.obs_horizon-1] = np.zeros_like(delta_point_tracking_full[:self.obs_horizon-1])
                    
                    delta_action_full[:self.obs_horizon-1] = np.zeros_like(delta_action_full[:self.obs_horizon-1])

                # normalize
                if not self.augment_se3:
                    for k, v in self.stats['delta_view_action'].items():
                        assert np.linalg.norm(self.stats['delta_view_action_for_uncondition'][k] -self.stats['delta_view_action'][k]) < 1e-6


                delta_view_action_full = normalize_data(delta_view_action_full, self.stats["delta_view_action_for_uncondition"], type=self.action_norm_type)
                delta_view_action_for_uncondition_full = normalize_data(delta_view_action_for_uncondition_full, self.stats["delta_view_action_for_uncondition"], type=self.action_norm_type)
                delta_action_full = normalize_data(delta_action_full, self.stats["delta_action"], type=self.action_norm_type)
                delta_point_tracking_full = normalize_data(delta_point_tracking_full, self.stats["delta_point_tracking_sequence"], type=self.flow_norm_type)
        

        noise_aug = np.random.rand() < self.noise_aug_prob


        # If obs_horizon (history length) is 3, Extract history (t-2, t-1, t) and future (t, t+1, ..., t+pred_horizon-1)
        point_tracking_history = point_tracking_full[:self.obs_horizon]  # [history_length, N(*num_obj), 3 or 4]
        point_tracking_future = point_tracking_full[self.obs_horizon-1:]  # [pred_horizon, N(*num_obj), 3 or 4] (starting at t)
        
        pixel_tracking_history = pixel_tracking_full[:self.obs_horizon]  # [history_length, N, 3]
        pixel_tracking_future = pixel_tracking_full[self.obs_horizon-1:]  # [pred_horizon, N, 3] (starting at t)
        
        proprioception_history = proprioception_full[:self.obs_horizon]  # [history_length, dim]
        proprioception_future = proprioception_full[self.obs_horizon-1:]  # [pred_horizon, dim] (starting at t)
        
        action_history = action_full[:self.obs_horizon-1]  # [history_length, dim] (t-2, t-1) if h=3
        action_future = action_full[self.obs_horizon-1:]  # [pred_horizon, dim] (starting at t)

        view_action_history = view_action_full[:self.obs_horizon-1]  # [history_length, 9]
        view_action_future = view_action_full[self.obs_horizon-1:]  # [pred_horizon, 9] (starting at t)


        if self.use_relative:   
            
            delta_view_action_future = delta_view_action_full[self.obs_horizon-1:]  # [pred_horizon, 6] (starting at t)
            delta_view_action_for_uncondition_future = delta_view_action_for_uncondition_full[self.obs_horizon-1:]  # [pred_horizon, 6] (starting at t)
            delta_action_future = delta_action_full[self.obs_horizon-1:]  # [pred_horizon, 7] (starting at t)
            
            delta_point_tracking_future = delta_point_tracking_full[self.obs_horizon-1:]  # [pred_horizon, N(*num_obj), 3 or 4] (starting at t)

            delta_view_action_history = delta_view_action_full[:self.obs_horizon-1]  # [history_length-1, 6]
            delta_view_action_for_uncondition_history = delta_view_action_for_uncondition_full[:self.obs_horizon-1]  # [history_length-1, 6]
            delta_action_history = delta_action_full[:self.obs_horizon-1]  # [history_length-1, 7]
            delta_point_tracking_history = delta_point_tracking_full[:self.obs_horizon-1]  # [history_length-1, N(*num_obj), 3 or 4]
            
            
        initial_pixel_tracking = self.pixel_tracking_data[episode_idx].copy()[0]  # (N,3)
        
        episode_moving_mask = None

        
        episode_text = self.text[episode_idx] # Use string directly, not numpy array
        

        # Process history data (t-2, t-1, t)
        # random sample pixels and normalize roughly [0, 1]
        pixel_tracking_history, pixel_tracking_random_sample_indices = self.process_pixel_tracking_data(
            pixel_tracking_history,
            # self.num_points if not self.use_dift else num_dift_points,
            self.point_tracking_img_size, # point_tracking_img_size is [724(w), 543(h)] (for TAPIP3D) , [256(w), 256(h)] (for depth_proj)
        ) # [history_length, N, 3]
        
        # Process future data (t, t+1, ..., t+pred_horizon-1)
        # Use the same random sample indices for consistency
        pixel_tracking_future, _ = self.process_pixel_tracking_data(
            pixel_tracking_future,
            # self.num_points if not self.use_dift else num_dift_points,
            self.point_tracking_img_size,
        ) # [pred_horizon, N, 3]
        

        point_tracking_history_unnorm = point_tracking_history.copy()
        # random sample points and normalize roughly [-1, 1]
        point_tracking_history = self.process_point_tracking_data(
            point_tracking_history,
            # self.num_points if not self.use_dift else num_dift_points,
            pt_min=self.pt_min,
            pt_max=self.pt_max,
            flow_transforms=self.flow_transforms,
            pt_mean=self.pt_mean,
            pt_std=self.pt_std,
            norm_type=self.flow_norm_type,
        ) # [history_length, N(*num_obj), 3 or 4]

        point_tracking_future_unnorm = point_tracking_future.copy()
        point_tracking_future = self.process_point_tracking_data(
            point_tracking_future,
            # self.num_points if not self.use_dift else num_dift_points,
            pt_min=self.pt_min,
            pt_max=self.pt_max,
            flow_transforms=self.flow_transforms,
            pt_mean=self.pt_mean,
            pt_std=self.pt_std,
            norm_type=self.flow_norm_type,
        ) # [pred_horizon, N(*num_obj), 3 or 4]
        
        
        # shuffle the points (apply to both history and future)
        if self.shuffle_points and not self.use_dift_point_tracking: # for permutation invariance
        
            if point_tracking_history.shape[1] != self.num_points_per_mask: # multiple object -> object wise shuffling
                # [T, N*(num_obj), 3 or 4]
                num_obj = int(point_tracking_history.shape[1] / self.num_points_per_mask)
                T, N_num_obj, dim = point_tracking_history.shape
                point_tracking_history = point_tracking_history.reshape(T, self.num_points_per_mask, num_obj, dim) # [T, N, num_obj, 3 or 4]
                point_tracking_future = point_tracking_future.reshape(T, self.num_points_per_mask, num_obj, dim) # [T, N, num_obj, 3 or 4]
                point_tracking_history_unnorm = point_tracking_history_unnorm.reshape(T, self.num_points_per_mask, num_obj, dim) # [T, N, num_obj, 3 or 4]
                point_tracking_future_unnorm = point_tracking_future_unnorm.reshape(T, self.num_points_per_mask, num_obj, dim) # [T, N, num_obj, 3 or 4]


                shuffled_indices = np.random.permutation(point_tracking_history.shape[1])
                for obj_idx in range(num_obj):
                    point_tracking_history[:, :, obj_idx, :] = point_tracking_history[:, shuffled_indices, obj_idx, :] # [T, N, 3 or 4]
                    point_tracking_future[:, :, obj_idx, :] = point_tracking_future[:, shuffled_indices, obj_idx, :] # [T, N, 3 or 4]
                    point_tracking_history_unnorm[:, :, obj_idx, :] = point_tracking_history_unnorm[:, shuffled_indices, obj_idx, :] # [T, N, 3 or 4]
                    point_tracking_future_unnorm[:, :, obj_idx, :] = point_tracking_future_unnorm[:, shuffled_indices, obj_idx, :] # [T, N, 3 or 4]


            else:
                
                shuffled_indices = np.random.permutation(point_tracking_history.shape[1])

                point_tracking_history = point_tracking_history[..., shuffled_indices, :]
                point_tracking_future = point_tracking_future[..., shuffled_indices, :]
                point_tracking_history_unnorm = point_tracking_history_unnorm[..., shuffled_indices, :]
                point_tracking_future_unnorm = point_tracking_future_unnorm[..., shuffled_indices, :]

                pixel_tracking_history = pixel_tracking_history[:, shuffled_indices, :]
                pixel_tracking_future = pixel_tracking_future[:, shuffled_indices, :]

        
        if noise_aug:
            # Apply noise augmentation to point_flow_history
            # Randomly select 1 to N indices in the N direction
            
            N = point_tracking_history.shape[-2]
            num_indices_to_select = np.random.randint(1, N + 1)
            selected_indices = np.random.choice(N, size=num_indices_to_select, replace=False)
            noise = np.random.randn(point_tracking_history.shape[0], len(selected_indices), 3) * self.noise_aug_std
            point_tracking_history[..., selected_indices, :3] += noise
            
            # Apply noise augmentation to point_flow_future
            N = point_tracking_future.shape[-2]
            num_indices_to_select = np.random.randint(1, N + 1)
            selected_indices = np.random.choice(N, size=num_indices_to_select, replace=False)
            noise = np.random.randn(point_tracking_future.shape[0], len(selected_indices), 3) * self.noise_aug_std
            point_tracking_future[..., selected_indices, :3] += noise


            N = point_tracking_future_unnorm.shape[-2]
            num_indices_to_select = np.random.randint(1, N + 1)
            selected_indices = np.random.choice(N, size=num_indices_to_select, replace=False)
            noise = np.random.randn(point_tracking_future_unnorm.shape[0], len(selected_indices), 3) * self.noise_aug_std
            point_tracking_future_unnorm[..., selected_indices, :3] += noise

            N = point_tracking_history_unnorm.shape[-2]
            num_indices_to_select = np.random.randint(1, N + 1)
            selected_indices = np.random.choice(N, size=num_indices_to_select, replace=False)
            noise = np.random.randn(point_tracking_history_unnorm.shape[0], len(selected_indices), 3) * self.noise_aug_std
            point_tracking_history_unnorm[..., selected_indices, :3] += noise

            
            # Apply noise augmentation to pixel_flow_history
            N = pixel_tracking_history.shape[1]
            num_indices_to_select = np.random.randint(1, N + 1)
            selected_indices = np.random.choice(N, size=num_indices_to_select, replace=False)
            noise = np.random.randn(pixel_tracking_history.shape[0], len(selected_indices), pixel_tracking_history.shape[2]) * self.noise_aug_std
            pixel_tracking_history[:, selected_indices, :] += noise

            # Apply noise augmentation to pixel_flow_future
            N = pixel_tracking_future.shape[1]
            num_indices_to_select = np.random.randint(1, N + 1)
            selected_indices = np.random.choice(N, size=num_indices_to_select, replace=False)
            noise = np.random.randn(pixel_tracking_future.shape[0], len(selected_indices), pixel_tracking_future.shape[2]) * self.noise_aug_std
            pixel_tracking_future[:, selected_indices, :] += noise
            
            # Apply noise augmentation to proprioception_history
            noise = np.random.randn(*proprioception_history.shape) * self.noise_aug_std
            proprioception_history += noise
            proprioception_history_rotation_matrix = compute_rotation_matrix_from_ortho6d(proprioception_history[..., 3:9])
            proprioception_history[..., 3:9] = compute_ortho6d_from_rotation_matrix(proprioception_history_rotation_matrix) # ensure valid ortho6d (since this corresponds to valid proprio input in real-world deployment)
            # Apply noise augmentation to proprioception_future
            noise = np.random.randn(*proprioception_future.shape) * self.noise_aug_std
            proprioception_future += noise
            proprioception_future_rotation_matrix = compute_rotation_matrix_from_ortho6d(proprioception_future[..., 3:9])
            proprioception_future[..., 3:9] = compute_ortho6d_from_rotation_matrix(proprioception_future_rotation_matrix) # ensure valid ortho6d (since this corresponds to valid proprio input in real-world deployment)

        # Prepare the final sample
        nsample["camera_0"] = 0
        nsample["text"] = episode_text
        
        # History data (t-2, t-1, t)
        nsample["proprioception_history"] = proprioception_history  # [history_length, dim] (normalized)
        nsample["point_flow_history"] = point_tracking_history  # [history_length, N, 3 or 4] (normalized)
        nsample["pixel_flow_history"] = pixel_tracking_history  # [history_length, N, 3] (normalized)
        
        # Future data (t, t+1, ..., t+pred_horizon-1) - for action prediction
        nsample["proprioception"] = proprioception_future  # [pred_horizon, dim] (starting at t) (normalized)
        nsample["point_flow"] = point_tracking_future  # [pred_horizon, N, 3 or 4] (normalized)
        nsample["pixel_flow"] = pixel_tracking_future  # [pred_horizon, N, 3] (normalized)
        
        nsample["point_flow_history_unnorm"] = point_tracking_history_unnorm  # [history_length, N, 3 or 4] (unnormalized)
        nsample["point_flow_future_unnorm"] = point_tracking_future_unnorm  # [pred_horizon, N, 3 or 4] (unnormalized)
        
        
        nsample["action"] = action_future  # [pred_horizon, dim] (normalized)
        nsample["view_actions"] = view_action_future  # [pred_horizon, 6]
        
        nsample["action_history"] = action_history  # [history_length, dim] (normalized)
        nsample["view_action_history"] = view_action_history  # [history_length, 6]

        if self.use_relative:
            nsample["delta_view_action"] = delta_view_action_future  # [pred_horizon, 6]
            nsample["delta_view_action_for_uncondition"] = delta_view_action_for_uncondition_future  # [pred_horizon, 6]
            nsample["delta_action"] = delta_action_future  # [pred_horizon, 7]
            nsample["delta_point_flow"] = delta_point_tracking_future  # [pred_horizon, N, 3 or 4]

            nsample["delta_view_action_history"] = delta_view_action_history  # [history_length-1, 6]
            nsample["delta_view_action_for_uncondition_history"] = delta_view_action_for_uncondition_history  # [history_length-1, 6]
            nsample["delta_action_history"] = delta_action_history  # [history_length-1, 7]
            nsample["delta_point_flow_history"] = delta_point_tracking_history  # [history_length-1, N, 3 or 4]


        return nsample

