import os

import hydra
from hydra.core.global_hydra import GlobalHydra
import torch
import zarr

from egoavflow.common.utility.diffusion import (
    build_eval_dataset,
    load_flow_diffusion_model,
)
from egoavflow.common.utility.file import read_pickle
from egoavflow.common.utility.model import load_config
from egoavflow.diffusion_policy.utility.evaluation_active_vision import (
    evaluate_active_vision_wo_env,
)

@hydra.main(
    version_base=None,
    config_path="../../config/active_vision",
    config_name="evaluate_active_vision",
)
def eval(cfg):
    model_cfg = load_config(cfg.model_path)
    model, _ = load_flow_diffusion_model(
        cfg.model_path, cfg.ckpt, use_ema=model_cfg.training.use_ema, load_noise_scheduler=False
    )

    # create model
    view_model, _ = load_flow_diffusion_model(
        cfg.model_path, cfg.ckpt, use_ema=model_cfg.training.use_ema, load_view_model=True, load_noise_scheduler=False
    )

    view_num_inference_steps = model_cfg.num_inference_steps

    model.requires_grad_(False)
    model.to('cuda')
    model.eval()
    print('loaded model')
    view_model.requires_grad_(False)
    view_model.to('cuda')
    view_model.eval()
    print('loaded view model')
    if hasattr(model_cfg.training, 'use_pixel_flow_for_robot_policy') and model_cfg.training.use_pixel_flow_for_robot_policy:
        assert not model.predict_flow and not model.predict_separate_flow, (
            "use_pixel_flow_for_robot_policy only supports robot action prediction; "
            "future flow prediction should stay in flow_model."
        )
    
    # Assert: when use_flow_model=True, model should not predict flow
    if hasattr(model_cfg.training, 'use_flow_model') and model_cfg.training.use_flow_model:
        assert not model.predict_flow, f"When use_flow_model=True, model.predict_flow must be False, but got {model.predict_flow}"
        assert not model.predict_separate_flow, f"When use_flow_model=True, model.predict_separate_flow must be False, but got {model.predict_separate_flow}"
    
    # Load flow_model if use_flow_model is True
    flow_model = None
    if hasattr(model_cfg.training, 'use_flow_model') and model_cfg.training.use_flow_model:
        flow_model, _ = load_flow_diffusion_model(
            cfg.model_path, cfg.ckpt, use_ema=model_cfg.training.use_ema, load_flow_model=True, load_noise_scheduler=False
        )
        flow_model.requires_grad_(False)
        flow_model.to('cuda')
        flow_model.eval()
        print('loaded flow model')

    stats = read_pickle(os.path.join(cfg.model_path, "stats.pickle"))

    gt_dataset_args = (
        model_cfg.evaluation.gt_eval_dataset_args
        if model_cfg.evaluation.gt_eval_dataset_args is not None
        else {}
    )
    if cfg.data_dirs is not None:
        gt_dataset_args = {**gt_dataset_args, "data_dirs": list(cfg.data_dirs)}
    gt_eval_dataset = build_eval_dataset(
        cfg.model_path,
        max_episode=None,
        optional_transforms=[],
        stats_from_training = stats,
        **gt_dataset_args,
    )
    gt_data_buffers = [
        zarr.open(data_dir, mode="r") for data_dir in gt_eval_dataset.data_dirs
    ]
    with torch.no_grad():
        gt_result_save_path = os.path.join(
            cfg.model_path, "evaluation", f"epoch_{cfg.ckpt}", f"gt_flow(vis_rew_type={cfg.visibility_reward_type}_vis_aware_retarget={cfg.use_visibility_aware_retargeting})"
        )
        os.makedirs(gt_result_save_path, exist_ok=True)
        evaluate_active_vision_wo_env(
            model=model,
            view_model=view_model,
            flow_model=flow_model,
            view_num_inference_steps=view_num_inference_steps,
            stats=stats,
            num_samples=model_cfg.evaluation.gt_eval_num,
            eval_dataset=gt_eval_dataset,
            data_buffers=gt_data_buffers,
            result_save_path=gt_result_save_path,
            seed=cfg.seed,
            downsample_ratio=cfg.downsample_ratio,
            use_masking=model_cfg.training.use_masking,
            cfgs = model_cfg,
            use_fake_depth = model_cfg.training.use_fake_depth,
            slam_fake_value = model_cfg.training.slam_fake_value,
            use_depth_estimate = model_cfg.training.use_depth_estimate,
            policy_type = model_cfg.training.policy_type,
            tracker_type = model_cfg.training.tracker_type,
            visualize_raw_view_policy_output = model_cfg.training.visualize_raw_view_policy_output,
            droid = model_cfg.training.droid,
            T_B_M=cfg.T_B_M,
            T_B_M_view=cfg.T_B_M_view,
            T_E_C_view=cfg.T_E_C_view,
            ee_bias=cfg.ee_bias,
            use_mask_depth_filter=cfg.use_mask_depth_filter,
            mask_point_tracking_threshold=cfg.mask_point_tracking_threshold,
            num_points_per_mask=cfg.num_points_per_mask,
            visibility_reward_type=cfg.visibility_reward_type,
            use_retargeting=cfg.use_retargeting,
            use_visibility_aware_retargeting=cfg.use_visibility_aware_retargeting,
            use_future_action_for_flow_model=cfg.use_future_action_for_flow_model,
            query_point_indices=cfg.query_point_indices,
            z1_urdf_path=cfg.z1_urdf_path,
            z1_mesh_base_path=cfg.z1_mesh_base_path,
            widowx_urdf_path=cfg.widowx_urdf_path,
            widowx_mesh_base_path=cfg.widowx_mesh_base_path,
            cotracker_checkpoint=cfg.cotracker_checkpoint,
            max_steps=cfg.max_steps,
        )

if __name__ == "__main__":
    # Clear any existing Hydra instance to avoid re-initialization errors
    GlobalHydra.instance().clear()
    eval()
