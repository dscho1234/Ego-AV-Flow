import copy
import os
import pickle
import subprocess
import time

import hydra
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from egoavflow.common.utility.model import load_config

os.environ["WANDB_CONSOLE"] = "off"


def use_pixel_flow_for_robot_policy(cfg):
    return bool(getattr(cfg.training, "use_pixel_flow_for_robot_policy", False))


def get_robot_policy_input_flow_dim(cfg):
    return int(getattr(cfg.training, "robot_policy_input_flow_dim", cfg.training.input_flow_dim))


def build_flow_tokens(flow_history, input_flow_dim):
    B, H, N, D = flow_history.shape
    assert input_flow_dim <= D, f"input_flow_dim ({input_flow_dim}) must be <= flow dim ({D})"
    return flow_history[:, :, :, :input_flow_dim].permute(0, 2, 1, 3).reshape(B, N, -1)


def build_random_flow_mask(visibility, cfg, device):
    if not cfg.training.use_masking:
        return None
    if torch.rand(1).item() >= cfg.training.masking_ratio:
        return None

    flow_mask = visibility.bool().clone()
    B, N = flow_mask.shape
    for b in range(B):
        if not torch.any(flow_mask[b]):
            random_idx = torch.randint(0, N, (1,), device=device).item()
            flow_mask[b, random_idx] = True

    if cfg.training.additional_masking:
        for b in range(B):
            true_indices = torch.where(flow_mask[b])[0]
            M = len(true_indices)
            if M > 1:
                k = torch.randint(0, M, (1,)).item()
                if k > 0:
                    indices_to_mask = torch.randperm(M, device=device)[:k]
                    mask_positions = true_indices[indices_to_mask]
                    flow_mask[b, mask_positions] = False

    assert torch.any(flow_mask, dim=1).all(), f"at least one should be visible(true) along N axis, flow_mask : {flow_mask}"
    return flow_mask


@hydra.main(
    version_base=None,
    config_path="../../config/active_vision",
    config_name="train_active_vision",
)
def train(cfg: DictConfig):
    # print gpu devices
    print(f"Using GPU devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps
    )

    # create save dir
    output_dir = HydraConfig.get().runtime.output_dir
    resume = cfg.training.resume
    if resume:
        # overwrite the config with the checkpoint config
        accelerator.print("resume from checkpoint")
        model_path = cfg.training.model_path
        model_ckpt = cfg.training.model_ckpt
        cfg = load_config(model_path)
        # save the checkpoint config to the output dir
        OmegaConf.save(cfg, os.path.join(output_dir, ".hydra", "config.yaml"))
    if accelerator.is_local_main_process:
        wandb.init(name = cfg.exp_name, project=cfg.project_name)
        wandb.config.update(OmegaConf.to_container(cfg))
        accelerator.print("Logging dir", output_dir)
        ckpt_save_dir = os.path.join(output_dir, "checkpoints")
        state_save_dir = os.path.join(output_dir, "state")
        os.makedirs(ckpt_save_dir, exist_ok=True)
        os.makedirs(state_save_dir, exist_ok=True)

    dataset = hydra.utils.instantiate(
        cfg.dataset, max_episode=5 if cfg.debug else cfg.dataset.max_episode
    )
    print("Total training samples:", len(dataset))
    # save training data statistics (min, max) for each dim
    stats = dataset.stats
    # open a file for writing in binary mode
    with open(os.path.join(output_dir, "stats.pickle"), "wb") as f:
        # write the dictionary to the file
        pickle.dump(stats, f)
    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=cfg.training.shuffle,
        pin_memory=cfg.training.pin_memory,
        persistent_workers=cfg.training.persistent_workers,
        drop_last=cfg.training.drop_last,
    )
    print("====================================")
    print("len of dataset", len(dataset))
    print("====================================")

    sample_batch = next(iter(dataloader))
    for k, v in sample_batch.items():
        if type(v) == torch.Tensor:
            if k != "text":
                accelerator.print(k, v.shape)

    # Create eval dataset
    eval_dataset = None
    eval_dataloader = None

    # Merge eval-specific config with base dataset config
    eval_dataset_cfg = OmegaConf.merge(cfg.dataset, cfg.evaluation.gt_eval_dataset_args)
    # Filter data_dirs to only include paths with 'validation'
    if 'data_dirs' in eval_dataset_cfg:
        eval_dataset_cfg.data_dirs = [d for d in eval_dataset_cfg.data_dirs if 'validation' in d]
    eval_dataset = hydra.utils.instantiate(
        eval_dataset_cfg, stats_from_training=stats
    )
    print("Total eval samples:", len(eval_dataset))
    
    # Create eval dataloader
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=False,  # Don't shuffle eval data
        pin_memory=cfg.training.pin_memory,
        persistent_workers=cfg.training.persistent_workers,
        drop_last=False,  # Don't drop last for eval
    )

    # Instantiate models from config
    # Set dtype based on config
    dtype = torch.bfloat16 if cfg.rdt.use_bfloat16 else torch.float32
    model = hydra.utils.instantiate(cfg.model, dtype=dtype)
    view_model = hydra.utils.instantiate(cfg.view_model, dtype=dtype)
    if use_pixel_flow_for_robot_policy(cfg):
        assert not model.predict_flow and not model.predict_separate_flow, (
            "use_pixel_flow_for_robot_policy uses pixel-flow conditions for policy/view/flow models. "
            "Disable model.predict_flow/model.predict_separate_flow and keep future flow prediction in flow_model."
        )
    
    # Assert: when use_flow_model=True, model should not predict flow
    if cfg.training.use_flow_model:
        assert not model.predict_flow, f"When use_flow_model=True, model.predict_flow must be False, but got {model.predict_flow}"
        assert not model.predict_separate_flow, f"When use_flow_model=True, model.predict_separate_flow must be False, but got {model.predict_separate_flow}"
    
    # Instantiate flow_model if use_flow_model is True
    flow_model = None
    if cfg.training.use_flow_model:
        flow_model = hydra.utils.instantiate(cfg.flow_model, dtype=dtype)
        # Assert: flow_model should not predict flow (predict_flow and predict_separate_flow should be False)
        # flow_model's action corresponds to flow, so it should only predict action (which is flow)
        assert not flow_model.predict_flow, f"flow_model.predict_flow must be False, but got {flow_model.predict_flow}"
        assert not flow_model.predict_separate_flow, f"flow_model.predict_separate_flow must be False, but got {flow_model.predict_separate_flow}"
    
    # Get condition_ratio from config (for random conditional/unconditional training)
    condition_ratio = cfg.training.condition_ratio

    # Create optimizer with flow_model parameters if use_flow_model is True
    if cfg.training.use_flow_model:
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=list(model.parameters())+list(view_model.parameters())+list(flow_model.parameters()))
    else:
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=list(model.parameters())+list(view_model.parameters()))
    num_update_steps_per_epoch = len(dataloader)
    max_train_steps = cfg.training.epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.num_warmup_steps, # * accelerator.num_processes,
        num_training_steps=max_train_steps, #  * accelerator.num_processes,
    )
    
    print(f'max train steps: {max_train_steps}, num processes: {accelerator.num_processes}')


    # Prepare everything with our `accelerator`.
    if cfg.training.use_flow_model:
        if eval_dataloader is not None:
            model, view_model, flow_model, optimizer, dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                model, view_model, flow_model, optimizer, dataloader, eval_dataloader, lr_scheduler
            )
        else:
            model, view_model, flow_model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
                model, view_model, flow_model, optimizer, dataloader, lr_scheduler
            )
    else:
        if eval_dataloader is not None:
            model, view_model, optimizer, dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                model, view_model, optimizer, dataloader, eval_dataloader, lr_scheduler
            )
        else:
            model, view_model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
                model, view_model, optimizer, dataloader, lr_scheduler
            )


    if resume:
        model_state = os.path.join(model_path, "state", f"epoch_{model_ckpt}")
        accelerator.load_state(model_state)
        print("successfully loaded model from checkpoint!")

    if cfg.training.use_ema:
        ema_model = copy.deepcopy(accelerator.unwrap_model(model))
        ema_view_model = copy.deepcopy(accelerator.unwrap_model(view_model))
        ema = hydra.utils.instantiate(cfg.ema, model=ema_model)
        ema_view = hydra.utils.instantiate(cfg.ema, model=ema_view_model)
        if cfg.training.use_flow_model:
            ema_flow_model = copy.deepcopy(accelerator.unwrap_model(flow_model))
            ema_flow = hydra.utils.instantiate(cfg.ema, model=ema_flow_model)


    model.train()
    if cfg.training.use_ema:
        ema_model.train()
    view_model.train()
    if cfg.training.use_ema:
        ema_view_model.train()
    if cfg.training.use_flow_model:
        flow_model.train()
        if cfg.training.use_ema:
            ema_flow_model.train()

    for epoch in range(cfg.training.epochs):
        epoch_loss = []
        epoch_action_loss = []
        epoch_view_action_loss = []
        epoch_flow_loss = []
        epoch_flow_model_loss = []
        start = time.time()
        
        # Initialize eval metrics (will be computed if needed)
        eval_metrics = None
        
        for batch in dataloader:
            with accelerator.accumulate(model, view_model):

                B, H, N, D = batch["point_flow_history"].shape
                point_flow_history = batch["point_flow_history"].float().to(accelerator.device) # [B, history_length, N, D]
                proprioception_history = batch["proprioception_history"].float().to(accelerator.device) # [B, history_length, D]
                point_flow_history_unnorm = batch["point_flow_history_unnorm"].float().to(accelerator.device) # [B, history_length, N, 3 or 4]
                if D == 4:
                    visibility = (point_flow_history_unnorm[:, -1, :, -1] == 1) # [B, N] - only the last timestep
                    flow_mask = build_random_flow_mask(visibility, cfg, accelerator.device)
                else:
                    flow_mask = None

                if use_pixel_flow_for_robot_policy(cfg):
                    condition_flow_history = batch["pixel_flow_history"].float().to(accelerator.device) # [B, history_length, N, 3], normalized x/y/visibility
                    assert condition_flow_history.shape[:3] == point_flow_history.shape[:3], (
                        f"pixel_flow_history shape {condition_flow_history.shape} should match point_flow_history shape {point_flow_history.shape} in B/H/N"
                    )
                    condition_visibility = (condition_flow_history[:, -1, :, -1] == 1)
                    condition_flow_mask = build_random_flow_mask(condition_visibility, cfg, accelerator.device)
                else:
                    condition_flow_history = point_flow_history
                    condition_flow_mask = flow_mask
                
                # Apply masking to proprioception_history with probability masking_ratio
                if cfg.training.use_proprio_masking:
                    # Apply masking with probability masking_ratio
                    if torch.rand(1).item() < cfg.training.masking_ratio:
                        B, H, _ = proprioception_history.shape
                        proprio_mask = torch.ones(B, H, dtype=torch.bool, device=accelerator.device)

                        # For each batch, randomly mask 0 to H-1 timesteps
                        for b in range(B):
                            # Randomly choose k from 0 to H-1 (inclusive) - number of timesteps to mask
                            k = torch.randint(0, H, (1,)).item()
                            
                            if k > 0:
                                # Randomly select k timesteps to mask (set to False)
                                timesteps_to_mask = torch.randperm(H, device=accelerator.device)[:k]
                                proprio_mask[b, timesteps_to_mask] = False
                        assert torch.any(proprio_mask, dim=1).all(), f"at least one should be true along H axis, proprio_mask : {proprio_mask}"
                    else:
                        proprio_mask = None
                else:
                    proprio_mask = None
                
                # Flow conditions can optionally use 2D pixel flow for all RDT models.
                flow_tokens = build_flow_tokens(condition_flow_history, get_robot_policy_input_flow_dim(cfg))
                condition_flow_tokens = flow_tokens
                
                
                if cfg.use_relative:
                    actions = batch["delta_action"].float().to(accelerator.device) # [B, T, 7]
                    view_actions = batch["delta_view_action"].float().to(accelerator.device) # [B, T, 6]
                    view_actions_for_uncondition = batch["delta_view_action_for_uncondition"].float().to(accelerator.device) # [B, T, 6]
                    point_flows = batch["delta_point_flow"].float().to(accelerator.device) # [B, T, N, 3 or 4], normalized
                    if cfg.training.use_ptp:
                        actions_history = batch["delta_action_history"].float().to(accelerator.device) # [B, H-1, 7]
                        view_actions_history = batch["delta_view_action_history"].float().to(accelerator.device) # [B, H-1, 6]
                        view_actions_history_for_uncondition = batch["delta_view_action_for_uncondition_history"].float().to(accelerator.device) # [B, H-1, 6]
                        point_flows_history = batch["delta_point_flow_history"].float().to(accelerator.device) # [B, H-1, N, 3 or 4], normalized
                        
                        actions = torch.cat([actions_history, actions], dim=1) # [B, (H-1)+T, 7]
                        view_actions = torch.cat([view_actions_history, view_actions], dim=1) # [B, (H-1)+T, 6]
                        view_actions_for_uncondition = torch.cat([view_actions_history_for_uncondition, view_actions_for_uncondition], dim=1) # [B, (H-1)+T, 6]
                        point_flows = torch.cat([point_flows_history, point_flows], dim=1) # [B, (H-1)+T, N, 3 or 4]
                    
                else:
                    actions = batch["action"].float().to(accelerator.device) # [B, T, dim]
                    view_actions = batch["view_actions"].float().to(accelerator.device) # [B, T, 9]
                    point_flows = batch["point_flow"].float().to(accelerator.device) # [B, T, N, D], normalized
                    if cfg.training.use_ptp:
                        actions_history = batch["action_history"].float().to(accelerator.device) # [B, H-1, dim]
                        view_actions_history = batch["view_action_history"].float().to(accelerator.device) # [B, H-1, 9]
                        point_flows_history = batch["point_flow_history"].float().to(accelerator.device) # [B, H-1, N, D], normalized

                        actions = torch.cat([actions_history, actions], dim=1) # [B, (H-1)+T, dim]
                        view_actions = torch.cat([view_actions_history, view_actions], dim=1) # [B, (H-1)+T, 6]
                        point_flows = torch.cat([point_flows_history, point_flows], dim=1) # [B, (H-1)+T, N, 3 or 4]

                
                B, T, N, D = point_flows.shape # [B, T or (H-1)+T, N, 3 or 4]

                point_flows = point_flows.reshape(B, T, -1) # [B, T or (H-1)+T, N*D]
                
                
                # Proprioception tokens: [B, H, D] -> already in correct shape
                proprio_tokens = proprioception_history  # [B, H, proprio_token_dim]

                
                viewpoint_tokens = batch["view_action_history"].float().to(accelerator.device)  # [B, H, 9], normalized

                # Apply masking to viewpoint_history with probability masking_ratio
                if cfg.training.use_viewpoint_masking:
                    # Apply masking with probability masking_ratio
                    if torch.rand(1).item() < cfg.training.masking_ratio:
                        B, H, _ = viewpoint_tokens.shape
                        viewpoint_mask = torch.ones(B, H, dtype=torch.bool, device=accelerator.device)
                        
                        # For each batch, randomly mask 0 to H-1 timesteps
                        for b in range(B):
                            # Randomly choose k from 0 to H-1 (inclusive) - number of timesteps to mask
                            k = torch.randint(0, H, (1,)).item()
                            
                            if k > 0:
                                # Randomly select k timesteps to mask (set to False)
                                timesteps_to_mask = torch.randperm(H, device=accelerator.device)[:k]
                                viewpoint_mask[b, timesteps_to_mask] = False
                        assert torch.any(viewpoint_mask, dim=1).all(), f"at least one should be true along H axis, viewpoint_mask : {viewpoint_mask}"
                    else:
                        viewpoint_mask = None
                else:
                    viewpoint_mask = None

                # Randomly decide use_condition based on condition_ratio for each batch
                use_condition = np.random.rand() < condition_ratio
                
                # Prepare tokens for view_model
                # If view_obs_horizon == 1, use only the latest timestep (index -1)
                if cfg.dataset.view_obs_horizon == 1:
                    # Extract only the last timestep for view_model
                    # Flow tokens: extract last timestep before reshaping
                    view_flow_tokens = build_flow_tokens(condition_flow_history[:, -1:, :, :], get_robot_policy_input_flow_dim(cfg))
                    
                    # Proprioception tokens: extract last timestep
                    view_proprio_tokens = proprioception_history[:, -1:, :]  # [B, 1, proprio_token_dim]
                    
                    # Viewpoint tokens: extract last timestep
                    view_viewpoint_tokens = viewpoint_tokens[:, -1:, :]  # [B, 1, 9]
                    
                    # All masks should be None when input is a single timestep
                    view_flow_mask = None
                    view_proprio_mask = None
                    view_viewpoint_mask = None
                else:
                    # Use all history for view_model
                    view_flow_tokens = condition_flow_tokens
                    view_proprio_tokens = proprio_tokens
                    view_viewpoint_tokens = viewpoint_tokens
                    view_flow_mask = condition_flow_mask
                    view_proprio_mask = proprio_mask
                    view_viewpoint_mask = viewpoint_mask
                
                # Prepare condition tokens based on use_condition
                if use_condition:
                    # Conditional: use actual tokens, append 1 to last dimension
                    # view_flow_tokens: [B, N, D] -> [B, N, D+1]
                    if view_flow_tokens is not None:
                        condition_flag = torch.ones(B, N, 1, dtype=view_flow_tokens.dtype, device=view_flow_tokens.device)
                        view_flow_tokens = torch.cat([view_flow_tokens, condition_flag], dim=-1)
                    
                    # view_proprio_tokens: [B, H, D] -> [B, H, D+1]
                    if view_proprio_tokens is not None:
                        H_proprio = view_proprio_tokens.shape[1]
                        condition_flag = torch.ones(B, H_proprio, 1, dtype=view_proprio_tokens.dtype, device=view_proprio_tokens.device)
                        view_proprio_tokens = torch.cat([view_proprio_tokens, condition_flag], dim=-1)
                    
                    # view_viewpoint_tokens: [B, H, 9] -> [B, H, 10]
                    if view_viewpoint_tokens is not None:
                        H_viewpoint = view_viewpoint_tokens.shape[1]
                        condition_flag = torch.ones(B, H_viewpoint, 1, dtype=view_viewpoint_tokens.dtype, device=view_viewpoint_tokens.device)
                        view_viewpoint_tokens = torch.cat([view_viewpoint_tokens, condition_flag], dim=-1)
                else:
                    # Unconditional: use zero tensors for conditions, append 0 to last dimension
                    # view_flow_tokens: create zero tensor with same shape, append 0
                    if view_flow_tokens is not None:
                        flow_token_dim = view_flow_tokens.shape[-1]
                        view_flow_tokens = torch.zeros(B, N, flow_token_dim, dtype=view_flow_tokens.dtype, device=view_flow_tokens.device)
                        condition_flag = torch.zeros(B, N, 1, dtype=view_flow_tokens.dtype, device=view_flow_tokens.device)
                        view_flow_tokens = torch.cat([view_flow_tokens, condition_flag], dim=-1)
                    
                    # view_proprio_tokens: create zero tensor with same shape, append 0
                    if view_proprio_tokens is not None:
                        H_proprio = view_proprio_tokens.shape[1]
                        proprio_token_dim = view_proprio_tokens.shape[-1]
                        view_proprio_tokens = torch.zeros(B, H_proprio, proprio_token_dim, dtype=view_proprio_tokens.dtype, device=view_proprio_tokens.device)
                        condition_flag = torch.zeros(B, H_proprio, 1, dtype=view_proprio_tokens.dtype, device=view_proprio_tokens.device)
                        view_proprio_tokens = torch.cat([view_proprio_tokens, condition_flag], dim=-1)
                    
                    # view_viewpoint_tokens: create zero tensor with same shape, append 0
                    if view_viewpoint_tokens is not None:
                        H_viewpoint = view_viewpoint_tokens.shape[1]
                        viewpoint_token_dim = view_viewpoint_tokens.shape[-1]
                        view_viewpoint_tokens = torch.zeros(B, H_viewpoint, viewpoint_token_dim, dtype=view_viewpoint_tokens.dtype, device=view_viewpoint_tokens.device)
                        condition_flag = torch.zeros(B, H_viewpoint, 1, dtype=view_viewpoint_tokens.dtype, device=view_viewpoint_tokens.device)
                        view_viewpoint_tokens = torch.cat([view_viewpoint_tokens, condition_flag], dim=-1)

                # Compute loss with fix_mask support
                # We need to compute per-timestep loss to apply fix_mask
                batch_size = flow_tokens.shape[0]
                

                action_loss, flow_loss, loss = \
                    model(cfg, batch_size, actions, flow_tokens, proprio_tokens, 
                    flow_mask=condition_flow_mask, proprio_mask=proprio_mask, point_flows=point_flows)
                
                if use_condition:
                    view_action_loss, view_flow_loss, view_loss = \
                        view_model(cfg, batch_size, view_actions, view_flow_tokens, view_proprio_tokens, 
                        flow_mask=view_flow_mask, proprio_mask=view_proprio_mask, point_flows=None, viewpoint_tokens=view_viewpoint_tokens, viewpoint_mask=view_viewpoint_mask)
                else:
                    view_action_loss, view_flow_loss, view_loss = \
                        view_model(cfg, batch_size, view_actions_for_uncondition, view_flow_tokens, view_proprio_tokens, 
                        flow_mask=view_flow_mask, proprio_mask=view_proprio_mask, point_flows=None, viewpoint_tokens=view_viewpoint_tokens, viewpoint_mask=view_viewpoint_mask)
                
                # Compute flow_model loss if use_flow_model is True
                flow_model_action_loss = None
                if cfg.training.use_flow_model:
                    # flow_model predicts future point flow
                    
                    flow_actions = point_flows  # [B, T, N*D]
                    
                    # Prepare future_action_tokens from batch["action"] (not delta_action)
                    # Use batch["action"] directly regardless of use_relative flag
                    future_action_tokens_raw = batch["action"].float().to(accelerator.device)  # [B, T, action_dim]
                    future_action_tokens = future_action_tokens_raw
                
                    
                    # Apply masking to future_action_tokens with probability masking_ratio (similar to proprio_mask)
                    if cfg.training.use_future_action_masking:
                        # Apply masking with probability masking_ratio
                        if torch.rand(1).item() < cfg.training.masking_ratio:
                            B_future, T_future, _ = future_action_tokens.shape
                            future_action_mask = torch.ones(B_future, T_future, dtype=torch.bool, device=accelerator.device)
                            
                            # For each batch, randomly mask 0 to T_future-1 timesteps
                            for b in range(B_future):
                                # Randomly choose k from 0 to T_future-1 (inclusive) - number of timesteps to mask
                                k = torch.randint(0, T_future, (1,)).item()
                                
                                if k > 0:
                                    # Randomly select k timesteps to mask (set to False)
                                    timesteps_to_mask = torch.randperm(T_future, device=accelerator.device)[:k]
                                    future_action_mask[b, timesteps_to_mask] = False
                            assert torch.any(future_action_mask, dim=1).all(), f"at least one should be true along T axis, future_action_mask : {future_action_mask}"
                        else:
                            future_action_mask = None
                    else:
                        future_action_mask = None
                    
                    # flow_model forward: input is flow_actions (future point flow), conditions are flow_tokens, proprio_tokens, viewpoint_tokens, and future_action_tokens
                    # Note: flow_model doesn't use point_flows parameter, so we pass None
                    # flow_model's action_loss corresponds to flow prediction loss (since action_dim = flow_output_dim)
                    flow_model_action_loss, flow_model_flow_loss, flow_model_loss = \
                        flow_model(cfg, batch_size, flow_actions, condition_flow_tokens, proprio_tokens,
                        flow_mask=condition_flow_mask, proprio_mask=proprio_mask, point_flows=None, viewpoint_tokens=viewpoint_tokens, viewpoint_mask=viewpoint_mask,
                        future_action_tokens=future_action_tokens, future_action_mask=future_action_mask)
                
                
                # total loss (loss already includes flow_loss if predict_flow is True)
                total_loss = loss + view_action_loss
                if cfg.training.use_flow_model:
                    total_loss = total_loss + flow_model_action_loss  # Use flow_model_action_loss (which is flow prediction loss)

                # optimize
                optimizer.zero_grad()
                accelerator.backward(total_loss)
                lr_scheduler.step()
                optimizer.step()
                # update ema
                if cfg.training.use_ema:
                    ema.step(accelerator.unwrap_model(model))
                    ema_view.step(accelerator.unwrap_model(view_model))
                    if cfg.training.use_flow_model:
                        ema_flow.step(accelerator.unwrap_model(flow_model))
                # logging
                epoch_loss.append(total_loss.item())
                epoch_action_loss.append(action_loss.item())
                epoch_view_action_loss.append(view_action_loss.item())
                epoch_flow_loss.append(flow_loss.item())
                if cfg.training.use_flow_model:
                    epoch_flow_model_loss.append(flow_model_action_loss.item())  # Use flow_model_action_loss
            
        # Compute eval loss if needed (before logging)
        if epoch % cfg.training.eval_logging_frequency == 0:
            # Compute loss for eval data
            if eval_dataloader is not None and accelerator.is_local_main_process and not cfg.debug:
                model.eval()
                view_model.eval()
                if cfg.training.use_flow_model:
                    flow_model.eval()
                eval_epoch_loss = []
                eval_epoch_action_loss = []
                eval_epoch_view_action_loss = []
                eval_epoch_flow_loss = []
                eval_epoch_flow_model_loss = []
                
                with torch.no_grad():
                    for eval_batch in eval_dataloader:
                        B, H, N, D = eval_batch["point_flow_history"].shape
                        point_flow_history = eval_batch["point_flow_history"].float().to(accelerator.device) # [B, history_length, N, D]
                        proprioception_history = eval_batch["proprioception_history"].float().to(accelerator.device) # [B, history_length, D]
                        point_flow_history_unnorm = eval_batch["point_flow_history_unnorm"].float().to(accelerator.device) # [B, history_length, N, 3 or 4]
                        if D == 4:
                            visibility = (point_flow_history_unnorm[:, -1, :, -1] == 1) # [B, N] - only the last timestep
                            flow_mask = build_random_flow_mask(visibility, cfg, accelerator.device)
                        else:
                            flow_mask = None
                        
                        if use_pixel_flow_for_robot_policy(cfg):
                            condition_flow_history = eval_batch["pixel_flow_history"].float().to(accelerator.device) # [B, history_length, N, 3], normalized x/y/visibility
                            assert condition_flow_history.shape[:3] == point_flow_history.shape[:3], (
                                f"pixel_flow_history shape {condition_flow_history.shape} should match point_flow_history shape {point_flow_history.shape} in B/H/N"
                            )
                            condition_visibility = (condition_flow_history[:, -1, :, -1] == 1)
                            condition_flow_mask = build_random_flow_mask(condition_visibility, cfg, accelerator.device)
                        else:
                            condition_flow_history = point_flow_history
                            condition_flow_mask = flow_mask

                        # Apply masking to proprioception_history with probability masking_ratio
                        if cfg.training.use_proprio_masking:
                            # Apply masking with probability masking_ratio
                            if torch.rand(1).item() < cfg.training.masking_ratio:
                                B, H, _ = proprioception_history.shape
                                proprio_mask = torch.ones(B, H, dtype=torch.bool, device=accelerator.device)
                                
                                # For each batch, randomly mask 0 to H-1 timesteps
                                for b in range(B):
                                    # Randomly choose k from 0 to H-1 (inclusive) - number of timesteps to mask
                                    k = torch.randint(0, H, (1,)).item()
                                    
                                    if k > 0:
                                        # Randomly select k timesteps to mask (set to False)
                                        timesteps_to_mask = torch.randperm(H, device=accelerator.device)[:k]
                                        proprio_mask[b, timesteps_to_mask] = False
                                assert torch.any(proprio_mask, dim=1).all(), f"at least one should be true along H axis, proprio_mask : {proprio_mask}"
                            else:
                                proprio_mask = None
                        else:
                            proprio_mask = None
                        
                        # Flow conditions can optionally use 2D pixel flow for all RDT models.
                        flow_tokens = build_flow_tokens(condition_flow_history, get_robot_policy_input_flow_dim(cfg))
                        condition_flow_tokens = flow_tokens
                        
                        
                        if cfg.use_relative:
                            actions = eval_batch["delta_action"].float().to(accelerator.device) # [B, T, 7]
                            view_actions = eval_batch["delta_view_action"].float().to(accelerator.device) # [B, T, 6]
                            view_actions_for_uncondition = eval_batch["delta_view_action_for_uncondition"].float().to(accelerator.device) # [B, T, 6]
                            point_flows = eval_batch["delta_point_flow"].float().to(accelerator.device) # [B, T, N, 3 or 4], normalized
                            if cfg.training.use_ptp:
                                actions_history = eval_batch["delta_action_history"].float().to(accelerator.device) # [B, H-1, 7]
                                view_actions_history = eval_batch["delta_view_action_history"].float().to(accelerator.device) # [B, H-1, 6]
                                view_actions_history_for_uncondition = eval_batch["delta_view_action_for_uncondition_history"].float().to(accelerator.device) # [B, H-1, 6]
                                point_flows_history = eval_batch["delta_point_flow_history"].float().to(accelerator.device) # [B, H-1, N, 3 or 4], normalized
                                
                                actions = torch.cat([actions_history, actions], dim=1) # [B, (H-1)+T, 7]
                                view_actions = torch.cat([view_actions_history, view_actions], dim=1) # [B, (H-1)+T, 6]
                                view_actions_for_uncondition = torch.cat([view_actions_history_for_uncondition, view_actions_for_uncondition], dim=1) # [B, (H-1)+T, 6]
                                point_flows = torch.cat([point_flows_history, point_flows], dim=1) # [B, (H-1)+T, N, 3 or 4]
                            
                        else:
                            actions = eval_batch["action"].float().to(accelerator.device) # [B, T, dim]
                            view_actions = eval_batch["view_actions"].float().to(accelerator.device) # [B, T, 9]
                            point_flows = eval_batch["point_flow"].float().to(accelerator.device) # [B, T, N, D], normalized
                            if cfg.training.use_ptp:
                                actions_history = eval_batch["action_history"].float().to(accelerator.device) # [B, H-1, dim]
                                view_actions_history = eval_batch["view_action_history"].float().to(accelerator.device) # [B, H-1, 9]
                                point_flows_history = eval_batch["point_flow_history"].float().to(accelerator.device) # [B, H-1, N, D], normalized

                                actions = torch.cat([actions_history, actions], dim=1) # [B, (H-1)+T, dim]
                                view_actions = torch.cat([view_actions_history, view_actions], dim=1) # [B, (H-1)+T, 6]
                                point_flows = torch.cat([point_flows_history, point_flows], dim=1) # [B, (H-1)+T, N, 3 or 4]

                        
                        B, T, N, D = point_flows.shape # [B, T or (H-1)+T, N, 3 or 4]

                        point_flows = point_flows.reshape(B, T, -1) # [B, T or (H-1)+T, N*D]
                        
                        
                        # Proprioception tokens: [B, H, D] -> already in correct shape
                        proprio_tokens = proprioception_history  # [B, H, proprio_token_dim]

                        # Viewpoint tokens: [B, H, 9] -> already in correct shape
                        viewpoint_tokens = eval_batch["view_action_history"].float().to(accelerator.device)  # [B, H, 9], normalized

                        # Apply masking to viewpoint_history with probability masking_ratio
                        if cfg.training.use_viewpoint_masking:
                            # Apply masking with probability masking_ratio
                            if torch.rand(1).item() < cfg.training.masking_ratio:
                                B, H, _ = viewpoint_tokens.shape
                                viewpoint_mask = torch.ones(B, H, dtype=torch.bool, device=accelerator.device)
                                
                                # For each batch, randomly mask 0 to H-1 timesteps
                                for b in range(B):
                                    # Randomly choose k from 0 to H-1 (inclusive) - number of timesteps to mask
                                    k = torch.randint(0, H, (1,)).item()
                                    
                                    if k > 0:
                                        # Randomly select k timesteps to mask (set to False)
                                        timesteps_to_mask = torch.randperm(H, device=accelerator.device)[:k]
                                        viewpoint_mask[b, timesteps_to_mask] = False
                                assert torch.any(viewpoint_mask, dim=1).all(), f"at least one should be true along H axis, viewpoint_mask : {viewpoint_mask}"
                            else:
                                viewpoint_mask = None
                        else:
                            viewpoint_mask = None

                        # For eval, always use condition=True
                        use_condition_eval = True
                        
                        # Prepare tokens for view_model
                        # If view_obs_horizon == 1, use only the latest timestep (index -1)
                        if cfg.dataset.view_obs_horizon == 1:
                            # Extract only the last timestep for view_model
                            # Flow tokens: extract last timestep before reshaping
                            view_flow_tokens = build_flow_tokens(condition_flow_history[:, -1:, :, :], get_robot_policy_input_flow_dim(cfg))
                            
                            # Proprioception tokens: extract last timestep
                            view_proprio_tokens = proprioception_history[:, -1:, :]  # [B, 1, proprio_token_dim]
                            
                            # Viewpoint tokens: extract last timestep
                            view_viewpoint_tokens = viewpoint_tokens[:, -1:, :]  # [B, 1, 9]
                            
                            # All masks should be None when input is a single timestep
                            view_flow_mask = None
                            view_proprio_mask = None
                            view_viewpoint_mask = None
                        else:
                            # Use all history for view_model
                            view_flow_tokens = condition_flow_tokens
                            view_proprio_tokens = proprio_tokens
                            view_viewpoint_tokens = viewpoint_tokens
                            view_flow_mask = condition_flow_mask
                            view_proprio_mask = proprio_mask
                            view_viewpoint_mask = viewpoint_mask
                        
                        # Prepare condition tokens based on use_condition_eval (always True for eval)
                        if use_condition_eval:
                            # Conditional: use actual tokens, append 1 to last dimension
                            # view_flow_tokens: [B, N, D] -> [B, N, D+1]
                            if view_flow_tokens is not None:
                                condition_flag = torch.ones(B, N, 1, dtype=view_flow_tokens.dtype, device=view_flow_tokens.device)
                                view_flow_tokens = torch.cat([view_flow_tokens, condition_flag], dim=-1)
                            
                            # view_proprio_tokens: [B, H, D] -> [B, H, D+1]
                            if view_proprio_tokens is not None:
                                H_proprio = view_proprio_tokens.shape[1]
                                condition_flag = torch.ones(B, H_proprio, 1, dtype=view_proprio_tokens.dtype, device=view_proprio_tokens.device)
                                view_proprio_tokens = torch.cat([view_proprio_tokens, condition_flag], dim=-1)
                            
                            # view_viewpoint_tokens: [B, H, 9] -> [B, H, 10]
                            if view_viewpoint_tokens is not None:
                                H_viewpoint = view_viewpoint_tokens.shape[1]
                                condition_flag = torch.ones(B, H_viewpoint, 1, dtype=view_viewpoint_tokens.dtype, device=view_viewpoint_tokens.device)
                                view_viewpoint_tokens = torch.cat([view_viewpoint_tokens, condition_flag], dim=-1)
                        else:
                            # Unconditional: use zero tensors for conditions, append 0 to last dimension
                            # view_flow_tokens: create zero tensor with same shape, append 0
                            if view_flow_tokens is not None:
                                flow_token_dim = view_flow_tokens.shape[-1]
                                view_flow_tokens = torch.zeros(B, N, flow_token_dim, dtype=view_flow_tokens.dtype, device=view_flow_tokens.device)
                                condition_flag = torch.zeros(B, N, 1, dtype=view_flow_tokens.dtype, device=view_flow_tokens.device)
                                view_flow_tokens = torch.cat([view_flow_tokens, condition_flag], dim=-1)
                            
                            # view_proprio_tokens: create zero tensor with same shape, append 0
                            if view_proprio_tokens is not None:
                                H_proprio = view_proprio_tokens.shape[1]
                                proprio_token_dim = view_proprio_tokens.shape[-1]
                                view_proprio_tokens = torch.zeros(B, H_proprio, proprio_token_dim, dtype=view_proprio_tokens.dtype, device=view_proprio_tokens.device)
                                condition_flag = torch.zeros(B, H_proprio, 1, dtype=view_proprio_tokens.dtype, device=view_proprio_tokens.device)
                                view_proprio_tokens = torch.cat([view_proprio_tokens, condition_flag], dim=-1)
                            
                            # view_viewpoint_tokens: create zero tensor with same shape, append 0
                            if view_viewpoint_tokens is not None:
                                H_viewpoint = view_viewpoint_tokens.shape[1]
                                viewpoint_token_dim = view_viewpoint_tokens.shape[-1]
                                view_viewpoint_tokens = torch.zeros(B, H_viewpoint, viewpoint_token_dim, dtype=view_viewpoint_tokens.dtype, device=view_viewpoint_tokens.device)
                                condition_flag = torch.zeros(B, H_viewpoint, 1, dtype=view_viewpoint_tokens.dtype, device=view_viewpoint_tokens.device)
                                view_viewpoint_tokens = torch.cat([view_viewpoint_tokens, condition_flag], dim=-1)

                        # Compute loss with fix_mask support
                        # We need to compute per-timestep loss to apply fix_mask
                        batch_size = flow_tokens.shape[0]
                        
                        action_loss, flow_loss, loss = \
                            model(cfg, batch_size, actions, flow_tokens, proprio_tokens, 
                            flow_mask=condition_flow_mask, proprio_mask=proprio_mask, point_flows=point_flows)
                        
                        if use_condition_eval:
                            view_action_loss, view_flow_loss, view_loss = \
                                view_model(cfg, batch_size, view_actions, view_flow_tokens, view_proprio_tokens, 
                                flow_mask=view_flow_mask, proprio_mask=view_proprio_mask, point_flows=None, viewpoint_tokens=view_viewpoint_tokens, viewpoint_mask=view_viewpoint_mask)
                        else:
                            view_action_loss, view_flow_loss, view_loss = \
                                view_model(cfg, batch_size, view_actions_for_uncondition, view_flow_tokens, view_proprio_tokens, 
                                flow_mask=view_flow_mask, proprio_mask=view_proprio_mask, point_flows=None, viewpoint_tokens=view_viewpoint_tokens, viewpoint_mask=view_viewpoint_mask)


                        total_loss = loss + view_action_loss
                        
                        # Compute flow_model loss if use_flow_model is True
                        if cfg.training.use_flow_model:
                            flow_actions = point_flows  # [B, T, N*D]
                            
                            future_action_tokens_raw = eval_batch["action"].float().to(accelerator.device)  # [B, T, action_dim]
                            future_action_tokens = future_action_tokens_raw
                        
                            
                            # Apply masking to future_action_tokens with probability masking_ratio (similar to proprio_mask)
                            if cfg.training.use_future_action_masking:
                                # Apply masking with probability masking_ratio
                                if torch.rand(1).item() < cfg.training.masking_ratio:
                                    B_future, T_future, _ = future_action_tokens.shape
                                    future_action_mask = torch.ones(B_future, T_future, dtype=torch.bool, device=accelerator.device)
                                    
                                    # For each batch, randomly mask 0 to T_future-1 timesteps
                                    for b in range(B_future):
                                        # Randomly choose k from 0 to T_future-1 (inclusive) - number of timesteps to mask
                                        k = torch.randint(0, T_future, (1,)).item()
                                        
                                        if k > 0:
                                            # Randomly select k timesteps to mask (set to False)
                                            timesteps_to_mask = torch.randperm(T_future, device=accelerator.device)[:k]
                                            future_action_mask[b, timesteps_to_mask] = False
                                    assert torch.any(future_action_mask, dim=1).all(), f"at least one should be true along T axis, future_action_mask : {future_action_mask}"
                                else:
                                    future_action_mask = None
                            else:
                                future_action_mask = None
                            
                            # flow_model's action_loss corresponds to flow prediction loss (since action_dim = flow_output_dim)
                            flow_model_action_loss, flow_model_flow_loss, flow_model_loss = \
                                flow_model(cfg, batch_size, flow_actions, condition_flow_tokens, proprio_tokens,
                                flow_mask=condition_flow_mask, proprio_mask=proprio_mask, point_flows=None, viewpoint_tokens=viewpoint_tokens, viewpoint_mask=viewpoint_mask,
                                future_action_tokens=future_action_tokens, future_action_mask=future_action_mask)
                            total_loss = total_loss + flow_model_action_loss  # Use flow_model_action_loss
                            eval_epoch_flow_model_loss.append(flow_model_action_loss.item())  # Use flow_model_action_loss
                        
                        eval_epoch_loss.append(total_loss.item())
                        eval_epoch_action_loss.append(action_loss.item())
                        eval_epoch_view_action_loss.append(view_action_loss.item())
                        eval_epoch_flow_loss.append(flow_loss.item())
                
                # Store eval metrics (will be logged together with training metrics)
                eval_metrics = {
                    "eval epoch loss": np.mean(eval_epoch_loss),
                    "eval epoch action loss": np.mean(eval_epoch_action_loss),
                    "eval epoch view action loss": np.mean(eval_epoch_view_action_loss),
                    "eval epoch flow loss": np.mean(eval_epoch_flow_loss),
                }
                if cfg.training.use_flow_model:
                    eval_metrics["eval epoch flow model loss"] = np.mean(eval_epoch_flow_model_loss)
                
                model.train()
                view_model.train()
                if cfg.training.use_flow_model:
                    flow_model.train()
        
        # Log training and eval metrics together
        if accelerator.is_local_main_process and not cfg.debug:
            log_dict = {
                "epoch loss": np.mean(epoch_loss),
                "epoch action loss": np.mean(epoch_action_loss),
                "epoch view action loss": np.mean(epoch_view_action_loss),
                "epoch flow loss": np.mean(epoch_flow_loss),
                "epoch time (m)": (time.time() - start)/60,
            }
            if cfg.training.use_flow_model:
                log_dict["epoch flow model loss"] = np.mean(epoch_flow_model_loss)
            # Add eval metrics if available
            if eval_metrics is not None:
                log_dict.update(eval_metrics)
            wandb.log(log_dict)


        if epoch % cfg.training.ckpt_frequency == 0 or epoch == cfg.training.epochs - 1 or cfg.debug:
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                ckpt_model = accelerator.unwrap_model(model)
                ckpt_view_model = accelerator.unwrap_model(view_model)
                accelerator.save(
                    ckpt_model.state_dict(),
                    os.path.join(ckpt_save_dir, f"epoch_{epoch}.ckpt"),
                )
                accelerator.save(
                    ckpt_view_model.state_dict(),
                    os.path.join(ckpt_save_dir, f"epoch_{epoch}_view.ckpt"),
                )
                if cfg.training.use_flow_model:
                    ckpt_flow_model = accelerator.unwrap_model(flow_model)
                    accelerator.save(
                        ckpt_flow_model.state_dict(),
                        os.path.join(ckpt_save_dir, f"epoch_{epoch}_flow.ckpt"),
                    )
                accelerator.print(f"Saved checkpoint at epoch {epoch}.")
                if cfg.training.use_ema:
                    accelerator.save(
                        ema_model.state_dict(),
                        os.path.join(ckpt_save_dir, f"ema_epoch_{epoch}.ckpt"),
                    )
                    accelerator.save(
                        ema_view_model.state_dict(),
                        os.path.join(ckpt_save_dir, f"ema_epoch_{epoch}_view.ckpt"),
                    )
                    if cfg.training.use_flow_model:
                        accelerator.save(
                            ema_flow_model.state_dict(),
                            os.path.join(ckpt_save_dir, f"ema_epoch_{epoch}_flow.ckpt"),
                        )
                    accelerator.print(f"Saved ema checkpoint at epoch {epoch}.")
                # also save the state
                accelerator.save_state(
                    output_dir=os.path.join(state_save_dir, f"epoch_{epoch}")
                )
                accelerator.print(f"Saved state checkpoint at epoch {epoch}.")

            
        if (
            accelerator.is_local_main_process
            and epoch % cfg.evaluation.eval_frequency == 0 and not cfg.eval_off
            or cfg.debug
        ):
            print("Enter evaluation loop.")
            torch.cuda.empty_cache()
            env = os.environ.copy()
            if cfg.debug:
                eval_model_path = model_path if resume else output_dir
                eval_epoch = model_ckpt if resume else epoch
            else:
                eval_model_path = output_dir
                eval_epoch = epoch

            processes = []
            process = subprocess.Popen(
                [
                    "python",
                    "evaluate_active_vision.py",
                    f"model_path={eval_model_path}",
                    f"ckpt={eval_epoch}",
                    f"downsample_ratio={cfg.dataset.downsample_rate}",
                ],
                env=env,
            )
            processes.append(process)
            for process in processes:
                process.wait()
            
            
            print("Evaluation loop finished.")

if __name__ == "__main__":
    train()
