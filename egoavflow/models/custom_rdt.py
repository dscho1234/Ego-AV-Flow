# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from collections import OrderedDict
import re

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from egoavflow.models.rdt.blocks import (FinalLayer, RDTBlock, TimestepEmbedder,
                               get_1d_sincos_pos_embed_from_grid,
                               get_multimodal_cond_pos_embed)


#################################################################################
#                          CustomRDTRunner Class                                #
#################################################################################
class CustomRDTRunner(nn.Module):
    """
    Custom RDT Runner that uses flow and proprioception conditions.
    Combines model, adaptors, and training/inference functions in one class.
    """
    def __init__(
        self,
        *,
        action_dim,
        pred_horizon,
        hidden_size,
        depth,
        num_heads,
        flow_token_dim,
        proprio_token_dim,
        max_flow_cond_len,
        max_proprio_cond_len,
        flow_adaptor_type='mlp2x_gelu',
        proprio_adaptor_type='mlp2x_gelu',
        action_adaptor_type='mlp3x_gelu',
        noise_scheduler_config=None,
        flow_pos_embed_config=None,
        proprio_pos_embed_config=None,
        dtype=torch.bfloat16,
        predict_flow=False,
        predict_separate_flow=False,
        flow_output_dim=None,
        use_dift_point_tracking=True,
        weight_pos=1.0,
        weight_rot=1.0,
        weight_gripper=1.0,
        # Viewpoint-related parameters (optional, for view_model)
        viewpoint_token_dim=None,
        max_viewpoint_cond_len=None,
        viewpoint_adaptor_type='mlp2x_gelu',
        viewpoint_pos_embed_config=None,
        # Future action-related parameters (optional, for flow_model)
        future_action_token_dim=None,
        max_future_action_cond_len=None,
        future_action_adaptor_type='mlp3x_gelu',
        future_action_pos_embed_config=None,
        cond_types=None,  # List of condition types, e.g., ['flow', 'proprio'] or ['flow', 'proprio', 'viewpoint', 'future_action']
        
    ):
        super(CustomRDTRunner, self).__init__()
        
        self.cond_types = cond_types
        print('cond types : ', cond_types)
        
        # Set max_viewpoint_cond_len if not provided but viewpoint is in cond_types
        if max_viewpoint_cond_len is None and 'viewpoint' in cond_types:
            max_viewpoint_cond_len = max_proprio_cond_len  # Default to same as proprio
        
        # Set max_future_action_cond_len if not provided but future_action is in cond_types
        if max_future_action_cond_len is None and 'future_action' in cond_types:
            max_future_action_cond_len = pred_horizon  # Default to same as pred_horizon
        
        # Create diffusion model
        self.model = CustomRDT(
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            max_flow_cond_len=max_flow_cond_len,
            max_proprio_cond_len=max_proprio_cond_len,
            max_viewpoint_cond_len=max_viewpoint_cond_len if 'viewpoint' in cond_types else 1,
            max_future_action_cond_len=max_future_action_cond_len if 'future_action' in cond_types else 1,
            flow_pos_embed_config=flow_pos_embed_config,
            proprio_pos_embed_config=proprio_pos_embed_config,
            viewpoint_pos_embed_config=viewpoint_pos_embed_config,
            future_action_pos_embed_config=future_action_pos_embed_config,
            dtype=dtype,
            predict_flow=predict_flow,
            predict_separate_flow=predict_separate_flow,
            flow_output_dim=flow_output_dim,
            use_dift_point_tracking=use_dift_point_tracking,
            cond_types=cond_types
        )
        
        self.predict_flow = predict_flow
        self.predict_separate_flow = predict_separate_flow
        self.weight_pos = weight_pos
        self.weight_rot = weight_rot
        self.weight_gripper = weight_gripper
        
        # Create adaptors for conditional inputs
        if 'flow' in cond_types:
            self.flow_adaptor = self.build_condition_adapter(
                flow_adaptor_type,
                in_features=flow_token_dim,
                out_features=hidden_size
            )
        else:
            self.flow_adaptor = None
        
        if 'proprio' in cond_types:
            self.proprio_adaptor = self.build_condition_adapter(
                proprio_adaptor_type,
                in_features=proprio_token_dim,
                out_features=hidden_size
            )
        else:
            self.proprio_adaptor = None
        
        if 'viewpoint' in cond_types:
            if viewpoint_token_dim is None:
                raise ValueError("viewpoint_token_dim must be provided when 'viewpoint' is in cond_types")
            self.viewpoint_adaptor = self.build_condition_adapter(
                viewpoint_adaptor_type,
                in_features=viewpoint_token_dim,
                out_features=hidden_size
            )
        else:
            self.viewpoint_adaptor = None
        
        if 'future_action' in cond_types:
            if future_action_token_dim is None:
                raise ValueError("future_action_token_dim must be provided when 'future_action' is in cond_types")
            self.future_action_adaptor = self.build_condition_adapter(
                future_action_adaptor_type,
                in_features=future_action_token_dim,
                out_features=hidden_size
            )
        else:
            self.future_action_adaptor = None
        
        # If predict_flow is True (but not predict_separate_flow), action_adaptor should accept concat(action, flow)
        # If predict_separate_flow is True, action_adaptor only accepts action (flow is reconstructed from intermediate rep)
        action_adaptor_input_dim = action_dim + flow_output_dim if predict_flow and not predict_separate_flow and flow_output_dim is not None else action_dim
        self.action_adaptor = self.build_condition_adapter(
            action_adaptor_type,
            in_features=action_adaptor_input_dim,  # action_dim or (action_dim + flow_output_dim) -> hidden_size
            out_features=hidden_size
        )
        
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
            clip_sample=noise_scheduler_config['clip_sample'],
        )

        self.noise_scheduler_sample = DDIMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
            clip_sample=noise_scheduler_config['clip_sample'],
        )
        
        self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']
        self.num_inference_steps = noise_scheduler_config['num_inference_steps']
        self.prediction_type = noise_scheduler_config['prediction_type']
        
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.flow_output_dim = flow_output_dim if (predict_flow or predict_separate_flow) else None
        
        param_list = [p.numel() for p in self.model.parameters()]
        if self.flow_adaptor is not None:
            param_list.extend([p.numel() for p in self.flow_adaptor.parameters()])
        if self.proprio_adaptor is not None:
            param_list.extend([p.numel() for p in self.proprio_adaptor.parameters()])
        if self.viewpoint_adaptor is not None:
            param_list.extend([p.numel() for p in self.viewpoint_adaptor.parameters()])
        if self.future_action_adaptor is not None:
            param_list.extend([p.numel() for p in self.future_action_adaptor.parameters()])
        param_list.extend([p.numel() for p in self.action_adaptor.parameters()])
        print("Diffusion params: %e" % sum(param_list))
    
    def build_condition_adapter(self, projector_type, in_features, out_features):
        """
        Build a condition adapter (projector) for transforming input features.
        
        Args:
            projector_type: Type of projector. Can be:
                - 'linear': Single linear layer
                - 'mlpNx_gelu': MLP with N layers and GELU activation
            in_features: Input feature dimension
            out_features: Output feature dimension (hidden_size)
        
        Returns:
            nn.Module: The adapter module
        """
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    def adapt_conditions(self, flow_tokens=None, proprio_tokens=None, viewpoint_tokens=None, future_action_tokens=None, action_tokens=None):
        """
        Adapt condition tokens to hidden_size dimension using adaptors.
        
        Args:
            flow_tokens: (batch_size, num_points, flow_token_dim) or None - raw flow tokens (H*D after reshape)
            proprio_tokens: (batch_size, history_len, proprio_token_dim) or None - raw proprioception tokens
            viewpoint_tokens: (batch_size, history_len, viewpoint_token_dim) or None - raw viewpoint tokens
            future_action_tokens: (batch_size, horizon, future_action_token_dim) or None - raw future action tokens
            action_tokens: (batch_size, horizon, action_token_dim) or None - raw action tokens
        
        Returns:
            tuple: (adapted_flow, adapted_proprio, adapted_viewpoint, adapted_future_action, adapted_action) all with shape (..., hidden_size)
            None values are returned for conditions not in cond_types or if tokens are None
        """
        adapted_flow = self.flow_adaptor(flow_tokens) if (self.flow_adaptor is not None and flow_tokens is not None) else None
        adapted_proprio = self.proprio_adaptor(proprio_tokens) if (self.proprio_adaptor is not None and proprio_tokens is not None) else None
        adapted_viewpoint = self.viewpoint_adaptor(viewpoint_tokens) if (self.viewpoint_adaptor is not None and viewpoint_tokens is not None) else None
        adapted_future_action = self.future_action_adaptor(future_action_tokens) if (self.future_action_adaptor is not None and future_action_tokens is not None) else None
        adapted_action = self.action_adaptor(action_tokens) if action_tokens is not None else None
        
        return adapted_flow, adapted_proprio, adapted_viewpoint, adapted_future_action, adapted_action
    
    def conditional_sample(self, flow_cond=None, proprio_cond=None, viewpoint_cond=None, future_action_cond=None,
                          fix_mask=None, prior=None, flow_mask=None, proprio_mask=None, viewpoint_mask=None, future_action_mask=None):
        """
        Perform conditional sampling using the diffusion model.
        
        Args:
            flow_cond: (batch_size, num_points, hidden_size) or None - flow condition
            proprio_cond: (batch_size, history_len, hidden_size) or None - proprioception condition
            viewpoint_cond: (batch_size, history_len, hidden_size) or None - viewpoint condition
            fix_mask: (batch_size, horizon, action_dim) or (batch_size, horizon, action_dim + flow_output_dim) - mask for fixed dimensions
            prior: (batch_size, horizon, action_dim) or (batch_size, horizon, action_dim + flow_output_dim) - prior values for fixed dimensions
            flow_mask: (batch_size, num_points) or None - flow condition mask (True for valid)
            proprio_mask: (batch_size, history_len) or None - proprioception condition mask (True for valid)
            viewpoint_mask: (batch_size, history_len) or None - viewpoint condition mask (True for valid)
        
        Returns:
            If predict_flow=False and predict_separate_flow=False: (batch_size, horizon, action_dim) - sampled action sequence
            If predict_flow=True: (batch_size, horizon, action_dim + flow_output_dim) - sampled concat(action, flow)
            If predict_separate_flow=True: tuple of ((batch_size, horizon, action_dim), (batch_size, horizon, flow_output_dim))
        """
        # Get device and dtype from any available condition
        if flow_cond is not None:
            device = flow_cond.device
            dtype = flow_cond.dtype
            batch_size = flow_cond.shape[0]
        elif proprio_cond is not None:
            device = proprio_cond.device
            dtype = proprio_cond.dtype
            batch_size = proprio_cond.shape[0]
        elif viewpoint_cond is not None:
            device = viewpoint_cond.device
            dtype = viewpoint_cond.dtype
            batch_size = viewpoint_cond.shape[0]
        else:
            raise ValueError("At least one condition (flow_cond, proprio_cond, or viewpoint_cond) must be provided")
        
        if self.predict_separate_flow:
            # Action diffusion only, flow is reconstructed from intermediate representation
            # Initialize noisy actions
            noisy_action = torch.randn(
                size=(batch_size, self.pred_horizon, self.action_dim),
                dtype=dtype, device=device
            )
            
            # Apply fix_mask and prior to initial noise if provided
            if fix_mask is not None and prior is not None:
                # Split fix_mask and prior for action
                action_fix_mask = fix_mask[:, :, :self.action_dim] if fix_mask.shape[-1] > self.action_dim else fix_mask
                action_prior = prior[:, :, :self.action_dim] if prior.shape[-1] > self.action_dim else prior
                noisy_action = (1. - action_fix_mask) * noisy_action + action_fix_mask * action_prior
            
            # Set step values
            self.noise_scheduler_sample.set_timesteps(self.num_inference_steps)
            
            # Store intermediate representation from the last timestep for flow reconstruction
            intermediate_rep = None
            all_attentions = None
            
            for idx, t in enumerate(self.noise_scheduler_sample.timesteps):
                # Adapt noisy_action to hidden_size for model input
                action_input = self.action_adaptor(noisy_action)
                
                # Only collect attention from the last timestep
                is_last = (idx == len(self.noise_scheduler_sample.timesteps) - 1)
                
                # Predict action and get intermediate representation
                if is_last:
                    intermediate_rep, action_output, _, all_attentions = self.model(
                        x=action_input,
                        t=t.unsqueeze(-1).to(device),
                        flow_c=flow_cond,
                        proprio_c=proprio_cond,
                        viewpoint_c=viewpoint_cond,
                        future_action_c=future_action_cond,
                        flow_mask=flow_mask,
                        proprio_mask=proprio_mask,
                        viewpoint_mask=viewpoint_mask,
                        future_action_mask=future_action_mask,
                        return_intermediate=True
                    )
                else:
                    intermediate_rep, action_output, _, _ = self.model(
                        x=action_input,
                        t=t.unsqueeze(-1).to(device),
                        flow_c=flow_cond,
                        proprio_c=proprio_cond,
                        viewpoint_c=viewpoint_cond,
                        future_action_c=future_action_cond,
                        flow_mask=flow_mask,
                        proprio_mask=proprio_mask,
                        viewpoint_mask=viewpoint_mask,
                        future_action_mask=future_action_mask,
                        return_intermediate=True
                    )
                
                # Compute previous actions: x_t -> x_t-1
                noisy_action = self.noise_scheduler_sample.step(
                    action_output, t, noisy_action
                ).prev_sample
                noisy_action = noisy_action.to(dtype)
                
                # Apply fix_mask and prior after each denoising step
                if fix_mask is not None and prior is not None:
                    action_fix_mask = fix_mask[:, :, :self.action_dim] if fix_mask.shape[-1] > self.action_dim else fix_mask
                    action_prior = prior[:, :, :self.action_dim] if prior.shape[-1] > self.action_dim else prior
                    noisy_action = (1. - action_fix_mask) * noisy_action + action_fix_mask * action_prior
            
            # Reconstruct flow from the final intermediate representation
            # intermediate_rep: (B, horizon+1, hidden_size)
            flow_output = self.model.final_layer_flow(intermediate_rep)  # (B, horizon+1, flow_output_dim)
            flow_pred = flow_output[:, -self.pred_horizon:]  # (B, horizon, flow_output_dim)
            
            return noisy_action, flow_pred, all_attentions
        
        else:
            # Original logic for predict_flow or no flow prediction
            # Initialize noisy actions (or concat(action, flow) if predict_flow is True)
            noisy_input_dim = self.action_dim + self.flow_output_dim if self.predict_flow else self.action_dim
            noisy_action = torch.randn(
                size=(batch_size, self.pred_horizon, noisy_input_dim),
                dtype=dtype, device=device
            )
            
            # Apply fix_mask and prior to initial noise if provided
            if fix_mask is not None and prior is not None:
                noisy_action = (1. - fix_mask) * noisy_action + fix_mask * prior
            
            # Set step values
            self.noise_scheduler_sample.set_timesteps(self.num_inference_steps)
            
            all_attentions = None
            for idx, t in enumerate(self.noise_scheduler_sample.timesteps):
                # Adapt noisy_action to hidden_size for model input
                # noisy_action: (B, horizon, action_dim) -> (B, horizon, hidden_size)
                action_input = self.action_adaptor(noisy_action)
                
                # Only collect attention from the last timestep
                is_last = (idx == len(self.noise_scheduler_sample.timesteps) - 1)
                
                # Predict the model output
                if is_last:
                    intermediate_rep, action_output, flow_output, all_attentions = self.model(
                        x=action_input,  # (B, horizon, hidden_size)
                        t=t.unsqueeze(-1).to(device),  # (num_timesteps,)
                        flow_c=flow_cond,
                        proprio_c=proprio_cond,
                        viewpoint_c=viewpoint_cond,
                        future_action_c=future_action_cond,
                        flow_mask=flow_mask,
                        proprio_mask=proprio_mask,
                        viewpoint_mask=viewpoint_mask,
                        future_action_mask=future_action_mask,
                        return_intermediate=True
                    )
                    # Model output is already concat(action, flow) if predict_flow is True
                    if self.predict_flow:
                        model_output_concat = torch.cat([action_output, flow_output], dim=-1)
                    else:
                        model_output_concat = action_output
                else:
                    model_output = self.model(
                        x=action_input,  # (B, horizon, hidden_size)
                        t=t.unsqueeze(-1).to(device),  # (num_timesteps,)
                        flow_c=flow_cond,
                        proprio_c=proprio_cond,
                        viewpoint_c=viewpoint_cond,
                        future_action_c=future_action_cond,
                        flow_mask=flow_mask,
                        proprio_mask=proprio_mask,
                        viewpoint_mask=viewpoint_mask,
                        future_action_mask=future_action_mask
                    )
                    # Model output is already concat(action, flow) if predict_flow is True
                    model_output_concat = model_output if not self.predict_flow else torch.cat(model_output, dim=-1)
                
                # Compute previous actions: x_t -> x_t-1
                noisy_action = self.noise_scheduler_sample.step(
                    model_output_concat, t, noisy_action
                ).prev_sample
                noisy_action = noisy_action.to(dtype)
                
                # Apply fix_mask and prior after each denoising step
                if fix_mask is not None and prior is not None:
                    noisy_action = (1. - fix_mask) * noisy_action + fix_mask * prior
            
            # If predict_flow is True, split the output
            if self.predict_flow:
                action_pred = noisy_action[:, :, :self.action_dim]
                flow_pred = noisy_action[:, :, self.action_dim:]
                return action_pred, flow_pred, all_attentions
            else:
                return noisy_action, None, all_attentions
    
    
    def predict_action(self, flow_tokens=None, proprio_tokens=None, viewpoint_tokens=None, future_action_tokens=None,
                      fix_mask=None, prior=None, flow_mask=None, proprio_mask=None, viewpoint_mask=None, future_action_mask=None, return_attn=False):
        """
        Predict action sequence given conditions.
        
        Args:
            flow_tokens: (batch_size, num_points, flow_token_dim) or None - raw flow tokens (H*D after reshape)
            proprio_tokens: (batch_size, history_len, proprio_token_dim) or None - raw proprioception tokens
            viewpoint_tokens: (batch_size, history_len, viewpoint_token_dim) or None - raw viewpoint tokens
            fix_mask: (batch_size, horizon, action_dim) or (batch_size, horizon, action_dim + flow_output_dim) - mask for fixed dimensions
            prior: (batch_size, horizon, action_dim) or (batch_size, horizon, action_dim + flow_output_dim) - prior values for fixed dimensions
            flow_mask: (batch_size, num_points) or None - flow condition mask (True for valid)
            proprio_mask: (batch_size, history_len) or None - proprioception condition mask (True for valid)
            viewpoint_mask: (batch_size, history_len) or None - viewpoint condition mask (True for valid)
            return_attn: bool, if True, return attention weights from the last timestep
        
        Returns:
            If return_attn=False:
                If predict_flow=False: (batch_size, horizon, action_dim) - predicted action sequence
                If predict_flow=True: tuple of ((batch_size, horizon, action_dim), (batch_size, horizon, flow_output_dim))
            If return_attn=True:
                If predict_flow=False: tuple of ((batch_size, horizon, action_dim), attention_list)
                If predict_flow=True: tuple of ((batch_size, horizon, action_dim), (batch_size, horizon, flow_output_dim), attention_list)
        """
        # Adapt conditions to hidden_size
        flow_cond, proprio_cond, viewpoint_cond, future_action_cond, _ = self.adapt_conditions(
            flow_tokens=flow_tokens,
            proprio_tokens=proprio_tokens,
            viewpoint_tokens=viewpoint_tokens,
            future_action_tokens=future_action_tokens,
            action_tokens=None  # Not needed for inference, will use noisy_action
        )
        
        # Run sampling with flow_mask and proprio_mask
        # conditional_sample already handles predict_flow and returns appropriately
        result = self.conditional_sample(
            flow_cond=flow_cond,
            proprio_cond=proprio_cond,
            viewpoint_cond=viewpoint_cond,
            future_action_cond=future_action_cond,
            fix_mask=fix_mask,
            prior=prior,
            flow_mask=flow_mask,
            proprio_mask=proprio_mask,
            viewpoint_mask=viewpoint_mask,
            future_action_mask=future_action_mask
        )
        
        if return_attn:
            return result
        else:
            # Remove attention from result if not requested
            if self.predict_separate_flow:
                return result[0], result[1]  # action, flow
            elif self.predict_flow:
                return result[0], result[1]  # action, flow
            else:
                return result[0]  # action
    
    def forward(self, cfg, batch_size, actions, flow_tokens, proprio_tokens, flow_mask=None, proprio_mask=None, point_flows=None, viewpoint_tokens=None, viewpoint_mask=None, future_action_tokens=None, future_action_mask=None):
        

        if self.predict_separate_flow:
            assert not self.predict_flow
            # Action diffusion only, flow is reconstructed from intermediate representation
            if cfg.use_fix_mask:
                action_fix_mask = torch.zeros_like(actions, device=actions.device)
                if not cfg.use_relative:
                    action_fix_mask[:, 0, :] = 1.0
                    
            
            # Sample noise and timesteps for action
            action_noise = torch.randn(actions.shape, dtype=actions.dtype, device=actions.device)
            action_timesteps = torch.randint(
                0, self.num_train_timesteps,
                (batch_size,), device=actions.device
            ).long()
            noisy_action = self.noise_scheduler.add_noise(actions, action_noise, action_timesteps)
            
            # Apply fix_mask after add_noise
            if cfg.use_fix_mask:
                noisy_action = (1. - action_fix_mask) * noisy_action + action_fix_mask * actions
            
            # Adapt conditions
            flow_cond, proprio_cond, viewpoint_cond, future_action_cond, _ = self.adapt_conditions(
                flow_tokens=flow_tokens,
                proprio_tokens=proprio_tokens,
                viewpoint_tokens=viewpoint_tokens,
                future_action_tokens=future_action_tokens,
                action_tokens=None
            )
            
            # Adapt noisy_action
            action_input = self.action_adaptor(noisy_action)
            
            # Predict action and get intermediate representation for flow reconstruction
            _, pred_action, pred_flow, _ = self.model(
                x=action_input,
                t=action_timesteps,
                flow_c=flow_cond,
                proprio_c=proprio_cond,
                viewpoint_c=viewpoint_cond,
                future_action_c=future_action_cond,
                flow_mask=flow_mask,
                proprio_mask=proprio_mask,
                viewpoint_mask=viewpoint_mask,
                future_action_mask=future_action_mask,
                return_intermediate=True
            )
            
            # Determine target for action
            if self.prediction_type == 'epsilon':
                action_target = action_noise
            elif self.prediction_type == 'sample':
                action_target = actions
            else:
                raise ValueError(f"Unsupported prediction type {self.prediction_type}")
            
            # Compute action loss (diffusion loss)
            action_mse_loss = (pred_action - action_target) ** 2
            
            # Apply weighted loss for use_relative=true and action_dim=7
            if cfg.use_relative:
                weight_pos = self.weight_pos
                weight_rot = self.weight_rot
                weight_gripper = self.weight_gripper
                action_weights = torch.ones_like(action_mse_loss)
                action_weights[:, :, :3] = weight_pos  # position
                action_weights[:, :, 3:6] = weight_rot  # rotation
                if actions.shape[-1] == 7:
                    action_weights[:, :, 6:7] = weight_gripper  # gripper
                action_mse_loss = action_mse_loss * action_weights
            
            # Compute flow loss (reconstruction loss)
            flow_mse_loss = (pred_flow - point_flows) ** 2
            
            if cfg.use_fix_mask:
                action_loss = (action_mse_loss * (1 - action_fix_mask)).mean()
                flow_loss = flow_mse_loss.mean()  # Flow doesn't use fix_mask since it is not diffusion outputs
            else:
                action_loss = action_mse_loss.mean()
                flow_loss = flow_mse_loss.mean()
            
            loss = action_loss + flow_loss
        
        elif self.predict_flow:
            assert not self.predict_separate_flow
            # Original predict_flow logic: concat actions and point_flows
            target_concat = torch.cat([actions, point_flows], dim=-1)  # [B, T, action_dim + flow_output_dim]

            if cfg.use_fix_mask:
                fix_mask = torch.zeros_like(target_concat, device=actions.device)
                if not cfg.use_relative:
                    fix_mask[:, 0, :] = 1.0
            
            # Sample noise and timesteps for action model
            noise = torch.randn(target_concat.shape, dtype=target_concat.dtype, device=actions.device)
            timesteps = torch.randint(
                0, self.num_train_timesteps,
                (batch_size,), device=actions.device
            ).long()
            noisy_action = self.noise_scheduler.add_noise(target_concat, noise, timesteps)
            
            # Apply fix_mask after add_noise
            if cfg.use_fix_mask:
                noisy_action = (1. - fix_mask) * noisy_action + fix_mask * target_concat
            
            # Adapt conditions
            flow_cond, proprio_cond, viewpoint_cond, future_action_cond, _ = self.adapt_conditions(
                flow_tokens=flow_tokens,
                proprio_tokens=proprio_tokens,
                viewpoint_tokens=viewpoint_tokens,
                future_action_tokens=future_action_tokens,
                action_tokens=None
            )
            
            # Adapt noisy_action
            action_input = self.action_adaptor(noisy_action)
            
            # Predict
            model_output = self.model(
                x=action_input,
                t=timesteps,
                flow_c=flow_cond,
                proprio_c=proprio_cond,
                viewpoint_c=viewpoint_cond,
                future_action_c=future_action_cond,
                flow_mask=flow_mask,
                proprio_mask=proprio_mask,
                viewpoint_mask=viewpoint_mask,
                future_action_mask=future_action_mask
            )
            
            # Handle output based on predict_flow flag
            pred_action, pred_flow = model_output
            pred_concat = torch.cat([pred_action, pred_flow], dim=-1)  # [B, T, action_dim + flow_output_dim]
            
            # Determine target
            if self.prediction_type == 'epsilon':
                target = noise
            elif self.prediction_type == 'sample':
                target = target_concat
            else:
                raise ValueError(f"Unsupported prediction type {self.prediction_type}")
            
            # Compute loss on concat output
            mse_loss = (pred_concat - target) ** 2  # [B, T, action_dim + flow_output_dim]
            
            # Apply weighted loss for use_relative=true and action_dim=7
            if cfg.use_relative:
                weight_pos = self.weight_pos
                weight_rot = self.weight_rot
                weight_gripper = self.weight_gripper
                action_weights = torch.ones_like(mse_loss[:, :, :actions.shape[-1]])
                action_weights[:, :, :3] = weight_pos  # position
                action_weights[:, :, 3:6] = weight_rot  # rotation
                if actions.shape[-1] == 7:
                    action_weights[:, :, 6:7] = weight_gripper  # gripper
                
                mse_loss[:, :, :actions.shape[-1]] = mse_loss[:, :, :actions.shape[-1]] * action_weights
            
            if cfg.use_fix_mask:
                loss = (mse_loss * (1 - fix_mask))
            else:
                loss = mse_loss
            
            action_loss = loss[:, :, :actions.shape[-1]].mean()
            flow_loss = loss[:, :, actions.shape[-1]:].mean()
            loss = loss.mean()
        
        else:
            # No flow prediction: only action
            target_concat = actions

            if cfg.use_fix_mask:
                fix_mask = torch.zeros_like(target_concat, device=actions.device)
                if not cfg.use_relative:
                    if cfg.training.use_ptp:
                        fix_mask[:, :cfg.dataset.obs_horizon, :] = 1.0 # [t-2, t-1, t] for h =3
                    else:
                        fix_mask[:, 0, :] = 1.0
            
            # Sample noise and timesteps for action model
            noise = torch.randn(target_concat.shape, dtype=target_concat.dtype, device=actions.device)
            timesteps = torch.randint(
                0, self.num_train_timesteps,
                (batch_size,), device=actions.device
            ).long()
            noisy_action = self.noise_scheduler.add_noise(target_concat, noise, timesteps)
            
            # Apply fix_mask after add_noise
            if cfg.use_fix_mask:
                noisy_action = (1. - fix_mask) * noisy_action + fix_mask * target_concat
            
            # Adapt conditions
            flow_cond, proprio_cond, viewpoint_cond, future_action_cond, _ = self.adapt_conditions(
                flow_tokens=flow_tokens,
                proprio_tokens=proprio_tokens,
                viewpoint_tokens=viewpoint_tokens,
                future_action_tokens=future_action_tokens,
                action_tokens=None
            )
            
            # Adapt noisy_action
            action_input = self.action_adaptor(noisy_action)
            
            # Predict
            model_output = self.model(
                x=action_input,
                t=timesteps,
                flow_c=flow_cond,
                proprio_c=proprio_cond,
                viewpoint_c=viewpoint_cond,
                future_action_c=future_action_cond,
                flow_mask=flow_mask,
                proprio_mask=proprio_mask,
                viewpoint_mask=viewpoint_mask,
                future_action_mask=future_action_mask
            )
            
            pred_concat = model_output
            
            # Determine target
            if self.prediction_type == 'epsilon':
                target = noise
            elif self.prediction_type == 'sample':
                target = target_concat
            else:
                raise ValueError(f"Unsupported prediction type {self.prediction_type}")
            
            # Compute loss
            mse_loss = (pred_concat - target) ** 2  # [B, T, action_dim]
            
            # Apply weighted loss for use_relative=true and action_dim=7
            if cfg.use_relative:
                weight_pos = self.weight_pos
                weight_rot = self.weight_rot
                weight_gripper = self.weight_gripper
                action_weights = torch.ones_like(mse_loss)
                action_weights[:, :, :3] = weight_pos  # position
                action_weights[:, :, 3:6] = weight_rot  # rotation
                if actions.shape[-1] == 7:
                    action_weights[:, :, 6:7] = weight_gripper  # gripper
                mse_loss = mse_loss * action_weights
            
            if cfg.use_fix_mask:
                loss = (mse_loss * (1 - fix_mask))
            else:
                loss = mse_loss
            
            action_loss = loss.mean()
            flow_loss = torch.tensor(0.0, device=actions.device)  # No flow loss
            loss = loss.mean()
        
        return action_loss, flow_loss, loss


class CustomRDT(nn.Module):
    """
    Custom RDT model that uses flow and proprioception conditions instead of
    language and image conditions. Also removes frequency embedding.
    """
    def __init__(
        self,
        output_dim=128,
        horizon=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        max_flow_cond_len=32,
        max_proprio_cond_len=32,
        max_viewpoint_cond_len=32,
        max_future_action_cond_len=32,
        flow_pos_embed_config=None,
        proprio_pos_embed_config=None,
        viewpoint_pos_embed_config=None,
        future_action_pos_embed_config=None,
        dtype=torch.bfloat16,
        predict_flow=False,
        predict_separate_flow=False,
        flow_output_dim=None,
        use_dift_point_tracking=True,
        cond_types=None  # List of condition types, e.g., ['flow', 'proprio'] or ['flow', 'proprio', 'viewpoint', 'future_action']
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.max_flow_cond_len = max_flow_cond_len
        self.max_proprio_cond_len = max_proprio_cond_len
        self.max_viewpoint_cond_len = max_viewpoint_cond_len
        self.max_future_action_cond_len = max_future_action_cond_len
        self.dtype = dtype
        self.flow_pos_embed_config = flow_pos_embed_config
        self.proprio_pos_embed_config = proprio_pos_embed_config
        self.viewpoint_pos_embed_config = viewpoint_pos_embed_config
        self.future_action_pos_embed_config = future_action_pos_embed_config
        self.predict_flow = predict_flow
        self.predict_separate_flow = predict_separate_flow
        self.action_dim = output_dim  # Store original action_dim for splitting
        self.use_dift_point_tracking = use_dift_point_tracking
        
        # Set default cond_types if not provided
        if cond_types is None:
            cond_types = ['flow', 'proprio']
        self.cond_types = cond_types

        # Only timestep embedder, no frequency embedder
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        
        # Positional embeddings for input sequence [timestep; action]
        # Removed ctrl_freq and state from the sequence
        self.x_pos_embed = nn.Parameter(
            torch.zeros(1, horizon+1, hidden_size))
        
        # Flow condition positional embeddings
        if 'flow' in self.cond_types:
            self.flow_cond_pos_embed = nn.Parameter(
                torch.zeros(1, max_flow_cond_len, hidden_size))
        else:
            self.flow_cond_pos_embed = None
        
        # Proprioception condition positional embeddings
        if 'proprio' in self.cond_types:
            self.proprio_cond_pos_embed = nn.Parameter(
                torch.zeros(1, max_proprio_cond_len, hidden_size))
        else:
            self.proprio_cond_pos_embed = None
        
        # Viewpoint condition positional embeddings
        if 'viewpoint' in self.cond_types:
            self.viewpoint_cond_pos_embed = nn.Parameter(
                torch.zeros(1, max_viewpoint_cond_len, hidden_size))
        else:
            self.viewpoint_cond_pos_embed = None
        
        # Future action condition positional embeddings
        if 'future_action' in self.cond_types:
            self.future_action_cond_pos_embed = nn.Parameter(
                torch.zeros(1, max_future_action_cond_len, hidden_size))
        else:
            self.future_action_cond_pos_embed = None

        self.blocks = nn.ModuleList([
            RDTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        # If predict_flow is True (but not predict_separate_flow), output_dim should be action_dim + flow_output_dim
        # If predict_separate_flow is True, we have separate final layers for action and flow
        # Otherwise, output_dim is just action_dim
        if predict_separate_flow:
            if flow_output_dim is None:
                raise ValueError("flow_output_dim must be provided when predict_separate_flow=True")
            # Use separate final layers for action and flow
            self.final_layer = FinalLayer(hidden_size, output_dim)
            self.final_layer_flow = FinalLayer(hidden_size, flow_output_dim)
            self.flow_output_dim = flow_output_dim
        elif predict_flow:
            if flow_output_dim is None:
                raise ValueError("flow_output_dim must be provided when predict_flow=True")
            # Use a single final layer that outputs concat(action, flow)
            self.final_layer = FinalLayer(hidden_size, output_dim + flow_output_dim)
            self.flow_output_dim = flow_output_dim
        else:
            self.final_layer = FinalLayer(hidden_size, output_dim)
            self.flow_output_dim = None
            
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding
        # Removed ctrl_freq and state from the sequence
        x_pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=OrderedDict([
                ('timestep', 1),
                ('action', self.horizon),
            ])
        )
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))

        # Initialize flow condition positional embeddings
        if 'flow' in self.cond_types:
            if self.flow_pos_embed_config is None:
                flow_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                    self.hidden_size, torch.arange(self.max_flow_cond_len))
            else:
                flow_cond_pos_embed = get_multimodal_cond_pos_embed(
                    embed_dim=self.hidden_size,
                    mm_cond_lens=OrderedDict(self.flow_pos_embed_config),
                    embed_modality=False
                )
            self.flow_cond_pos_embed.data.copy_(
                torch.from_numpy(flow_cond_pos_embed).float().unsqueeze(0))
        
        # Initialize proprioception condition positional embeddings
        if 'proprio' in self.cond_types:
            if self.proprio_pos_embed_config is None:
                proprio_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                    self.hidden_size, torch.arange(self.max_proprio_cond_len))
            else:
                proprio_cond_pos_embed = get_multimodal_cond_pos_embed(
                    embed_dim=self.hidden_size,
                    mm_cond_lens=OrderedDict(self.proprio_pos_embed_config),
                    embed_modality=False
                )
            self.proprio_cond_pos_embed.data.copy_(
                torch.from_numpy(proprio_cond_pos_embed).float().unsqueeze(0))
        
        # Initialize viewpoint condition positional embeddings
        if 'viewpoint' in self.cond_types:
            if self.viewpoint_pos_embed_config is None:
                viewpoint_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                    self.hidden_size, torch.arange(self.max_viewpoint_cond_len))
            else:
                viewpoint_cond_pos_embed = get_multimodal_cond_pos_embed(
                    embed_dim=self.hidden_size,
                    mm_cond_lens=OrderedDict(self.viewpoint_pos_embed_config),
                    embed_modality=False
                )
            self.viewpoint_cond_pos_embed.data.copy_(
                torch.from_numpy(viewpoint_cond_pos_embed).float().unsqueeze(0))
        
        # Initialize future action condition positional embeddings
        if 'future_action' in self.cond_types:
            if self.future_action_pos_embed_config is None:
                future_action_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                    self.hidden_size, torch.arange(self.max_future_action_cond_len))
            else:
                future_action_cond_pos_embed = get_multimodal_cond_pos_embed(
                    embed_dim=self.hidden_size,
                    mm_cond_lens=OrderedDict(self.future_action_pos_embed_config),
                    embed_modality=False
                )
            self.future_action_cond_pos_embed.data.copy_(
                torch.from_numpy(future_action_cond_pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP (removed freq_embedder initialization)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            
        # Initialize the final layer: zero-out the final linear layer
        # Note: This is a common practice in diffusion models to start with small outputs
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        
        # Initialize final_layer_flow if it exists
        if self.predict_separate_flow:
            nn.init.constant_(self.final_layer_flow.ffn_final.fc2.weight, 0)
            nn.init.constant_(self.final_layer_flow.ffn_final.fc2.bias, 0)
        
        # Move all the params to given data type:
        self.to(self.dtype)

    def forward(self, x, t, flow_c=None, proprio_c=None, viewpoint_c=None, future_action_c=None,
                flow_mask=None, proprio_mask=None, viewpoint_mask=None, future_action_mask=None, return_intermediate=False):
        """
        Forward pass of Custom RDT.
        
        Args:
            x: (B, T, D), action token sequence, T = horizon,
                dimension D is assumed to be the same as the hidden size.
            t: (B,) or (1,), diffusion timesteps.
            flow_c: (B, num_points, D) or None, flow condition tokens (point-wise data),
                dimension D is assumed to be the same as the hidden size.
            proprio_c: (B, history_len, D) or None, proprioception condition tokens (time-series data),
                dimension D is assumed to be the same as the hidden size.
            viewpoint_c: (B, history_len, D) or None, viewpoint condition tokens (time-series data),
                dimension D is assumed to be the same as the hidden size.
            future_action_c: (B, horizon, D) or None, future action condition tokens (time-series data),
                dimension D is assumed to be the same as the hidden size.
            flow_mask: (B, num_points) or None, flow condition mask (True for valid).
            proprio_mask: (B, history_len) or None, proprioception condition mask (True for valid).
            viewpoint_mask: (B, history_len) or None, viewpoint condition mask (True for valid).
            future_action_mask: (B, horizon) or None, future action condition mask (True for valid).
            return_intermediate: bool, if True, return intermediate representation before final_layer
        
        Returns:
            If return_intermediate=True: tuple of (intermediate_rep, action_output, flow_output)
                - intermediate_rep: (B, horizon+1, hidden_size) - representation before final layers
                - action_output: (B, horizon, output_dim) or None
                - flow_output: (B, horizon, flow_output_dim) or None
            If predict_flow=False and predict_separate_flow=False: (B, horizon, output_dim), predicted action sequence
            If predict_flow=True: tuple of ((B, horizon, output_dim), (B, horizon, flow_output_dim))
            If predict_separate_flow=True: (B, horizon, output_dim), predicted action sequence
        """
        # Embed timestep (no frequency embedding)
        t = self.t_embedder(t).unsqueeze(1)             # (B, 1, D) or (1, 1, D)
        
        # Append timestep to the input tokens (no frequency, no state)
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        x = torch.cat([t, x], dim=1)                       # (B, horizon+1, D)
        
        # Add multimodal position embeddings
        x = x + self.x_pos_embed
        
        # Add positional embeddings to conditions
        # Note: flow_c is point-wise (num_points dimension), proprio_c and viewpoint_c are time-series (history_len dimension)
        if 'flow' in self.cond_types and flow_c is not None:
            if self.use_dift_point_tracking:
                flow_c = flow_c + self.flow_cond_pos_embed[:, :flow_c.shape[1]]  # Apply along num_points dimension
            # If use_dift_point_tracking=False, flow_c remains unchanged (no positional embedding)
        
        if 'proprio' in self.cond_types and proprio_c is not None:
            proprio_c = proprio_c + self.proprio_cond_pos_embed[:, :proprio_c.shape[1]]  # Apply along history_len dimension
        
        if 'viewpoint' in self.cond_types and viewpoint_c is not None:
            viewpoint_c = viewpoint_c + self.viewpoint_cond_pos_embed[:, :viewpoint_c.shape[1]]  # Apply along history_len dimension
        
        if 'future_action' in self.cond_types and future_action_c is not None:
            future_action_c = future_action_c + self.future_action_cond_pos_embed[:, :future_action_c.shape[1]]  # Apply along horizon dimension

        # Forward pass through blocks with conditions based on cond_types
        # Build conds and masks lists based on cond_types
        conds = []
        masks = []
        for cond_type in self.cond_types:
            if cond_type == 'flow':
                conds.append(flow_c)
                masks.append(flow_mask)
            elif cond_type == 'proprio':
                conds.append(proprio_c)
                masks.append(proprio_mask)
            elif cond_type == 'viewpoint':
                conds.append(viewpoint_c)
                masks.append(viewpoint_mask)
            elif cond_type == 'future_action':
                conds.append(future_action_c)
                masks.append(future_action_mask)
        
        num_conds = len(conds)
        all_attentions = [] if return_intermediate else None
        for i, block in enumerate(self.blocks):
            c, mask = conds[i % num_conds], masks[i % num_conds]
            cond_type = self.cond_types[i % num_conds]
            if return_intermediate:
                x, attn = block(x, c, mask, return_attn=True)
                all_attentions.append({
                    'block_idx': i,
                    'cond_type': cond_type,
                    'attention': attn.detach().cpu().numpy()  # (B, N, L)
                })
            else:
                x = block(x, c, mask)                       # (B, T+1, D)
        
        # If return_intermediate is True, return intermediate representation
        if return_intermediate:
            intermediate_rep = x  # (B, horizon+1, hidden_size)
            if self.predict_separate_flow:
                action_output = self.final_layer(intermediate_rep)  # (B, T+1, output_dim)
                action_output = action_output[:, -self.horizon:]  # (B, horizon, output_dim)
                flow_output = self.final_layer_flow(intermediate_rep)  # (B, T+1, flow_output_dim)
                flow_output = flow_output[:, -self.horizon:]  # (B, horizon, flow_output_dim)
                return intermediate_rep, action_output, flow_output, all_attentions
            elif self.predict_flow:
                output = self.final_layer(intermediate_rep)  # (B, T+1, output_dim + flow_output_dim)
                output = output[:, -self.horizon:]  # (B, horizon, output_dim + flow_output_dim)
                action_output = output[:, :, :self.action_dim]  # (B, horizon, action_dim)
                flow_output = output[:, :, self.action_dim:]  # (B, horizon, flow_output_dim)
                return intermediate_rep, action_output, flow_output, all_attentions
            else:
                action_output = self.final_layer(intermediate_rep)  # (B, T+1, output_dim)
                action_output = action_output[:, -self.horizon:]  # (B, horizon, output_dim)
                return intermediate_rep, action_output, None, all_attentions
        
        # Handle different prediction modes (only for evaluation)
        if self.predict_separate_flow:
            # Only predict action using final_layer (flow will be predicted from intermediate rep separately)
            output = self.final_layer(x)  # (B, T+1, output_dim)
            output = output[:, -self.horizon:]  # (B, horizon, output_dim)
            return output
        elif self.predict_flow:
            # Final layer outputs concat(action, flow)
            output = self.final_layer(x)  # (B, T+1, output_dim + flow_output_dim)
            output = output[:, -self.horizon:]  # (B, horizon, output_dim + flow_output_dim)
            # Split the output into action and flow
            action_output = output[:, :, :self.action_dim]  # (B, horizon, action_dim)
            flow_output = output[:, :, self.action_dim:]  # (B, horizon, flow_output_dim)
            return action_output, flow_output
        else:
            # Final layer outputs just action
            output = self.final_layer(x)  # (B, T+1, output_dim)
            output = output[:, -self.horizon:]  # (B, horizon, output_dim)
            return output

