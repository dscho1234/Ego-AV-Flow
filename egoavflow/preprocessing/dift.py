# Diffusion features (DIFT, https://github.com/Tsingularity/dift, MIT license) from Stable Diffusion 2.1.
# Requires diffusers==0.15 (the U-Net forward below follows its UNet2DConditionModel).
import gc
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionModel


class FeatureUNet2DConditionModel(UNet2DConditionModel):
    """U-Net that returns the features of the requested up-sampling blocks instead of the noise."""

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None
        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps).to(dtype=self.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            emb = emb + self.class_embedding(class_labels).to(dtype=self.dtype)

        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):
            if i > np.max(up_ft_indices):
                break
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
            if i in up_ft_indices:
                up_ft[i] = sample.detach()
        return {"up_ft": up_ft}


class OneStepSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(self, img_tensor, t, up_ft_indices, prompt_embeds=None, cross_attention_kwargs=None):
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        return self.unet(
            latents_noisy, t, up_ft_indices, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cross_attention_kwargs
        )


class SDFeaturizer:
    def __init__(self, sd_path, null_prompt=""):
        unet = FeatureUNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
        pipe = OneStepSDPipeline.from_pretrained(sd_path, unet=unet, safety_checker=None)
        pipe.vae.decoder = None
        pipe.scheduler = DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler")
        gc.collect()
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        pipe.enable_xformers_memory_efficient_attention()
        self.null_prompt = null_prompt
        self.null_prompt_embeds = pipe._encode_prompt(
            prompt=null_prompt, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False
        )
        self.pipe = pipe

    @torch.no_grad()
    def forward(self, img_tensor, prompt="", t=261, up_ft_index=1, ensemble_size=8):
        """Features [1, C, h, w] of a [C, H, W] image in [-1, 1], averaged over `ensemble_size` noise samples."""
        img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).cuda()
        if prompt == self.null_prompt:
            prompt_embeds = self.null_prompt_embeds
        else:
            prompt_embeds = self.pipe._encode_prompt(
                prompt=prompt, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False
            )
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1)
        unet_ft = self.pipe(img_tensor=img_tensor, t=t, up_ft_indices=[up_ft_index], prompt_embeds=prompt_embeds)
        return unet_ft["up_ft"][up_ft_index].mean(0, keepdim=True)


@torch.inference_mode()
def get_dift_features(img_np, featurizer, prompt, ensemble_size, img_size):
    """Feature map [C, img_size, img_size] of an RGB uint8 image and the resized image."""
    img_resized = cv2.resize(img_np, (img_size, img_size))
    img_tensor = (torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0 - 0.5) * 2
    ft = featurizer.forward(img_tensor, prompt=prompt, ensemble_size=ensemble_size).squeeze(0)
    ft_up = F.interpolate(ft.unsqueeze(0), size=(img_size, img_size), mode="bilinear")[0]
    return ft_up, img_resized


def get_corresponding_points(source_ft, source_points, target_ft, img_size):
    """Nearest neighbor (cosine similarity) in `target_ft` of the source feature at every source point."""
    trg_vec = F.normalize(target_ft.view(target_ft.size(0), -1).T, dim=1)
    corresponding_points = []
    for pt in source_points:
        x, y = int(pt[0]), int(pt[1])
        src_vec = F.normalize(source_ft[:, y, x].unsqueeze(0), dim=1)
        cos_sim = torch.matmul(src_vec, trg_vec.T).view(img_size, img_size).cpu().numpy()
        max_idx = np.unravel_index(np.argmax(cos_sim), cos_sim.shape)
        corresponding_points.append([max_idx[1], max_idx[0]])
    return np.array(corresponding_points)
