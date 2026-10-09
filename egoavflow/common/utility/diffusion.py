import os

import hydra
import torch

from egoavflow.common.utility.model import load_config


def load_flow_diffusion_model(
    model_path, ckpt, load_pretrain_weight=True, use_ema=False, load_noise_scheduler=True, load_view_model=False, load_flow_model=False, **kwargs
):
    model_cfg = load_config(model_path)
    if load_flow_model:
        model = hydra.utils.instantiate(model_cfg.flow_model)
        if use_ema:
            print("Loading ema flow model checkpoints!")
            ckpt_path = os.path.join(model_path, "checkpoints", f"ema_epoch_{ckpt}_flow.ckpt")
        else:
            ckpt_path = os.path.join(model_path, "checkpoints", f"epoch_{ckpt}_flow.ckpt")
    elif load_view_model:
        model = hydra.utils.instantiate(model_cfg.view_model)
        if use_ema:
            print("Loading ema model checkpoints!")
            ckpt_path = os.path.join(model_path, "checkpoints", f"ema_epoch_{ckpt}_view.ckpt")
        else:
            ckpt_path = os.path.join(model_path, "checkpoints", f"epoch_{ckpt}_view.ckpt")
    else:
        model = hydra.utils.instantiate(model_cfg.model)
        if use_ema:
            print("Loading ema model checkpoints!")
            ckpt_path = os.path.join(model_path, "checkpoints", f"ema_epoch_{ckpt}.ckpt")
        else:
            ckpt_path = os.path.join(model_path, "checkpoints", f"epoch_{ckpt}.ckpt")
    if load_pretrain_weight:
        model.load_state_dict(torch.load(ckpt_path, weights_only=False))
    model.to("cuda")
    model.eval()
    if load_noise_scheduler:
        noise_scheduler = hydra.utils.instantiate(model_cfg.noise_scheduler)
    else:
        noise_scheduler = None

    return model, noise_scheduler


def build_eval_dataset(model_path, **args):
    model_cfg = load_config(model_path)
    eval_dataset_cfg = model_cfg["dataset"]
    eval_dataset = hydra.utils.instantiate(eval_dataset_cfg, **args)
    return eval_dataset
