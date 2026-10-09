import os

import omegaconf
from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    OmegaConf.register_new_resolver("eval", eval)
except:
    pass
OmegaConf.register_new_resolver("egoavflow_root", lambda: REPO_ROOT, replace=True)


# Checkpoints trained before the package was renamed store these `_target_` prefixes.
_LEGACY_TARGET_PREFIXES = (
    ("im2flow2act.", "egoavflow."),
    ("models.", "egoavflow.models."),
)


def _rename_legacy_targets(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "_target_" and isinstance(value, str):
                for old, new in _LEGACY_TARGET_PREFIXES:
                    if value.startswith(old):
                        node[key] = new + value[len(old):]
                        break
            else:
                _rename_legacy_targets(value)
    elif isinstance(node, list):
        for value in node:
            _rename_legacy_targets(value)


def load_config(model_path):
    cfg = omegaconf.OmegaConf.load(os.path.join(model_path, ".hydra/config.yaml"))
    cfg = OmegaConf.to_container(cfg, resolve=False)
    _rename_legacy_targets(cfg)
    cfg = OmegaConf.create(cfg)
    return cfg
