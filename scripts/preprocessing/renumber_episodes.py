"""Renumber the episodes of recording directories to 0, 1, 2, ... after deleting invalid ones.

Each zarr episode group is the `episode_<i>` folder itself (next to the raw recording), so renaming
the folders renames the groups. Delete the folders of invalid episodes first, then run this script.
"""
import os
import shutil

import hydra
from omegaconf import DictConfig

from egoavflow.preprocessing.common import list_episode_indices, resolve_data_dirs


@hydra.main(version_base=None, config_path="../../config/preprocessing", config_name="renumber_episodes")
def main(cfg: DictConfig):
    for data_dir in resolve_data_dirs(cfg.data_dirs):
        indices = list_episode_indices(data_dir)
        plan = [(old, new) for new, old in enumerate(indices) if old != new]
        if not plan:
            print(f"{data_dir}: {len(indices)} episodes, already numbered 0..{len(indices) - 1}")
            continue
        print(f"{data_dir}: renaming {len(plan)} of {len(indices)} episodes")
        for old, new in plan:
            print(f"  episode_{old} -> episode_{new}")
        if not cfg.auto_confirm and input("Proceed? (yes/no): ").strip().lower() not in ["yes", "y"]:
            print("Aborted.")
            continue
        # two passes so that a new name never collides with an existing folder
        for old, new in plan:
            shutil.move(os.path.join(data_dir, f"episode_{old}"), os.path.join(data_dir, f"episode_tmp_{new}"))
        for _, new in plan:
            shutil.move(os.path.join(data_dir, f"episode_tmp_{new}"), os.path.join(data_dir, f"episode_{new}"))
        assert list_episode_indices(data_dir) == list(range(len(indices)))
        print(f"{data_dir}: done")


if __name__ == "__main__":
    main()
