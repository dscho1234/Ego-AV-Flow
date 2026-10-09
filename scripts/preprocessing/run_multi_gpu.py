"""Run a preprocessing step on several GPUs by splitting the episodes of every directory.

    python run_multi_gpu.py --gpus 0,1,2,3 estimate_camera_trajectory.py \
        'data_dirs=[/data/task,/data/task_invisible]'

Every process gets a contiguous range of episodes (`episode_start` / `episode_end`) of one directory;
the directories are processed one after another. The remaining arguments are passed to the step as
Hydra overrides. Steps that open windows (manual segmentation, query point clicks) should be run
directly instead.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
from hydra import compose, initialize_config_dir

from egoavflow.common.utility.model import REPO_ROOT
from egoavflow.preprocessing.common import episode_range, resolve_data_dirs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", required=True, help="comma-separated GPU indices, e.g. 0,1,2,3")
    parser.add_argument("script", help="preprocessing script, e.g. generate_flow.py")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides of the step")
    args = parser.parse_args()
    args.gpus = [int(g) for g in args.gpus.split(",")]

    script = os.path.join(SCRIPT_DIR, os.path.basename(args.script))
    with initialize_config_dir(config_dir=os.path.join(REPO_ROOT, "config", "preprocessing"), version_base=None):
        cfg = compose(config_name="common", overrides=[o for o in args.overrides if o.split("=")[0] in
                                                       ("data_dirs", "episode_start", "episode_end")])
    passthrough = [o for o in args.overrides if o.split("=")[0] not in ("data_dirs", "episode_start", "episode_end")]

    for data_dir in resolve_data_dirs(cfg.data_dirs):
        episodes = episode_range(data_dir, cfg.episode_start, cfg.episode_end)
        chunks = [c for c in np.array_split(np.array(episodes), len(args.gpus)) if len(c) > 0]
        processes = []
        for gpu, chunk in zip(args.gpus, chunks):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [sys.executable, script, f"data_dirs=[{data_dir}]",
                   f"episode_start={int(chunk[0])}", f"episode_end={int(chunk[-1]) + 1}", *passthrough]
            print(f"[GPU {gpu}] {' '.join(cmd)}")
            processes.append(subprocess.Popen(cmd, env=env))
        codes = [p.wait() for p in processes]
        if any(codes):
            sys.exit(f"{data_dir}: a process exited with {codes}")


if __name__ == "__main__":
    main()
