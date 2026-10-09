import concurrent.futures
import glob
import os
import re

import cv2
import imageio
import numcodecs
import numpy as np
import zarr
from hydra.utils import to_absolute_path

from egoavflow.common.imagecodecs_numcodecs import JpegXl, register_codecs
from egoavflow.common.utility.model import REPO_ROOT

register_codecs()

THIRD_PARTY_ROOT = os.path.join(REPO_ROOT, "third_party")
HAMER_ROOT = os.path.join(THIRD_PARTY_ROOT, "hamer")
DROID_SLAM_ROOT = os.path.join(THIRD_PARTY_ROOT, "DROID-SLAM")
GROUNDED_SAM_ROOT = os.path.join(THIRD_PARTY_ROOT, "Grounded-Segment-Anything")

_EPISODE_DIR = re.compile(r"^episode_(\d+)$")


def resolve_data_dirs(data_dirs):
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    return [to_absolute_path(d) for d in data_dirs]


def list_episode_indices(data_dir):
    """Sorted indices of the `episode_<i>` folders of a recording directory."""
    indices = []
    for name in os.listdir(data_dir):
        match = _EPISODE_DIR.match(name)
        if match and os.path.isdir(os.path.join(data_dir, name)):
            indices.append(int(match.group(1)))
    return sorted(indices)


def episode_range(data_dir, episode_start=None, episode_end=None):
    """Episodes [episode_start, episode_end) of a recording directory (default: all of them)."""
    indices = list_episode_indices(data_dir)
    start = 0 if episode_start is None else episode_start
    end = (indices[-1] + 1 if indices else 0) if episode_end is None else episode_end
    return [i for i in indices if start <= i < end]


def imread(path, flags=cv2.IMREAD_COLOR):
    image = cv2.imread(path, flags)
    if image is None:
        with open(path, "rb") as f:
            if f.read(24).startswith(b"version https://git-lfs"):
                raise RuntimeError(f"{path} is a Git LFS pointer; download the file with `git lfs pull`")
        raise RuntimeError(f"Could not read {path}")
    return image


def load_rgbd_pngs(episode_dir):
    """Raw recording of an episode: RGB frames [T, H, W, 3] (uint8) and depth [T, H, W] (uint16, mm)."""
    color_paths = sorted(glob.glob(os.path.join(episode_dir, "color", "*.png")))
    depth_paths = sorted(glob.glob(os.path.join(episode_dir, "depth", "*.png")))
    if len(color_paths) == 0:
        raise FileNotFoundError(f"No PNG files found in {episode_dir}/color")
    if len(color_paths) != len(depth_paths):
        print(f"Warning: {len(color_paths)} color and {len(depth_paths)} depth frames in {episode_dir}")
        n = min(len(color_paths), len(depth_paths))
        color_paths, depth_paths = color_paths[:n], depth_paths[:n]
    rgb = np.stack([cv2.cvtColor(imread(p), cv2.COLOR_BGR2RGB) for p in color_paths])
    depth = np.stack([imread(p, cv2.IMREAD_UNCHANGED) for p in depth_paths])
    return rgb, depth


def write_episode(
    root,
    data_dir,
    episode_idx,
    rgb,
    depth,
    info,
    save_video=True,
    video_fps=30,
    max_workers=8,
    **arrays,
):
    """Write an episode group: `camera_0/{rgb,depth}`, `info`, and the given arrays.

    RGB frames are stored with the JPEG XL codec (one chunk per frame), everything else with the
    zarr defaults, which is the layout the training data loader expects.
    """
    episode = root.require_group(f"episode_{episode_idx}")
    camera = episode.require_group("camera_0")
    if "rgb" in camera:
        del camera["rgb"]
    n, h, w, c = rgb.shape
    rgb_arr = camera.create_dataset(
        "rgb",
        shape=(n, h, w, c),
        chunks=(1, h, w, c),
        dtype=np.uint8,
        compressor=JpegXl(level=80, numthreads=1),
    )

    def encode(i):
        try:
            rgb_arr[i] = rgb[i]
            _ = rgb_arr[i]
            return True
        except Exception as e:
            print(e)
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        if not all(executor.map(encode, range(n))):
            raise RuntimeError("Failed to encode image!")

    if "depth" in camera:
        del camera["depth"]
    camera["depth"] = zarr.array(depth)
    if save_video:
        imageio.mimwrite(os.path.join(data_dir, f"episode_{episode_idx}", "camera_0.mp4"), rgb, fps=video_fps)

    for key, value in arrays.items():
        if key in episode:
            del episode[key]
        episode[key] = zarr.array(value)

    info_arr = episode.require_dataset(
        "info",
        shape=(len(info),),
        chunks=(1,),
        dtype="object",
        object_codec=numcodecs.JSON(),
        overwrite=True,
    )
    for i, info_dict in enumerate(info):
        info_arr[i] = info_dict


def save_array(group, key, value):
    if key in group:
        del group[key]
    group[key] = zarr.array(value)
