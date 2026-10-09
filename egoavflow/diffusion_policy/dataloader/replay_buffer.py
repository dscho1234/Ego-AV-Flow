import concurrent.futures
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
import zarr

from egoavflow.common.imagecodecs_numcodecs import register_codecs

register_codecs()


class ReplayBuffer:
    def __init__(
        self,
        data_path,
        load_camera_ids=[],
        camera_resize_shape=[],
        max_episode=None,
        max_workers=32,
    ) -> None:
        self.data_path = data_path
        self.load_camera_ids = load_camera_ids
        self.camera_resize_shape = camera_resize_shape
        self.max_workers = max_workers
        self.max_episode = max_episode
        self.initiate_memory_buffer()
        self.load_data_to_memory()

    def initiate_memory_buffer(self):
        self.memory_buffer = defaultdict(list)

    def load_data_to_memory(self):
        load_episode_num = 0
        for path in self.data_path:
            root = zarr.open(path, mode="a")
            episodes = list(root.group_keys())
            for episode in episodes:
                self.memory_buffer["action"].append(
                    self.load_low_dim_data(root, osp.join(episode, "action"))
                )
                self.memory_buffer["proprioception"].append(
                    self.load_low_dim_data(root, osp.join(episode, "proprioception"))
                )
                for camera_ids in self.load_camera_ids: # [0]
                    cam_name = f"camera_{camera_ids}"
                    self.memory_buffer[cam_name].append(
                        self.load_visual_data(root, osp.join(episode, cam_name, "rgb"))
                    )
                load_episode_num += 1
                if (
                    self.max_episode is not None
                    and load_episode_num >= self.max_episode
                ):
                    break

        self.eps_end = np.cumsum([len(x) for x in self.memory_buffer["action"]])
        for k, v in self.memory_buffer.items():
            self.memory_buffer[k] = np.concatenate(v)

    def load_low_dim_data(self, root, low_dim_path):
        return root[low_dim_path][:].astype(np.float32)

    def load_visual_data(self, root, visual_path, dim=3):
        visual_shape = root[visual_path].shape
        np_arr_shape = (
            (visual_shape[0], *self.camera_resize_shape, dim)
            if self.camera_resize_shape
            else visual_shape
        )
        np_arr = np.zeros(np_arr_shape, dtype=np.uint8)

        def load_img(zarr_arr, visual_path, index, np_arr):
            try:
                if self.camera_resize_shape:
                    np_arr[index] = cv2.resize(
                        zarr_arr[visual_path][index], self.camera_resize_shape
                    )
                else:
                    np_arr[index] = zarr_arr[visual_path][index]
                return True
            except Exception as e:
                print(e)
                return False

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = set()
            for i in range(visual_shape[0]):
                futures.add(executor.submit(load_img, root, visual_path, i, np_arr))

            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError("Failed to load image!")

        return np_arr

    def __repr__(self) -> str:
        rep = ""
        for k, v in self.memory_buffer.items():
            rep += f"{k}, {v.shape}\n"
        rep += f"eps_end, {self.eps_end}\n"
        return rep

    def __getitem__(self, key):
        return self.memory_buffer[key]

    def remove_key(self, key):
        del self.memory_buffer[key]

