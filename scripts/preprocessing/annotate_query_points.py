"""Click the query points on the first frame of episode 0 (needs a display).

The points are stored as `dift_points` [N, 2] (x, y in 640x480 pixels). The other episodes receive
their query points by DIFT matching in `generate_flow.py`. Pass `episode_start` / `episode_end` to
label other episodes by hand instead. The clicks are also saved to episode_<i>/query_points.json and
reused when the script is run again (`reuse_clicks=true`), so that no window is opened.
"""
import json
import os

import cv2
import hydra
import matplotlib
import numpy as np
import zarr
from omegaconf import DictConfig

from egoavflow.preprocessing.common import resolve_data_dirs, save_array

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

POINTS_FILE = "query_points.json"


def collect_mouse_points(image, n_points):
    """Returns the clicked [n_points, 2] pixel coordinates of an RGB image, or None on ESC."""
    points = []
    display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    window_name = f"Click {n_points} points (ESC: skip)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            points.append([x, y])
            vis = display.copy()
            for idx, (px, py) in enumerate(points, 1):
                cv2.circle(vis, (px, py), 5, (0, 0, 255), -1)
                cv2.circle(vis, (px, py), 8, (255, 255, 255), 2)
                cv2.putText(vis, f"{idx}", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(window_name, vis)
            print(f"Point {len(points)}/{n_points}: ({x}, {y})")

    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, display)
    while len(points) < n_points:
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyWindow(window_name)
            return None
    cv2.waitKey(1)
    cv2.destroyWindow(window_name)
    cv2.waitKey(10)
    return np.array(points)


@hydra.main(version_base=None, config_path="../../config/preprocessing", config_name="query_points")
def main(cfg: DictConfig):
    start = 0 if cfg.episode_start is None else cfg.episode_start
    end = start + 1 if cfg.episode_end is None else cfg.episode_end
    for data_dir in resolve_data_dirs(cfg.data_dirs):
        root = zarr.open(data_dir, mode="a")
        for episode_idx in range(start, end):
            episode = root[f"episode_{episode_idx}"]
            # first frame of the episode (the first frame in which all objects are visible)
            first_frame = episode["camera_0/rgb"][0]
            points_path = os.path.join(data_dir, f"episode_{episode_idx}", POINTS_FILE)
            if cfg.reuse_clicks and os.path.exists(points_path):
                with open(points_path) as f:
                    points = np.array(json.load(f)["points"])
                print(f"Using the points saved in {points_path}")
            else:
                points = collect_mouse_points(first_frame, cfg.n_points)
                if points is None:
                    print(f"{data_dir} episode {episode_idx}: skipped")
                    continue
                with open(points_path, "w") as f:
                    json.dump({"points": points.tolist()}, f)
            save_array(episode, "dift_points", points)

            plt.figure()
            plt.imshow(first_frame)
            plt.scatter(points[:, 0], points[:, 1], c="r", s=50)
            plt.axis("off")
            plt.savefig(os.path.join(data_dir, f"episode_{episode_idx}", f"episode_{episode_idx}_points.png"))
            plt.close()


if __name__ == "__main__":
    main()
