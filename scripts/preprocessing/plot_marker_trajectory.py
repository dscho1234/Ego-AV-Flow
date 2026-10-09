"""Plot the gripper trajectories and the 3D query point tracks of all episodes in the marker frame.

Writes <data_dir>/marker_frame_trajectories.html (interactive, plotly) for a visual check that the
hand pose, the camera trajectory, and the flow are consistent before training.
"""
import os

import hydra
import numpy as np
import plotly.graph_objects as go
import zarr
from omegaconf import DictConfig

from egoavflow.preprocessing.common import episode_range, resolve_data_dirs


def to_marker_frame(points_camera, T_mc):
    """Points [T, 3] in the camera frame of every timestep -> marker frame, with T_mc [T, 4, 4]."""
    homo = np.concatenate([points_camera, np.ones((len(points_camera), 1))], axis=1)
    return np.einsum("tij,tj->ti", T_mc, homo)[:, :3]


@hydra.main(version_base=None, config_path="../../config/preprocessing", config_name="plot_marker_trajectory")
def main(cfg: DictConfig):
    pose_key = "T_mc_from_first_detection_droid" if cfg.use_unoptimized_pose else "T_mc_opt_droid"
    for data_dir in resolve_data_dirs(cfg.data_dirs):
        root = zarr.open(data_dir, mode="r")
        fig = go.Figure()
        for episode_idx in episode_range(data_dir, cfg.episode_start, cfg.episode_end):
            key = f"episode_{episode_idx}"
            if key not in root or pose_key not in root[key] or "dift_point_tracking_sequence" not in root[key]:
                print(f"{key}: missing {pose_key} or dift_point_tracking_sequence, skipping")
                continue
            episode = root[key]
            T_mc = episode[pose_key][:]
            gripper = to_marker_frame(episode["proprioception"][:, :3], T_mc)
            fig.add_trace(go.Scatter3d(
                x=gripper[:, 0], y=gripper[:, 1], z=gripper[:, 2], mode="lines",
                line=dict(width=4, color=np.arange(len(gripper)), colorscale="Viridis"),
                name=f"ep {episode_idx} gripper", legendgroup=key,
            ))
            tracks = episode["dift_point_tracking_sequence"][:]  # [N, T, 4]
            assert tracks.shape[1] == len(T_mc)
            for point_idx, track in enumerate(tracks):
                points = to_marker_frame(track[:, :3], T_mc)
                if cfg.show_only_visible:
                    points = points[track[:, 3] == 1]
                fig.add_trace(go.Scatter3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
                    marker=dict(size=2), name=f"ep {episode_idx} point {point_idx}", legendgroup=key, showlegend=False,
                ))
        fig.update_layout(scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)"),
                          title=f"{os.path.basename(data_dir)} in the marker frame ({pose_key})")
        out_path = os.path.join(data_dir, "marker_frame_trajectories.html")
        fig.write_html(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
