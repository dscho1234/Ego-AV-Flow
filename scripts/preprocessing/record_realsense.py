"""Record egocentric RGB-D (+ IMU) demonstrations with an Intel RealSense D435(i).

Every episode is written to <out_dir>/episode_<i>/ as PNG frames with their timestamps:

    episode_<i>/
    ├── color/<timestamp>.png      # BGR8, 640x480
    ├── depth/<timestamp>.png      # uint16 depth in mm, aligned to the color frame
    ├── depth_vis/<timestamp>.png  # colorized depth (for inspection only)
    ├── imu_data.json
    ├── rgbd_timestamps.{txt,json}
    ├── association.txt
    ├── K_color.txt, K_depth.txt
    └── metadata.json

The first `num_lag_steps` saved frames are camera motion before the demonstration, used only
to initialize SLAM. Press ESC to stop an episode early.
"""
import argparse
import json
import signal
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def colorize_depth(depth):
    valid = depth > 0
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth[valid], [2, 98])
    scaled = np.clip((depth.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    vis = cv2.applyColorMap(((1.0 - scaled) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    vis[~valid] = 0
    return vis


def ask_to_continue():
    while True:
        answer = input("Continue to next episode? (y/n): ").strip().lower()
        if answer == "y":
            return True
        if answer == "n":
            return False
        print("Please enter 'y' for yes or 'n' for no.")


def record_episode(episode_dir, frames_per_episode, buffer_frames, num_lag_steps):
    for sub in ["color", "depth", "depth_vis"]:
        (episode_dir / sub).mkdir(parents=True, exist_ok=True)
    imu_records = []  # (timestamp_ms, stream, (x, y, z))
    rgbd_timestamps = []

    width, height, fps = 640, 480, 30
    gyro_hz, accel_hz = 200, 250

    cfg = rs.config()
    cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, gyro_hz)
    cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, accel_hz)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    pipe = rs.pipeline()

    # The callback only pushes frames to queues; saving happens in the main loop.
    color_q = deque(maxlen=400)
    depth_q = deque(maxlen=400)
    imu_q = deque(maxlen=20000)
    align = rs.align(rs.stream.color)

    def on_frame(frame):
        if frame.is_motion_frame():
            m = frame.as_motion_frame()
            md = m.get_motion_data()
            stream = "gyro" if m.get_profile().stream_type() == rs.stream.gyro else "accel"
            imu_q.append((m.get_timestamp(), stream, (md.x, md.y, md.z)))
            return
        if frame.is_frameset():
            fs = frame.as_frameset()
            ts = fs.get_timestamp()
            aligned = align.process(fs)
            c = aligned.get_color_frame()
            d = aligned.get_depth_frame()
            if c:
                color_q.append((ts, c))
            if d:
                depth_q.append((ts, d))

    profile = pipe.start(cfg, on_frame)

    print("Warming up camera (100 frames)...")
    for _ in range(100):
        while color_q or depth_q or imu_q:
            if imu_q:
                imu_q.popleft()
            if color_q:
                color_q.popleft()
            if depth_q:
                depth_q.popleft()
        time.sleep(0.01)
    print("Warm-up completed!")

    def get_K(stream):
        intr = profile.get_stream(stream).as_video_stream_profile().get_intrinsics()
        return np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)

    np.savetxt(episode_dir / "K_color.txt", get_K(rs.stream.color), fmt="%.6f",
               header=f"Color K saved: {datetime.now()}", comments="")
    np.savetxt(episode_dir / "K_depth.txt", get_K(rs.stream.depth), fmt="%.6f",
               header=f"Depth K saved: {datetime.now()}", comments="")
    metadata = {
        "episode": int(episode_dir.name.split("_")[-1]),
        "timestamp": datetime.now().isoformat(),
        "resolution": {"width": width, "height": height},
        "fps": fps,
        "frames": frames_per_episode,
        "imu_accel_fps": accel_hz,
        "imu_gyro_fps": gyro_hz,
    }
    with open(episode_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for sensor in profile.get_device().query_sensors():
        try:
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1)
            if sensor.supports(rs.option.frames_queue_size):
                sensor.set_option(rs.option.frames_queue_size, 1024)
        except Exception:
            pass

    saved_color = 0
    saved_depth = 0
    running = True

    def handle_sigint(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while running and (saved_color < frames_per_episode or saved_depth < frames_per_episode):
            while imu_q:
                imu_records.append(imu_q.popleft())
            if saved_color == buffer_frames + num_lag_steps:
                print("initial lag steps completed")

            # The first `buffer_frames` frames are dropped.
            while saved_color < frames_per_episode and color_q:
                ts, c = color_q.popleft()
                if saved_color >= buffer_frames:
                    img = np.asanyarray(c.get_data())
                    cv2.imwrite(str(episode_dir / "color" / f"{ts / 1000.0:.6f}.png"), img)
                    cv2.imshow("Recording Color Video", img)
                    rgbd_timestamps.append({
                        "frame_idx": saved_color - buffer_frames,
                        "color_timestamp_ms": ts,
                        "depth_timestamp_ms": ts,
                    })
                saved_color += 1

            while saved_depth < frames_per_episode and depth_q:
                ts, d = depth_q.popleft()
                if saved_depth >= buffer_frames:
                    depth = np.asanyarray(d.get_data())
                    name = f"{ts / 1000.0:.6f}.png"
                    cv2.imwrite(str(episode_dir / "depth" / name), depth)
                    cv2.imwrite(str(episode_dir / "depth_vis" / name), colorize_depth(depth))
                saved_depth += 1

            if cv2.waitKey(1) & 0xFF == 27:
                print("ESC pressed. Stopping recording.")
                break
            time.sleep(0.001)
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        while imu_q:
            imu_records.append(imu_q.popleft())

        if imu_records:
            imu_data = [
                {"timestamp": ts, "type": stream, "x": x, "y": y, "z": z}
                for ts, stream, (x, y, z) in imu_records
            ]
            with open(episode_dir / "imu_data.json", "w") as f:
                json.dump(imu_data, f, indent=2)

        if rgbd_timestamps:
            with open(episode_dir / "rgbd_timestamps.txt", "w") as f:
                f.write("# frame_idx color_timestamp_ms depth_timestamp_ms\n")
                for d in rgbd_timestamps:
                    f.write(f"{d['frame_idx']} {d['color_timestamp_ms']:.3f} {d['depth_timestamp_ms']:.3f}\n")
            with open(episode_dir / "rgbd_timestamps.json", "w") as f:
                json.dump(rgbd_timestamps, f, indent=2)

        # Match every color frame with the depth frame closest in time.
        color_files = sorted((episode_dir / "color").glob("*.png"))
        depth_files = sorted((episode_dir / "depth").glob("*.png"))
        depth_ts = np.array([float(p.stem) for p in depth_files])
        with open(episode_dir / "association.txt", "w") as f:
            for color_file in color_files:
                color_ts = float(color_file.stem)
                if len(depth_files) > 0:
                    closest = depth_files[int(np.argmin(np.abs(depth_ts - color_ts)))]
                    f.write(f"{color_ts:.6f} color/{color_file.name} {float(closest.stem):.6f} depth/{closest.name}\n")

        print(f"{episode_dir.name} completed: {saved_color} color frames, {saved_depth} depth frames, "
              f"{len(imu_records)} IMU samples")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", required=True, help="recording directory, e.g. data/<task>")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--episode_start", type=int, default=0)
    parser.add_argument("--episode_frames", type=int, default=200, help="frames of the demonstration itself")
    parser.add_argument("--num_lag_steps", type=int, default=100,
                        help="frames recorded before the demonstration to initialize SLAM")
    parser.add_argument("--buffer_frames", type=int, default=10, help="frames dropped at the start")
    args = parser.parse_args()

    frames_per_episode = args.episode_frames + args.num_lag_steps + args.buffer_frames
    for episode in range(args.episode_start, args.episode_start + args.num_episodes):
        print(f"\n=== Episode {episode} ===")
        if not ask_to_continue():
            print("Exiting.")
            sys.exit(0)
        record_episode(Path(args.out_dir) / f"episode_{episode}", frames_per_episode,
                       args.buffer_frames, args.num_lag_steps)
    print(f"\nAll {args.num_episodes} episodes recorded in {args.out_dir}")


if __name__ == "__main__":
    main()
