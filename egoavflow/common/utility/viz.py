import colorsys

import cv2
import imageio
import matplotlib
matplotlib.use('Agg')  # use the non-interactive Agg backend (avoids Qt backend issues)
import numpy as np
from matplotlib import cm


def draw_point_tracking_sequence(
    image, sequence, draw_line=True, thickness=1, radius=3, add_alpha_channel=False, color_type='default'
):
    # sequence: (num_points,T,3)
    frame = image.copy()

    def draw_point_flow(frame, tracking_data, color, draw_line):
        # tracking_data: (T,3)
        for i in range(len(tracking_data) - 1):
            visible = tracking_data[i][2]
            if visible == 1:
                start_point = (int(tracking_data[i][0]), int(tracking_data[i][1]))
                end_point = (int(tracking_data[i + 1][0]), int(tracking_data[i + 1][1]))
                if draw_line:
                    cv2.line(
                        frame,
                        start_point,
                        end_point,
                        color,
                        thickness=thickness,
                        lineType=16,
                    )  # Adjust the thickness here
                if i == len(tracking_data) - 2:
                    cv2.circle(frame, end_point, radius, color, -1, lineType=16)
    
    def draw_point_flow_with_cool_colormap(frame, tracking_data, draw_line, cool_colormap, max_length):
        # tracking_data: (T,3)
        # Use cool colormap along the time direction (length direction)
        T = len(tracking_data)
        for i in range(T - 1):
            visible = tracking_data[i][2]
            if visible == 1:
                start_point = (int(tracking_data[i][0]), int(tracking_data[i][1]))
                end_point = (int(tracking_data[i + 1][0]), int(tracking_data[i + 1][1]))
                
                # Get color based on position along the trajectory (normalized by max_length)
                # Normalize position: i / max_length gives position along the trajectory
                color_value = i / max(1.0, float(max_length - 1)) if max_length > 1 else 0.0
                color_rgba = cool_colormap(color_value)  # Returns [R, G, B, A] in range [0, 1]
                color = np.array(color_rgba[:3]) * 255  # Convert to [0, 255] range
                color = color.astype(np.uint8)
                
                if draw_line:
                    cv2.line(
                        frame,
                        start_point,
                        end_point,
                        tuple(color.tolist()),
                        thickness=thickness,
                        lineType=16,
                    )
                if i == T - 2:
                    cv2.circle(frame, end_point, radius, tuple(color.tolist()), -1, lineType=16)

    num_points = len(sequence)
    
    if color_type == 'default':
        for i in range(num_points):
            color_map = cm.get_cmap("jet")
            color = np.array(color_map(i / max(1, float(num_points - 1)))[:3]) * 255
            color_alpha = 1
            hsv = colorsys.rgb_to_hsv(color[0], color[1], color[2])
            color = colorsys.hsv_to_rgb(hsv[0], hsv[1] * color_alpha, hsv[2])
            if add_alpha_channel:
                color = (color[0], color[1], color[2], 255.0)
            draw_point_flow(frame, sequence[i], color, draw_line)
    
    else:
        # Use matplotlib's 'cool' colormap (sequential2)
        cool_colormap = cm.get_cmap(color_type)
        
        # Find maximum trajectory length across all points for normalization
        max_length = max(len(seq) for seq in sequence) if num_points > 0 else 1
        
        # Draw all points with cool colormap along time direction
        for i in range(num_points):
            draw_point_flow_with_cool_colormap(frame, sequence[i], draw_line, cool_colormap, max_length)
    
    
    return frame


def viz_point_tracking_flow(
    frames,
    point_tracking_sequences,
    output_path,
    viz_key=[1],
    point_per_key=-1,
    viz_horizon=-1,
    draw_line=True,
    thickness=1,
    radius=1,
    add_alpha_channel=False,
    output_format='gif',  # 'gif' or 'mp4'
    fps=30,  # frames per second for mp4
    color_type='default',
):
    viz_points = []
    if isinstance(point_tracking_sequences, dict):
        for key, value in point_tracking_sequences.items():
            if key in viz_key:
                for _ in range(point_per_key):
                    viz_points.append(value[np.random.randint(len(value))])
        viz_points = np.array(viz_points)
    else:
        if point_per_key == -1:
            viz_points = point_tracking_sequences
        else:
            viz_points = point_tracking_sequences[
                np.random.randint(len(point_tracking_sequences), size=point_per_key)
            ]


    episode_length = frames.shape[0]
    if viz_horizon == -1:
        viz_horizon = episode_length
    viz_frames = []
    for i in range(1, episode_length):
        frame = draw_point_tracking_sequence(
            frames[i],
            viz_points[:, max(0, i + 1 - viz_horizon) : i + 1],
            draw_line,
            thickness,
            radius,
            add_alpha_channel,
            color_type=color_type,
        )
        # Add frame number to the frame
        frame_text = f"Frame: {i}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # White color
        thickness = 1
        # Get text size to position it properly
        (text_width, text_height), baseline = cv2.getTextSize(frame_text, font, font_scale, thickness)
        # Position text at top-left corner with some padding
        text_x = 10
        text_y = 20
        # Draw text (no background rectangle)
        viz_frames.append(frame)
    
    if output_path is not None:
        if output_format.lower() == 'mp4':
            # Save as MP4 using cv2.VideoWriter
            height, width = viz_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in viz_frames:
                # Convert BGR to RGB if needed (cv2 uses BGR)
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                video_writer.write(frame_bgr)
            
            video_writer.release()
        else:
            # Save as GIF using imageio
            imageio.mimsave(output_path, viz_frames, duration=1)
    else:
        return viz_frames

