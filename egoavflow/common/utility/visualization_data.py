"""
Script to save visualization data for later use in test_vis.py
"""
import os
import pickle
import numpy as np
import torch

def save_visualization_data(visualization_data_list, save_path):
    """
    Save visualization data to pickle file.
    
    Args:
        visualization_data_list: List of visualization data dictionaries from Policy.visualization_data_list
        save_path: Path to save the pickle file
    """
    # Convert data to pickleable format
    pickleable_data = []
    
    for viz_data in visualization_data_list:
        pickleable_viz_data = {}
        
        # Convert open3d mesh to pickleable format
        if viz_data.get('color_mesh') is not None:
            mesh = viz_data['color_mesh']
            pickleable_viz_data['color_mesh'] = {
                'vertices': np.asarray(mesh.vertices) if len(mesh.vertices) > 0 else None,
                'triangles': np.asarray(mesh.triangles) if len(mesh.triangles) > 0 else None,
                'vertex_colors': np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None,
            }
        else:
            pickleable_viz_data['color_mesh'] = None
        
        # Convert open3d point cloud to pickleable format
        if viz_data.get('point_cloud') is not None:
            point_cloud = viz_data['point_cloud']
            pickleable_viz_data['point_cloud'] = {
                'points': np.asarray(point_cloud.points) if len(point_cloud.points) > 0 else None,
                'colors': np.asarray(point_cloud.colors) if point_cloud.has_colors() else None,
            }
        else:
            pickleable_viz_data['point_cloud'] = None
        
        # Convert camera_pose (torch tensor to numpy)
        if viz_data.get('camera_pose') is not None:
            camera_pose = viz_data['camera_pose']
            if isinstance(camera_pose, torch.Tensor):
                pickleable_viz_data['camera_pose'] = camera_pose.detach().cpu().numpy()
            else:
                pickleable_viz_data['camera_pose'] = camera_pose
        else:
            pickleable_viz_data['camera_pose'] = None
        
        # Copy other data (should already be pickleable)
        pickleable_viz_data['query_points'] = viz_data.get('query_points')
        pickleable_viz_data['camera_intrinsics'] = viz_data.get('camera_intrinsics')
        pickleable_viz_data['image_size'] = viz_data.get('image_size')
        pickleable_viz_data['current_camera_pose_list'] = viz_data.get('current_camera_pose_list')
        pickleable_viz_data['current_visibility_results_dict'] = viz_data.get('current_visibility_results_dict')
        pickleable_viz_data['active_camera_pose_list'] = viz_data.get('active_camera_pose_list')
        pickleable_viz_data['active_visibility_results_dict'] = viz_data.get('active_visibility_results_dict')
        pickleable_viz_data['retargeted_active_camera_pose_list'] = viz_data.get('retargeted_active_camera_pose_list')
        pickleable_viz_data['retargeted_active_visibility_results_dict'] = viz_data.get('retargeted_active_visibility_results_dict')
        pickleable_viz_data['active_visibility_results_info'] = viz_data.get('active_visibility_results_info')
        pickleable_viz_data['retargeted_active_visibility_results_info'] = viz_data.get('retargeted_active_visibility_results_info')
        pickleable_viz_data['selected_indices'] = viz_data.get('selected_indices')
        pickleable_viz_data['query_point_indices'] = viz_data.get('query_point_indices')
        
        pickleable_data.append(pickleable_viz_data)
    
    # Save to pickle file
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(pickleable_data, f)
    
    print(f"Saved {len(pickleable_data)} visualization data entries to {save_path}")

