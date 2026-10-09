import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def project_points_to_image(points_3d, intrinsics, extrinsics):
    """
    points_3d: (T, N, 3)
    intrinsics: (T, 3, 3)
    extrinsics: (T, 4, 4) # T_wc or T_mc
    Returns:
        points_2d: (T, N, 2)
    """
    T, N, _ = points_3d.shape
    extrinsics_inv = np.linalg.inv(extrinsics) # # T_cw or T_cm
    # 1. Homogeneous coordinates
    pts_h = np.concatenate([points_3d, np.ones((T, N, 1))], axis=-1)  # (T, N, 4)
    # 2. World (or marker) to camera coordinate(batched matmul)
    cam_pts = np.einsum('tij,tnj->tni', extrinsics_inv, pts_h)  # (T, N, 4)
    cam_pts = cam_pts[..., :3]  # (T, N, 3)
    # 3. Camera to image (batched matmul)
    img_pts = np.einsum('tij,tnj->tni', intrinsics, cam_pts)  # (T, N, 3)
    # 4. Normalize by z
    img_pts = img_pts / img_pts[..., 2:3]
    points_2d = img_pts[..., :2]  # (T, N, 2)
    return points_2d

def get_pixel_queries(xy_coords: torch.Tensor, depth: torch.Tensor, intrinsic: torch.Tensor, extrinsic: torch.Tensor):
    """
    Args:
        xy_coords: (N, 2) or (B, N, 2) tensor, pixel coordinates (x, y)
        depth: (H, W) or (B, H, W) tensor, depth map of the first frame
        intrinsic: (3, 3) or (B, 3, 3) tensor, camera intrinsics of the first frame
        extrinsic: (4, 4) or (B, 4, 4) tensor, camera extrinsics of the first frame
    Returns:
        queries: (N, 4) or (B, N, 4) tensor, 3D queries in the form (0, X, Y, Z)
    """
    if len(depth.shape) == 2:
        # (H, W) -> (1, H, W)
        return get_pixel_queries(
            xy_coords.unsqueeze(0),
            depth.unsqueeze(0),
            intrinsic.unsqueeze(0),
            extrinsic.unsqueeze(0)
        ).squeeze(0)

    # assume batch size 1
    assert depth.shape[0] == 1, "batch size must be 1"
    H, W = depth.shape[1:]
    N = xy_coords.shape[-2]
    
    # use the first frame (frame 0)
    ji = torch.round(xy_coords).to(torch.int32)
    d = depth[0][ji[..., 1], ji[..., 0]]  # only the depth of the first frame is used
    mask = d > 0
    d = d[mask]
    
    # index xy_coords with a mask of matching dimensions
    if xy_coords.dim() == 3:
        # xy_coords: (B, N, 2), mask: (B, N) -> (num_valid, 2)
        xy = xy_coords[mask]
        ji = ji[mask]
    else:
        # xy_coords: (N, 2), mask: (N,) -> (num_valid, 2)
        xy = xy_coords[mask]
        ji = ji[mask]

    inv_intrinsic0 = torch.linalg.inv(intrinsic[0])  # intrinsics of the first frame
    extrinsics0 = extrinsic[0]

    # xy has shape (num_valid, 2)
    xy_homo = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)  # (num_valid, 3)
    xy_homo = torch.einsum('ij,nj->ni', inv_intrinsic0, xy_homo)  # (num_valid, 3)
    local_coords = xy_homo * d[..., None]  # (num_valid, 3)
    local_coords_homo = torch.cat([local_coords, torch.ones_like(local_coords[..., :1])], dim=-1)  # (num_valid, 4)
    world_coords = torch.einsum('ij,nj->ni', extrinsics0, local_coords_homo)  # (num_valid, 4)
    world_coords = world_coords[..., :3]  # (num_valid, 3)

    queries = torch.cat([torch.zeros_like(xy[..., :1]), world_coords], dim=-1).to(depth.device)  # (num_valid, 4)
    return queries


def fill_depth_zeros(depth_img):
    mask = (depth_img == 0)
    if not np.any(mask):
        return depth_img
    distance, indices = distance_transform_edt(mask, return_indices=True)
    filled = depth_img[tuple(indices)]
    return filled
