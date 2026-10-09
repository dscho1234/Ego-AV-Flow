
import numpy as np
import torch


def normalize_vector(v):
    v_mag = np.linalg.norm(v, axis=-1, keepdims=True)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag


def cross_product(u, v):
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = np.stack((i, j, k), axis=1)
    return out
        

def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]
        
    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    y = cross_product(z, x)
    
    x = x.reshape(-1, 3, 1)
    y = y.reshape(-1, 3, 1)
    z = z.reshape(-1, 3, 1)
    matrix = np.concatenate((x, y, z), axis=2)
    return matrix


def compute_ortho6d_from_rotation_matrix(matrix):
    # The ortho6d represents the first two column vectors a1 and a2 of the
    # rotation matrix: [ | , |,  | ]
    #                  [ a1, a2, a3]
    #                  [ | , |,  | ]
    ortho6d = matrix[:, :, :2].transpose(0, 2, 1).reshape(matrix.shape[0], -1)
    return ortho6d


def compute_fix_mask_weights(pred_horizon, num_fix_steps, num_fresh_steps):
    weights = torch.zeros(pred_horizon)
    for i in range(pred_horizon):
        if i < num_fix_steps:
            weights[i] = 1.0
        elif i>=num_fix_steps and i < pred_horizon-num_fresh_steps:
            c_i = (pred_horizon-num_fresh_steps-i)/(pred_horizon-num_fresh_steps-num_fix_steps+1)
            weights[i] = c_i*((np.exp(c_i)-1)/(np.exp(1)-1))
        else:
            weights[i] = 0.0
    
    return weights
