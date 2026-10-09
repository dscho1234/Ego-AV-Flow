"""Scene/robot geometry for visibility reasoning.

Scene meshes are reconstructed with nvblox, the Z1 (camera arm) is rendered from its URDF, and
visibility of query points is computed by raycasting against the combined scene.
"""
import os
import time

import cv2
import matplotlib
matplotlib.use('Agg')  # use the non-interactive Agg backend (avoids Qt backend issues)
import numpy as np
import open3d as o3d
import torch
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from urdf_parser_py.urdf import URDF

# ========= Verbose logging control =========
VERBOSE = False

def vprint(*args, **kwargs) -> None:
    """Print only when VERBOSE is True."""
    if VERBOSE:
        print(*args, **kwargs)


# SO3 constraint null objective function
class SO3Constraint:
    def __init__(self, SO3_des=None):
        if SO3_des is None:
            # Default to identity matrix (no rotation preference)
            self.SO3_des = np.eye(3)
        else:
            self.SO3_des = SO3_des.copy()
    
    def evaluate(self, SO3):
        # SO3 error metric: 0.5 * (3 - trace(R * R_des^T))
        # This measures the deviation from desired rotation
        so3_err = 0.5 * (3 - np.trace(SO3 @ self.SO3_des.T))
        return so3_err

        
# ========= Z1 Robot Visualizer Class =========
class Z1RobotVisualizer:
    def __init__(self, urdf_path, mesh_base_path, pipe_env = None, T_E_C=None):
        """
        Z1 robot visualizer
        
        Args:
            urdf_path: path to the URDF file
            mesh_base_path: base directory of the mesh files
        """
        self.urdf_path = urdf_path
        self.mesh_base_path = mesh_base_path
        self.T_E_C = T_E_C
        self.robot = None
        self.joint_angles = np.zeros(6)  # 6 joint angles
        self.link_transforms = {}  # transform of each link
        self.meshes = {}  # loaded meshes
        # color

        # silver 
        self.link_colors = {
            'link00': [0.8, 0.8, 0.8],    
            'link01': [0.8, 0.8, 0.8],         
            'link02': [0.8, 0.8, 0.8],         
            'link03': [0.8, 0.8, 0.8],         
            'link04': [0.8, 0.8, 0.8],         
            'link05': [0.8, 0.8, 0.8],         
            'link06': [0.8, 0.8, 0.8],         
            'z1_GripperMover': [0.8, 0.8, 0.8],  
            'z1_GripperStator': [0.8, 0.8, 0.8],
        }

        # Joint limits for IK
        # NOTE: the default limits differ from the joint range the robot can actually reach, so they are adjusted here
        self.joint_limits = [
            (-2.618, 2.618),   # J1: ±150°
            (0, 3.142),        # J2: 0—180°
            (-2.879, 0),       # J3: -165°—0
            (-1.75, 1.65),      # J4: ±80° # (-1.396, 1.396) (default)
            (-1.75, 1.6),      # J5: ±85° # (-1.484, 1.484) (default)
            (-2.793, 2.793)    # J6: ±160°
        ]
        

        # optional arm interface (not needed for visualization / FK)
        try:
            self.arm_interface = pipe_env # unitree_arm_interface.ArmInterface(hasGripper=True)
            vprint("Arm interface set")
        except Exception as e:
            vprint(f"Failed to set arm interface: {e}")
            self.arm_interface = None
        
        # parse URDF
        self.parse_urdf()
        
        # load meshes
        self.load_meshes()
        
    def parse_urdf(self):
        """Parse the URDF file to extract the robot structure"""
        try:
            self.robot = URDF.from_xml_file(self.urdf_path)
            vprint(f"Parsed URDF: {len(self.robot.joints)} joints, {len(self.robot.links)} links")
            
            # print joint info
            for i, joint in enumerate(self.robot.joints):
                vprint(f"Joint {i+1}: {joint.name}, Type: {joint.type}, Axis: {joint.axis}")
                
        except Exception as e:
            vprint(f"URDF parsing error: {e}")
            
    def load_meshes(self):
        """Load the mesh file of each link"""
        # extract mesh file paths from the URDF
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        
        for link in root.findall('link'):
            link_name = link.get('name')
            visual = link.find('visual')
            
            if visual is not None:
                geometry = visual.find('geometry')
                if geometry is not None:
                    mesh = geometry.find('mesh')
                    if mesh is not None:
                        mesh_filename = mesh.get('filename')
                        if mesh_filename and 'package://' in mesh_filename:
                            # resolve package:// paths to real file paths
                            mesh_path = mesh_filename.replace('package://z1_description/', self.mesh_base_path)
                            
                            # try the STL file first
                            stl_path = mesh_path.replace('.dae', '.STL').replace('visual/', 'collision/')
                            if os.path.exists(stl_path):
                                # load the STL file
                                mesh_obj = o3d.io.read_triangle_mesh(stl_path)
                                if len(mesh_obj.vertices) > 0:
                                    # simplify the mesh (faster raycasting)
                                    original_vertices = len(mesh_obj.vertices)
                                    original_triangles = len(mesh_obj.triangles)
                                    
                                    # reduce the number of triangles (set target_triangle_count)
                                    if link_name in ['link00', 'link01', 'link02', 'link03', 'link04']:
                                        target_triangle_count = 100  # keep at least 100 triangles
                                    else:
                                        target_triangle_count = max(100, original_triangles // 4)  # keep at least 100 triangles
                                    
                                    # simplify the mesh
                                    mesh_obj = mesh_obj.simplify_quadric_decimation(target_triangle_count)
                                    
                                    # also clean up vertices (remove duplicates, etc.)
                                    mesh_obj.remove_degenerate_triangles()
                                    mesh_obj.remove_duplicated_triangles()
                                    mesh_obj.remove_duplicated_vertices()
                                    mesh_obj.remove_unreferenced_vertices()
                                    
                                    # check whether the mesh was over-simplified
                                    if len(mesh_obj.vertices) < 10:
                                        # fall back to the original mesh if over-simplified
                                        mesh_obj = o3d.io.read_triangle_mesh(stl_path)
                                        print(f"Mesh over-simplified, using the original: {link_name}")
                                    else:
                                        print(f"Simplified mesh: {link_name} - {original_vertices}->{len(mesh_obj.vertices)} vertices, {original_triangles}->{len(mesh_obj.triangles)} triangles")
                                    
                                    mesh_obj.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array(self.link_colors[link_name])[None, :], (len(mesh_obj.vertices), 1)))
                                    self.meshes[link_name] = mesh_obj
                                    vprint(f"Loaded STL mesh: {link_name} -> {stl_path} ({len(mesh_obj.vertices)} vertices)")
                                    continue
                            
                            
        # also load the gripper meshes (not defined in the URDF, but the mesh files exist)
        self.load_gripper_meshes()
                                
    def load_gripper_meshes(self):
        """Load the gripper mesh files"""
        gripper_meshes = ['z1_GripperMover', 'z1_GripperStator']
        
        for gripper_name in gripper_meshes:
            # DAE file path
            dae_path = os.path.join(self.mesh_base_path, 'meshes', 'visual', f'{gripper_name}.dae')
            # STL file path
            stl_path = os.path.join(self.mesh_base_path, 'meshes', 'collision', f'{gripper_name}.STL')
            
            try:
                # try the STL file first
                if os.path.exists(stl_path):
                    mesh_obj = o3d.io.read_triangle_mesh(stl_path)
                    if len(mesh_obj.vertices) > 0:
                        # simplify the mesh (faster raycasting)
                        original_vertices = len(mesh_obj.vertices)
                        original_triangles = len(mesh_obj.triangles)
                        
                        # reduce the number of triangles (set target_triangle_count)
                        target_triangle_count = max(100, original_triangles // 4)  # keep at least 100 triangles
                        
                        # simplify the mesh
                        mesh_obj = mesh_obj.simplify_quadric_decimation(target_triangle_count)
                        
                        # also clean up vertices (remove duplicates, etc.)
                        mesh_obj.remove_degenerate_triangles()
                        mesh_obj.remove_duplicated_triangles()
                        mesh_obj.remove_duplicated_vertices()
                        mesh_obj.remove_unreferenced_vertices()
                        
                        # check whether the mesh was over-simplified
                        if len(mesh_obj.vertices) < 10:
                            # fall back to the original mesh if over-simplified
                            mesh_obj = o3d.io.read_triangle_mesh(stl_path)
                            print(f"Gripper mesh over-simplified, using the original: {gripper_name}")
                        else:
                            print(f"Simplified gripper mesh: {gripper_name} - {original_vertices}->{len(mesh_obj.vertices)} vertices, {original_triangles}->{len(mesh_obj.triangles)} triangles")
                        
                        mesh_obj.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array(self.link_colors[gripper_name])[None, :], (len(mesh_obj.vertices), 1)))            
                        self.meshes[gripper_name] = mesh_obj
                        vprint(f"Loaded gripper STL mesh: {gripper_name} -> {stl_path} ({len(mesh_obj.vertices)} vertices)")
                        continue
                
                        
            except Exception as e:
                vprint(f"Failed to load gripper mesh: {gripper_name} - {e}")
                                
    
    def set_joint_angles(self, angles):
        """Set joint angles"""
        self.joint_angles = np.array(angles)
        vprint(f"Joint angles set: {self.joint_angles}")
    
    
    def jacobian_position(self, q):
        epsilon = 1e-6
        epsilon_inv = 1/epsilon
        T = self.compute_forward_kinematics(q)

        p = T[:3, 3]
        jac = np.zeros([3, 6])
        for i in range(6):
            q_ = q.copy()
            q_[i] = q_[i] + epsilon
            T_ = self.compute_forward_kinematics(q_)

            p_ = T_[:3, 3]
            jac[:, i] = (p_ - p)*epsilon_inv
        return jac

    def solve_ik_null_space(self, target_T, initial_guess=None, max_iterations=100, tolerance=1e-2, tolerance_null=1e-5, epsilon=1e-6):
        """
        Solve IK using pseudo-inverse with null-space approach.
        
        Args:
            target_T: Target 4x4 transformation matrix
            initial_guess: Initial joint angle guess
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance for position error
            tolerance_null: Convergence tolerance for null objective
            epsilon: Small value for numerical differentiation
            desired_rotation: 3x3 rotation matrix for null objective (default: identity)
            
        Returns:
            tuple: (success, joint_angles, iterations, final_error)
        """
        if initial_guess is None:
            initial_guess = np.zeros(6)
        
        
        q = initial_guess.copy()
        
        
        # Initialize null objective with desired rotation (default: identity)
        target_SO3 = target_T[:3, :3]
        null_obj = SO3Constraint(target_SO3)
        
        iter_taken = 0

        
        while True:
            # Compute current forward kinematics
            current_T = self.compute_forward_kinematics(q)
            
            # Compute position error only (like in the original code)
            pos_error = target_T[:3, 3] - current_T[:3, 3]
            err = np.linalg.norm(pos_error)
            
            # Compute null objective value
            current_SO3 = current_T[:3, :3]
            null_obj_val = null_obj.evaluate(current_SO3)

            # Compute Jacobian
            J = self.jacobian_position(q)
            
            # Check convergence: both position error and null objective must be satisfied
            if (err < tolerance and null_obj_val < tolerance_null) or iter_taken >= max_iterations:
                break
            else:
                iter_taken += 1
            
            
            # Pseudo-inverse approach
            
            J_dagger = np.linalg.pinv(J)
            J_null = np.eye(6) - J_dagger @ J  # null space of Jacobian
            
            # Compute null objective gradient using numerical differentiation
            phi = np.zeros(6)
            
            for i in range(6):
                q_perturb = q.copy()
                q_perturb[i] += epsilon
                # Apply joint limits to perturbed configuration
                q_perturb = np.clip(q_perturb, [limit[0] for limit in self.joint_limits], [limit[1] for limit in self.joint_limits])
                
                
                perturb_T = self.compute_forward_kinematics(q_perturb)


                perturb_SO3 = perturb_T[:3, :3]
                null_obj_val_perturb = null_obj.evaluate(perturb_SO3)
                phi[i] = (null_obj_val_perturb - null_obj_val) / epsilon
            
            
            # Update using pseudo-inverse + null-space approach
            # delta_x = ee_pos - x (position error)
            delta_x = pos_error
            delta_q = J_dagger @ delta_x - J_null @ phi
            q = q + delta_q
            
            
            # Apply joint limits
            q = np.clip(q, [limit[0] for limit in self.joint_limits], [limit[1] for limit in self.joint_limits])
        
        # Final error check (position error only, like in original code)
        current_T = self.compute_forward_kinematics(q)

        
        final_error = err # np.linalg.norm(final_pos_error)
        
        # Check if both conditions are satisfied for success
        success = (final_error < tolerance and null_obj_val < tolerance_null)
        
        
        return success, q, iter_taken, final_error, null_obj_val

    
    # jacobian ver
    # helper: current error / cost

    # adaptive damping (based on singular values / manipulability)

    
    def solve_inverse_kinematics(self, target_pose):
        """Compute joint angles for a target pose with inverse kinematics"""
        if self.arm_interface is None:
            vprint("Arm interface is not set; skipping IK.")
            return False
            
        try:
            # use the current joint angles as the initial guess
            current_q = self.joint_angles.copy()
            vprint(f"Initial joint angles: {current_q}")
            if (current_q == np.zeros(6)).all():
                initial_guess = None
            else:
                initial_guess = current_q

            
            # inverse kinematics (it often outputs fail even though the target_pose and initial_guess are identical. Maybe due to its internal logic to consider singularities, workspace limits, etc.)

            
            # custom inverse kinematics v2
            success, q_forward, iter_taken, error, null_obj_val = self.solve_ik_null_space(target_pose, initial_guess=initial_guess, max_iterations=50, tolerance=1e-2, tolerance_null=1e-3, epsilon=1e-6)
            
            
            # update with the computed joint angles
            self.joint_angles = q_forward
            vprint(f'Custom IK iter: {iter_taken} error: {error:.6f} null_obj_val: {null_obj_val:.6f}')
            vprint(f"Joint angles after IK: {q_forward}")
            
            if success:
                vprint(f"Inverse kinematics succeeded: {self.joint_angles}")
                return True
            else:
                vprint("Inverse kinematics failed: the target pose is outside the workspace.")
                return False
                
        except Exception as e:
            vprint(f"Inverse kinematics error: {e}")
            return False
        
    def compute_forward_kinematics(self, joint_angles,  gripper_angle=0.0):
        """Forward kinematics from the URDF"""
        # computed from the URDF
        self.joint_angles = joint_angles.copy()

        self.compute_forward_kinematics_urdf(gripper_angle)
        return self.link_transforms['z1_GripperMover'].copy()
    
    def compute_forward_kinematics_urdf(self, gripper_angle=0.0):
        """URDF-based forward kinematics"""
        # joint info as a dict
        joint_info = {}
        for joint in self.robot.joints:
            joint_info[joint.name] = joint
            
        # link info as a dict
        link_info = {}
        for link in self.robot.links:
            link_info[link.name] = link
            
        # initialize transforms
        self.link_transforms = {}
        
        # world -> link00 (fixed joint)
        self.link_transforms['world'] = np.eye(4)
        self.link_transforms['link00'] = np.eye(4)
        
        # compute the transform of each joint
        joint_angle_idx = 0
        for joint in self.robot.joints:
            if joint.type == 'revolute' and joint_angle_idx < len(self.joint_angles):
                # joint angle
                angle = self.joint_angles[joint_angle_idx]
                joint_angle_idx += 1
                
                # joint axis
                axis = np.array(joint.axis)
                
                # joint origin
                origin = joint.origin
                if origin is not None:
                    xyz = np.array(origin.xyz) if origin.xyz else np.zeros(3)
                    rpy = np.array(origin.rpy) if origin.rpy else np.zeros(3)
                else:
                    xyz = np.zeros(3)
                    rpy = np.zeros(3)
                
                # rotation matrix (RPY)
                if np.any(rpy):
                    rot_matrix = R.from_euler('xyz', rpy).as_matrix()
                else:
                    rot_matrix = np.eye(3)
                
                # joint rotation (rotation about the axis)
                joint_rot_matrix = R.from_rotvec(axis * angle).as_matrix()
                
                # build the transform
                T_joint = np.eye(4)
                T_joint[:3, :3] = rot_matrix @ joint_rot_matrix
                T_joint[:3, 3] = xyz
                
                # compose with the parent link transform
                parent_transform = self.link_transforms.get(joint.parent, np.eye(4))
                child_transform = parent_transform @ T_joint
                
                self.link_transforms[joint.child] = child_transform
                
            elif joint.type == 'fixed':
                # fixed joint
                origin = joint.origin
                if origin is not None:
                    xyz = np.array(origin.xyz) if origin.xyz else np.zeros(3)
                    rpy = np.array(origin.rpy) if origin.rpy else np.zeros(3)
                else:
                    xyz = np.zeros(3)
                    rpy = np.zeros(3)
                
                # rotation matrix
                if np.any(rpy):
                    rot_matrix = R.from_euler('xyz', rpy).as_matrix()
                else:
                    rot_matrix = np.eye(3)
                
                # build the transform
                T_fixed = np.eye(4)
                T_fixed[:3, :3] = rot_matrix
                T_fixed[:3, 3] = xyz
                
                # compose with the parent link transform
                parent_transform = self.link_transforms.get(joint.parent, np.eye(4))
                child_transform = parent_transform @ T_fixed
                
                self.link_transforms[joint.child] = child_transform
        
        # gripper pose (exact offsets from the URDF)
        if 'link06' in self.link_transforms:
            # gripperStator is attached to link06 with offset xyz="0.051 0.0 0.0"
            gripper_stator_offset = np.array([0.051, 0.0, 0.0])  # offset defined in the URDF
            gripper_stator_transform = self.link_transforms['link06'].copy()
            gripper_stator_transform[:3, 3] += gripper_stator_transform[:3, :3] @ gripper_stator_offset
            
            self.link_transforms['z1_GripperStator'] = gripper_stator_transform
            
            # gripperMover is attached to gripperStator with offset xyz="0.049 0.0 0"
            gripper_mover_offset = np.array([0.049, 0.0, 0.0])  # offset defined in the URDF
            gripper_mover_transform = gripper_stator_transform.copy()
            gripper_mover_transform[:3, 3] += gripper_mover_transform[:3, :3] @ gripper_mover_offset
            
            # apply the gripper rotation (about the Y axis, axis xyz="0 1 0" in the URDF)
            if gripper_angle != 0.0:
                gripper_rotation = R.from_rotvec([0, gripper_angle, 0]).as_matrix()
                gripper_mover_transform[:3, :3] = gripper_mover_transform[:3, :3] @ gripper_rotation
            
            self.link_transforms['z1_GripperMover'] = gripper_mover_transform
                
    
    def transform_mesh_to_world(self, mesh, transform):
        """Transform a mesh to the world frame"""
        
        # 1. transform the vertices directly (no mesh copy)
        vertices = np.asarray(mesh.vertices)
        
        # 2. split the transform into a 3x3 rotation and a 3x1 translation
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        
        # 3. vectorized transform
        vertices_world = vertices @ rotation.T + translation
        
        # 4. create a new mesh (minimal copy)
        transformed_mesh = o3d.geometry.TriangleMesh()
        transformed_mesh.triangles = mesh.triangles
        transformed_mesh.vertex_colors = mesh.vertex_colors
        
        # 5. assign the transformed vertices
        transformed_mesh.vertices = o3d.utility.Vector3dVector(vertices_world)
        
        return transformed_mesh
        
    
    def get_robot_meshes_in_specified_transform(self, gripper_angle=0.0, T_target=None):
        """Return all robot link meshes transformed into the given frame
        
        Args:
            gripper_angle: gripper angle
            T_target: transform of the target frame (4x4). If None, the world frame is used
        
        Returns:
            transformed_meshes: dict of meshes in the target frame
        """
        # forward kinematics
        self.compute_forward_kinematics(self.joint_angles, gripper_angle)
        
        transformed_meshes = {}
        for link_name, mesh in self.meshes.items():
            
            if link_name in self.link_transforms:
                if T_target is not None:
                    # transform from the world frame to the target frame
                    world_transform = self.link_transforms[link_name]
                    target_transform = T_target @ world_transform
                    
                    # transform the mesh to the target frame
                    target_mesh = self.transform_mesh_to_world(mesh, target_transform)
                    transformed_meshes[link_name] = target_mesh
                else:
                    # if T_target is None, use the world frame
                    world_mesh = self.transform_mesh_to_world(mesh, self.link_transforms[link_name])
                    transformed_meshes[link_name] = world_mesh
            
        return transformed_meshes

    
OFFSET_DISTANCE = 0.01 # for convex part of the constructed mesh


# Mesh quality: "high_resolution" (5mm), "medium_resolution" (10mm), "low_resolution" (20mm)
MESH_QUALITY = "medium_resolution"

if MESH_QUALITY == "high_resolution":
    VOXEL_SIZE = 0.005  # 5mm - high resolution, fragmented mesh
    
elif MESH_QUALITY == "medium_resolution":
    VOXEL_SIZE = 0.01   # 10mm - medium resolution, balanced mesh
    
else:  # low_resolution
    VOXEL_SIZE = 0.02   # 20mm - low resolution, smooth mesh
    

vprint(f"Using {MESH_QUALITY}: voxel_size={VOXEL_SIZE}m")


from scipy.ndimage import distance_transform_edt

def fill_depth_zeros(depth_img):
    # treat zero / invalid values as holes
    mask = (depth_img == 0) | (depth_img <= 0) | np.isnan(depth_img)
    if not np.any(mask):
        return depth_img
    distance, indices = distance_transform_edt(mask, return_indices=True)
    filled = depth_img[tuple(indices)]
    return filled
    
def convert_to_relative(se3_matrices):
    # SE(3) matrices [B, H, 4, 4]
    
    # Get inverse of first transformation for each batch [B, 4, 4]
    first_transforms = se3_matrices[:, 0]  # [B, 4, 4]
    first_inv = np.linalg.inv(first_transforms)  # [B, 4, 4]
    
    # Apply relative transformation: T_rel = T_first_inv @ T_current
    # Broadcasting: [B, 4, 4] @ [B, H, 4, 4] -> [B, H, 4, 4]
    relative_se3 = np.einsum('bij,bhjk->bhik', first_inv, se3_matrices)
    
    # Set first index to identity
    relative_se3[:, 0] = np.eye(4)
    
    return relative_se3


def create_camera_frustum_visualization(camera_position, camera_rotation_rpy, camera_intrinsics, image_size, max_distance=0.5):
    """
    Create the vertices used to visualize a camera frustum
    
    Args:
        camera_position: camera position [x, y, z]
        camera_rotation_rpy: camera rotation [roll, pitch, yaw] (degrees)
        camera_intrinsics: camera intrinsics (3x3)
        image_size: image size (width, height)
        max_distance: frustum depth
    
    Returns:
        frustum_vertices: frustum vertices (camera center + 4 far corners)
    """
    # camera rotation matrix
    roll, pitch, yaw = np.deg2rad(camera_rotation_rpy)
    R_cam = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    
    
    # camera intrinsics
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    width, height = image_size
    
    # 4 image corners (pixel coordinates)
    corners_2d = np.array([
        [0, 0],           # top-left
        [width, 0],       # top-right
        [width, height],  # bottom-right
        [0, height]       # bottom-left
    ])
    
    # frustum vertices at each distance
    frustum_vertices = []
    
    # camera position (near)
    frustum_vertices.append(camera_position)
    
    # far frustum vertices
    for i, corner_2d in enumerate(corners_2d):
        u, v = corner_2d
        
        # 3D direction in camera coordinates (Z points forward)
        x_cam = (u - cx) * max_distance / fx
        y_cam = (v - cy) * max_distance / fy
        z_cam = max_distance
        
        # to world coordinates
        point_cam = np.array([x_cam, y_cam, z_cam])
        point_world = camera_position + R_cam @ point_cam
        
        frustum_vertices.append(point_world)
        
    
    return np.array(frustum_vertices)


def apply_fake_depth_to_mask(depths, mask, fake_value):
    """
    Apply fake depth (0) to depth values where mask=1
    
    Args:
        depths: numpy array of shape [t, h, w] with depth values
        mask: numpy array of shape [t, h, w] with binary values (0 or 1)
    
    Returns:
        Modified depths with 0 values in masked regions
    """    
    # Create a copy to avoid modifying original data
    modified_depths = depths.copy()
    
    # Apply 0 depth to masked regions
    modified_depths = np.where(mask == 1, fake_value, modified_depths)
    
    return modified_depths


def rgbd_to_point_cloud(rgb_frame, depth_frame, camera_intrinsics, max_depth=5.0, depth_scale=1.0):
    """
    Convert an RGB-D frame to a point cloud
    
    Args:
        rgb_frame: RGB image (H, W, 3)
        depth_frame: depth image (H, W)
        camera_intrinsics: camera intrinsics (3, 3)
        max_depth: maximum depth (larger values are discarded)
        depth_scale: depth scale factor
    
    Returns:
        o3d.geometry.PointCloud
    """
    
    H, W = depth_frame.shape
    
    # keep valid depth only (0 < depth < max_depth)
    valid_mask = (depth_frame > 0) & (depth_frame < max_depth)
    
    if not np.any(valid_mask):
        print("Warning: No valid depth values found")
        return o3d.geometry.PointCloud()
    
    # pixel coordinate grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # keep valid pixels only
    valid_u = u[valid_mask]
    valid_v = v[valid_mask]
    valid_depth = depth_frame[valid_mask] * depth_scale
    
    # camera intrinsics
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    
    # 3D coordinates
    x = (valid_u - cx) * valid_depth / fx
    y = (valid_v - cy) * valid_depth / fy
    z = valid_depth
    
    # stack 3D points
    points_3d = np.stack([x, y, z], axis=1)
    
    # RGB colors (valid pixels only)
    colors = rgb_frame[valid_mask] / 255.0  # normalize to [0, 1]
    
    # create the Open3D PointCloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    
    return point_cloud


def batch_raycasting_with_scene(scene, origins, directions, max_distances):
    """
    Batch raycasting with a precomputed Open3D RaycastingScene
    
    Args:
        scene: precomputed o3d.t.geometry.RaycastingScene
        origins: ray origins [M, 3]
        directions: ray directions (normalized) [M, 3]
        max_distances: maximum distances [M]
    
    Returns:
        (visible_array, hit_distances_array)
        - visible_array: [M] boolean array, True if visible
        - hit_distances_array: [M] float array, hit distances
    """
    if scene is None:
        print("    Scene is None, assuming all visible")
        return np.ones(len(origins), dtype=bool), max_distances.copy()
    
    total_start_time = time.time()
    
    # 1. build the batch of rays
    step1_start = time.time()
    rays = np.hstack([origins, directions]) # [M*N, 3+3]
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    step1_time = time.time() - step1_start
    vprint(f"      Step 1 - Ray preparation: {step1_time:.4f}s")
    
    # 2. batch raycasting
    step2_start = time.time()
    ans = scene.cast_rays(rays_tensor)
    step2_time = (time.time() - step2_start)
    vprint(f"      Step 2 - Ray casting: {step2_time:.4f}s")
    
    # 3. analyze the results
    step3_start = time.time()
    hit_distances = ans['t_hit'].numpy()  # [M]
    
    # visible if the hit distance is inf or larger than max_distance
    visible = np.logical_or(np.isinf(hit_distances), hit_distances > max_distances)
    
    
    step3_time = time.time() - step3_start
    vprint(f"      Step 3 - Result processing: {step3_time:.4f}s")
    
    total_time = time.time() - total_start_time
    vprint(f"    Batch raycasting ({len(origins)} rays): {total_time:.4f}s total")
    
    # timing summary per stage
    vprint(f"      Time breakdown: rays={step1_time:.4f}s, casting={step2_time:.4f}s, processing={step3_time:.4f}s")
    
    return visible, hit_distances


from nvblox_torch.examples.utils.visualization import Visualizer
class CustomVisualizer(Visualizer):
        """
        Custom visualizer that combines robot mesh with scene mesh in _visualize_nvblox_mesh
        and adds Line of Sight (LOS) visualization functionality
        
        Camera Control:
        - follow_camera_view=False: Static camera view (uses initial camera pose)
        - follow_camera_view=True: Dynamic camera view (follows each frame's camera pose)
        - active_camera=False: Use current camera pose for visualization
        - active_camera=True: Use active_camera_pose_list for multiple camera frustum visualization
        - video_save: Enable/disable video recording
        """
        def __init__(self, deep_feature_embedding_dim=None, video_save=False, result_save_path=None, follow_camera_view=False, active_camera=False, vis_window=True):
            super().__init__(deep_feature_embedding_dim)
            self.los_geometries = {}  # Store LOS geometries for each visualizer
            self.camera_frustum_geometries = {}  # Store camera frustum geometries for each visualizer
            self.active_camera_frustum_geometries = {}  # Store active camera frustum geometries for each visualizer
            self.retargeted_active_camera_frustum_geometries = {}  # Store retargeted active camera frustum geometries for each visualizer
            self.current_camera_frustum_geometries = {}  # Store current camera pose frustum geometries (white color)
            self.query_points = None
            self.camera_center = None
            self.camera_intrinsics = None
            self.image_size = None
            self.pending_camera_updates = {}  # Store pending camera updates for each visualizer
            
            # Video recording settings
            self.video_save = video_save
            self.result_save_path = result_save_path  # Store result_save_path for generating video_path in save_video
            self.captured_frames_mesh = []  # Store captured frames for mesh video
            self.captured_frames_point_cloud = []  # Store captured frames for point cloud video
            self.initial_camera_pose = None  # Store initial camera pose
            
            # Camera following settings
            self.follow_camera_view = follow_camera_view
            self.current_camera_pose = None  # Store current camera pose for following
            self.visualizers = {}
            
            # Active camera settings
            self.active_camera = active_camera
            self.active_camera_pose_list = None  # Store list of active camera poses
            self.retargeted_active_camera_pose_list = None  # Store list of retargeted active camera poses
            
            # Visualization window settings
            self.vis_window = vis_window  # True: show real-time windows, False: hide windows
            self.rendering_available = None  # checked on the first visualize() call
            
            # Visibility info for frame annotation
            # Structure: {step: {cam_idx: {'active': info_dict, 'retargeted': info_dict}}}
            self.current_visibility_info = {}
            self.current_step = None
            
        
        def _create_visualizer(self, window_name: str):
            """Create visualizer with custom settings for LOS visualization"""
            
            visualizer = o3d.visualization.VisualizerWithKeyCallback()
            
            if self.vis_window:
                # show the window on screen
                visualizer.create_window(width=800, height=600, window_name=window_name)
            else:
                # hide the window (offscreen rendering)
                visualizer.create_window(width=800, height=600, window_name=window_name, visible=False)
            
            visualizer.get_render_option().line_width = 5  # thicker line-of-sight lines
            visualizer.get_render_option().point_size = 3
            visualizer.get_render_option().background_color = np.asarray([0, 0, 0])
            visualizer.register_key_callback(ord(' '), lambda vis: self._toggle_pause(vis))
            return visualizer
        
        def _loop_while_paused(self):
            """Override to capture camera position continuously while paused"""
            
            while self.pause:
                for visualizer_name, visualizer in self.visualizers.items():
                    visualizer.poll_events()
                    visualizer.update_renderer()
                    
                    # Continuously capture camera position while paused
                    view_control = visualizer.get_view_control()
                    camera_params = view_control.convert_to_pinhole_camera_parameters()
                    view_status = visualizer.get_view_status()
                    
                    # Store camera update for later application
                    self.pending_camera_updates[visualizer_name] = {
                        'camera_params': camera_params,
                        'view_status': view_status,
                    }
                    
                
                time.sleep(0.001)
        
        def set_initial_camera_pose(self, camera_pose):
            """Set the initial camera pose for the visualizer"""
            self.initial_camera_pose = camera_pose.copy()
        
        def set_current_camera_pose(self, camera_pose):
            """Set the current camera pose for following mode"""
            self.current_camera_pose = camera_pose.cpu().numpy().copy()
        
        def _set_camera_pose_from_matrix(self, visualizer, camera_pose_matrix):
            """Set camera pose from 4x4 transformation matrix"""
            
            
            # Convert torch tensor to numpy if needed
            if isinstance(camera_pose_matrix, torch.Tensor):
                camera_pose_matrix = camera_pose_matrix.cpu().numpy()
            
            # Extract camera position and orientation
            camera_position = camera_pose_matrix[:3, 3]
            camera_rotation = camera_pose_matrix[:3, :3]
            
            # Calculate lookat point (camera position + front direction)
            front_direction = -camera_rotation[:, 2]  # Negative Z axis is front
            lookat_point = camera_position + front_direction * 0.0  # Look 2 units ahead
            
            # Set camera parameters
            view_control = visualizer.get_view_control()
            view_control.set_lookat(lookat_point)
            view_control.set_up(-camera_rotation[:, 1])  # Y axis is up
            view_control.set_front(front_direction)
            view_control.set_zoom(0.2)
            
            
            # Move camera to position
        
        def _capture_frame(self, visualizer, visualizer_name):
            """Capture current frame for video recording"""
            if self.video_save:
            
                # Capture screen image
                image = visualizer.capture_screen_float_buffer(do_render=True)
                # Convert to numpy array and scale to 0-255
                image_np = np.asarray(image)
                image_np = (image_np * 255).astype(np.uint8)
                
                # Add visibility info text to frame if available
                # Display all cam_idx information for current step
                if self.current_step is not None:
                    if self.current_step in self.current_visibility_info:
                        step_info = self.current_visibility_info[self.current_step]
                        # Get all cam_idx information for this step
                        all_cam_info = {}
                        for cam_idx, cam_info in step_info.items():
                            all_cam_info[cam_idx] = {
                                'active': cam_info.get('active'),
                                'retargeted': cam_info.get('retargeted')
                            }
                        image_np = self._add_visibility_info_text(image_np, all_cam_info)
                
                # Store frames based on visualizer type
                if 'color_mesh' == visualizer_name:
                    self.captured_frames_mesh.append(image_np)
                elif 'point_cloud' == visualizer_name:
                    self.captured_frames_point_cloud.append(image_np)
                else:
                    raise ValueError(f"Invalid visualizer name: {visualizer_name}")
        
        def _add_visibility_info_text(self, frame, all_cam_info):
            """Add visibility information as text overlay on frame for all cam_idx
            
            Args:
                frame: Input frame (RGB)
                all_cam_info: Dict of {cam_idx: {'active': info_dict, 'retargeted': info_dict}}
            """
            import cv2
            
            # Convert RGB to BGR for cv2
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            thickness = 1
            color = (255, 255, 255)  # White text
            bg_color = (0, 0, 0)  # Black background
            line_height = 13
            section_spacing = 3  # Spacing between different cam_idx sections
            
            # Helper function to get text lines from visibility info
            def get_text_lines(visibility_info, prefix=""):
                if visibility_info is None:
                    return []
                
                text_lines = []
                batch_visible = visibility_info.get('batch_visible')
                frustum_visible = visibility_info.get('frustum_visible')
                mask_filtered_point_indices = visibility_info.get('mask_filtered_point_indices')
                hit_distances = visibility_info.get('hit_distances')
                
                # Display actual values
                if batch_visible is not None:
                    # Convert boolean array to string representation
                    batch_str = np.array2string(batch_visible, separator=',', max_line_width=100, threshold=20)
                    text_lines.append(f"{prefix}Batch: {batch_str}")
                
                if frustum_visible is not None:
                    # Convert boolean array to string representation
                    frustum_str = np.array2string(frustum_visible, separator=',', max_line_width=100, threshold=20)
                    text_lines.append(f"{prefix}Frustum: {frustum_str}")
                
                if mask_filtered_point_indices is not None and len(mask_filtered_point_indices) > 0:
                    # Display actual indices
                    mask_str = np.array2string(mask_filtered_point_indices, separator=',', max_line_width=100, threshold=20)
                    text_lines.append(f"{prefix}Mask: {mask_str}")
                
                if hit_distances is not None:
                    # Display actual hit distances
                    hit_str = np.array2string(hit_distances, separator=',', precision=3, max_line_width=100, threshold=20)
                    text_lines.append(f"{prefix}Hit: {hit_str}")
                
                return text_lines
            
            # Sort cam_idx for consistent display order
            sorted_cam_indices = sorted(all_cam_info.keys())
            
            # Draw active info on left side (top-left corner)
            y_offset_left = 10
            for cam_idx in sorted_cam_indices:
                cam_info = all_cam_info[cam_idx]
                active_info = cam_info.get('active')
                
                if active_info is not None:
                    # Add cam_idx header
                    header_text = f"Cam {cam_idx} [Active]:"
                    (header_width, header_height), baseline = cv2.getTextSize(header_text, font, font_scale, thickness)
                    cv2.rectangle(frame_bgr, (5, y_offset_left - header_height - 2), (5 + header_width + 2, y_offset_left + baseline), bg_color, -1)
                    cv2.putText(frame_bgr, header_text, (5, y_offset_left), font, font_scale, color, thickness, cv2.LINE_AA)
                    y_offset_left += line_height
                    
                    # Get and draw active info lines
                    active_lines = get_text_lines(active_info, prefix="  ")
                    for i, text in enumerate(active_lines):
                        y_pos = y_offset_left + i * line_height
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                        cv2.rectangle(frame_bgr, (5, y_pos - text_height - 2), (5 + text_width + 2, y_pos + baseline), bg_color, -1)
                        cv2.putText(frame_bgr, text, (5, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)
                    
                    y_offset_left += len(active_lines) * line_height + section_spacing
            
            # Draw retargeted info on right side (top-right corner)
            y_offset_right = 10
            frame_width = frame_bgr.shape[1]
            for cam_idx in sorted_cam_indices:
                cam_info = all_cam_info[cam_idx]
                retargeted_info = cam_info.get('retargeted')
                
                if retargeted_info is not None:
                    # Add cam_idx header
                    header_text = f"Cam {cam_idx} [Retargeted]:"
                    (header_width, header_height), baseline = cv2.getTextSize(header_text, font, font_scale, thickness)
                    x_pos = frame_width - header_width - 5
                    cv2.rectangle(frame_bgr, (x_pos - 2, y_offset_right - header_height - 2), (x_pos + header_width + 2, y_offset_right + baseline), bg_color, -1)
                    cv2.putText(frame_bgr, header_text, (x_pos, y_offset_right), font, font_scale, color, thickness, cv2.LINE_AA)
                    y_offset_right += line_height
                    
                    # Get and draw retargeted info lines
                    retargeted_lines = get_text_lines(retargeted_info, prefix="  ")
                    for i, text in enumerate(retargeted_lines):
                        y_pos = y_offset_right + i * line_height
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                        x_pos = frame_width - text_width - 5
                        cv2.rectangle(frame_bgr, (x_pos - 2, y_pos - text_height - 2), (x_pos + text_width + 2, y_pos + baseline), bg_color, -1)
                        cv2.putText(frame_bgr, text, (x_pos, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)
                    
                    y_offset_right += len(retargeted_lines) * line_height + section_spacing
            
            # Convert back to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return frame_rgb
        
        def _update_visualization(self, visualizer: o3d.visualization.VisualizerWithKeyCallback, visualizer_name: str) -> None:
            visualizer.poll_events()
            visualizer.update_renderer()
            
            # Capture frame for video if enabled
            self._capture_frame(visualizer, visualizer_name)
            if visualizer_name in self.pending_camera_updates:
                view_control = visualizer.get_view_control()
                camera_params = view_control.convert_to_pinhole_camera_parameters()
                view_status = visualizer.get_view_status()
                
                # Store camera update for later application
                self.pending_camera_updates[visualizer_name] = {
                    'camera_params': camera_params,
                    'view_status': view_status,
                }
                time.sleep(0.001)


        def _visualize_nvblox_mesh(self, color_mesh, name='color_mesh'):
            if name not in self.visualizers:
                self.visualizers[name] = self._create_visualizer(name)
            self.visualizers[name].clear_geometries()
            self.visualizers[name].add_geometry(color_mesh)
            self.visualizers[name].update_renderer()
        
        def _visualize_nvblox_point_cloud(self, point_cloud, name='point_cloud'):
            """
            Visualize an nvblox point cloud
            
            Args:
                point_cloud: point cloud to visualize (numpy array or Open3D PointCloud)
                name: visualizer name (default: 'point_cloud')
            """
            
            # create the visualizer if needed
            if name not in self.visualizers:
                self.visualizers[name] = self._create_visualizer(name)
            
            # remove the previous geometry
            self.visualizers[name].clear_geometries()
            
            # convert a numpy point cloud to an Open3D PointCloud
            if isinstance(point_cloud, np.ndarray):
                o3d_point_cloud = o3d.geometry.PointCloud()
                o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud)
                
                # default color if none is given
                if len(point_cloud) > 0:
                    colors = np.ones((len(point_cloud), 3)) * 0.8  # light gray
                    o3d_point_cloud.colors = o3d.utility.Vector3dVector(colors)
                
                point_cloud = o3d_point_cloud
            
            # add the point cloud to the visualizer
            self.visualizers[name].add_geometry(point_cloud)
            self.visualizers[name].update_renderer()
            
            
        def create_multiple_lines_of_sight(self, camera_center, target_points, colors=None):
            """
            Create lines of sight from the camera center to several target points
            
            Args:
                camera_center: camera center
                target_points: list of target points
                colors: color of each line (default color if None)
            
            Returns:
                o3d.geometry.LineSet containing all lines of sight
            """
            
            
            if colors is None:
                colors = [[1, 0, 0]] * len(target_points)  # red by default
            
            points = [camera_center]
            lines = []
            line_colors = []
            
            for i, target_point in enumerate(target_points):
                points.append(target_point)
                lines.append([0, i + 1])  # from the camera center (0) to each target point
                line_colors.append(colors[i])
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.array(points))
            line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
            line_set.colors = o3d.utility.Vector3dVector(np.array(line_colors))
            
            return line_set
        
        def create_camera_frustum_geometry(self, camera_pose, camera_intrinsics, image_size, max_distance=0.1, color=None):
            """
            Create a camera frustum as Open3D geometry
            
            Args:
                camera_pose: camera pose (4x4 transform, torch.Tensor or numpy.ndarray)
                camera_intrinsics: camera intrinsics (3x3)
                image_size: image size (width, height)
                max_distance: frustum depth (default: 0.1m)
                color: frustum color [R, G, B] (default color if None)
            
            Returns:
                o3d.geometry.LineSet representing the camera frustum
            """
            
            
            # camera position and rotation (supports torch.Tensor and numpy.ndarray)
            if isinstance(camera_pose, torch.Tensor):
                camera_position = camera_pose[:3, 3].cpu().numpy()
                camera_rotation_matrix = camera_pose[:3, :3].cpu().numpy()
            else:
                camera_position = camera_pose[:3, 3]
                camera_rotation_matrix = camera_pose[:3, :3]
            camera_rotation_rpy = R.from_matrix(camera_rotation_matrix).as_euler('xyz', degrees=True)
            
            # frustum vertices
            frustum_vertices = create_camera_frustum_visualization(
                camera_position, camera_rotation_rpy, camera_intrinsics, image_size, max_distance
            )
            
            # frustum as a LineSet
            # vertices: 0 is the camera center, 1-4 are the far corners
            points = frustum_vertices
            lines = [
                # from the camera center to each far corner
                [0, 1], [0, 2], [0, 3], [0, 4],
                # connect the far corners (rectangle)
                [1, 2], [2, 3], [3, 4], [4, 1]
            ]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
            
            # color (cyan by default, or the given color)
            if color is None:
                color = [0, 1, 1]  # cyan (default)
            line_set.colors = o3d.utility.Vector3dVector(np.array([color] * len(lines)))
            
            return line_set
        
        def _visualize_camera_pose(self, camera_pose):
            """
            Visualize the camera frustum (overrides _visualize_camera_pose)
            """
            if self.camera_intrinsics is None or self.image_size is None:
                print("Warning: Camera intrinsics or image size not set, skipping camera frustum visualization")
                return
            
            # add the camera frustum to every visualizer
            for name, visualizer in self.visualizers.items():
                # remove the previous camera frustum geometry
                if name in self.camera_frustum_geometries:
                    visualizer.remove_geometry(self.camera_frustum_geometries[name])
                
                # create and add the new camera frustum
                frustum_geometry = self.create_camera_frustum_geometry(
                    camera_pose, self.camera_intrinsics, self.image_size
                )
                self.camera_frustum_geometries[name] = frustum_geometry
                visualizer.add_geometry(frustum_geometry)
                
                
        def _visualize_current_camera_poses(self, current_camera_pose_list, selected_indices=None):
            """
            Visualize only the selected indices of current_camera_pose_list
            """
            if self.camera_intrinsics is None or self.image_size is None:
                print("Warning: Camera intrinsics or image size not set, skipping current camera frustum visualization")
                return
            
            if current_camera_pose_list is None or len(current_camera_pose_list) == 0:
                print("Warning: No current camera poses provided")
                return
            
            # use all indices if none are selected
            if selected_indices is None:
                selected_indices = list(range(len(current_camera_pose_list)))
            
            # add only the selected current camera frustums to every visualizer
            for name, visualizer in self.visualizers.items():
                # remove the previous current camera frustum geometries
                if name in self.camera_frustum_geometries:
                    for geometry in self.camera_frustum_geometries[name]:
                        visualizer.remove_geometry(geometry)
                    self.camera_frustum_geometries[name] = []
                
                # create and add frustums only for the selected current camera poses
                current_frustum_geometries = []
                num_selected = len(selected_indices)
                # Base color: cyan [0, 1, 1]
                base_color_cyan = np.array([0.0, 1.0, 1.0])
                for idx, i in enumerate(selected_indices):
                    if i < len(current_camera_pose_list):
                        camera_pose = current_camera_pose_list[i]
                        # shift the color slightly away from cyan along the horizon
                        # normalize idx to [0, 1]
                        if num_selected > 1:
                            normalized_idx = idx / (num_selected - 1)
                        else:
                            normalized_idx = 0.0
                        # small change from cyan (adjust the R and B channels slightly)
                        # normalized_idx == 0: [0, 1, 1] (cyan)
                        # normalized_idx == 1: [0.3, 1, 0.7] (slightly less blue)
                        color = base_color_cyan.copy()
                        color[0] = 0.0 + 0.3 * normalized_idx  # R: 0 → 0.3
                        color[2] = 1.0 - 0.3 * normalized_idx  # B: 1.0 → 0.7
                        color = list(color)
                        
                        frustum_geometry = self.create_camera_frustum_geometry(
                            camera_pose, self.camera_intrinsics, self.image_size, color=color
                        )
                        current_frustum_geometries.append(frustum_geometry)
                        visualizer.add_geometry(frustum_geometry)
                
                self.camera_frustum_geometries[name] = current_frustum_geometries
                
        def _visualize_active_camera_poses(self, active_camera_pose_list, selected_indices=None):
            """
            Visualize only the selected indices of active_camera_pose_list
            """
            if self.camera_intrinsics is None or self.image_size is None:
                print("Warning: Camera intrinsics or image size not set, skipping active camera frustum visualization")
                return
            
            if active_camera_pose_list is None or len(active_camera_pose_list) == 0:
                print("Warning: No active camera poses provided")
                return
            
            # use all indices if none are selected
            if selected_indices is None:
                selected_indices = list(range(len(active_camera_pose_list)))
            
            # add only the selected active camera frustums to every visualizer
            for name, visualizer in self.visualizers.items():
                # remove the previous active camera frustum geometries
                if name in self.active_camera_frustum_geometries:
                    for geometry in self.active_camera_frustum_geometries[name]:
                        visualizer.remove_geometry(geometry)
                    self.active_camera_frustum_geometries[name] = []
                
                # create and add frustums only for the selected active camera poses
                active_frustum_geometries = []
                num_selected = len(selected_indices)
                # Base color: yellow [1, 1, 0]
                base_color_yellow = np.array([1.0, 1.0, 0.0])
                for idx, i in enumerate(selected_indices):
                    if i < len(active_camera_pose_list):
                        camera_pose = active_camera_pose_list[i]
                        # shift the color slightly away from yellow along the horizon
                        # normalize idx to [0, 1]
                        if num_selected > 1:
                            normalized_idx = idx / (num_selected - 1)
                        else:
                            normalized_idx = 0.0
                        # small change from yellow (adjust the R and G channels slightly)
                        # normalized_idx == 0: [1, 1, 0] (yellow)
                        # normalized_idx == 1: [0.7, 1, 0.3] (slightly orange)
                        color = base_color_yellow.copy()
                        color[0] = 1.0 - 0.3 * normalized_idx  # R: 1.0 → 0.7
                        color[2] = 0.0 + 0.3 * normalized_idx  # B: 0 → 0.3
                        color = list(color)
                        
                        frustum_geometry = self.create_camera_frustum_geometry(
                            camera_pose, self.camera_intrinsics, self.image_size, color=color
                        )
                        active_frustum_geometries.append(frustum_geometry)
                        visualizer.add_geometry(frustum_geometry)
                
                self.active_camera_frustum_geometries[name] = active_frustum_geometries
        
        def _visualize_retargeted_active_camera_poses(self, retargeted_active_camera_pose_list, selected_indices=None):
            """
            Visualize only the selected indices of retargeted_active_camera_pose_list
            Same logic as the active camera, only the color differs (magenta)
            """
            if self.camera_intrinsics is None or self.image_size is None:
                print("Warning: Camera intrinsics or image size not set, skipping retargeted active camera frustum visualization")
                return
            
            if retargeted_active_camera_pose_list is None or len(retargeted_active_camera_pose_list) == 0:
                print("Warning: No retargeted active camera poses provided")
                return
            
            # use all indices if none are selected
            if selected_indices is None:
                selected_indices = list(range(len(retargeted_active_camera_pose_list)))
            
            # add only the selected retargeted active camera frustums to every visualizer
            for name, visualizer in self.visualizers.items():
                # remove the previous retargeted active camera frustum geometries
                if name in self.retargeted_active_camera_frustum_geometries:
                    for geometry in self.retargeted_active_camera_frustum_geometries[name]:
                        visualizer.remove_geometry(geometry)
                    self.retargeted_active_camera_frustum_geometries[name] = []
                
                # create and add frustums only for the selected retargeted active camera poses
                retargeted_active_frustum_geometries = []
                num_selected = len(selected_indices)
                # Base color: magenta [1, 0, 1]
                base_color_magenta = np.array([1.0, 0.0, 1.0])
                for idx, i in enumerate(selected_indices):
                    if i < len(retargeted_active_camera_pose_list):
                        camera_pose = retargeted_active_camera_pose_list[i]
                        # shift the color slightly away from magenta along the horizon
                        # normalize idx to [0, 1]
                        if num_selected > 1:
                            normalized_idx = idx / (num_selected - 1)
                        else:
                            normalized_idx = 0.0
                        # small change from magenta (adjust the R and B channels slightly)
                        # normalized_idx == 0: [1, 0, 1] (magenta)
                        # normalized_idx == 1: [0.7, 0, 1] (slightly more blue)
                        color = base_color_magenta.copy()
                        color[0] = 1.0 - 0.3 * normalized_idx  # R: 1.0 → 0.7
                        color[1] = 0.0  # G: 0.0 (fixed)
                        color[2] = 1.0  # B: 1.0 (fixed)
                        color = list(color)
                        
                        frustum_geometry = self.create_camera_frustum_geometry(
                            camera_pose, self.camera_intrinsics, self.image_size, color=color
                        )
                        retargeted_active_frustum_geometries.append(frustum_geometry)
                        visualizer.add_geometry(frustum_geometry)
                
                self.retargeted_active_camera_frustum_geometries[name] = retargeted_active_frustum_geometries
                
        
        def visualize(self, color_mesh=None, feature_mesh=None, point_cloud=None,camera_pose=None, 
                     query_points=None,  camera_intrinsics=None, image_size=None,
                     current_camera_pose_list=None, current_visibility_results_dict=None,
                     active_camera_pose_list=None, active_visibility_results_dict=None,
                     retargeted_active_camera_pose_list=None, retargeted_active_visibility_results_dict=None,
                     active_visibility_results_info=None, retargeted_active_visibility_results_info=None,
                     selected_indices=None, query_point_indices=None, step = None):
            """
            Visualization including lines of sight (overrides visualize)
            
            Args:
                color_mesh: Color mesh to visualize
                feature_mesh: Feature mesh to visualize  
                camera_pose: Camera pose to visualize
                query_points: Query points for LOS visualization (optional)
                camera_intrinsics: Camera intrinsics for frustum visualization (optional)
                image_size: Image size for frustum visualization (optional)
                current_camera_pose_list: List of current camera poses (optional)
                current_visibility_results_dict: Dictionary of visibility results for multiple current cameras (optional)
                active_camera_pose_list: List of active camera poses for multiple frustum visualization (optional)
                active_visibility_results_dict: Dictionary of visibility results for multiple active cameras (optional)
                retargeted_active_camera_pose_list: List of retargeted active camera poses for multiple frustum visualization (optional)
                retargeted_active_visibility_results_dict: Dictionary of visibility results for multiple retargeted active cameras (optional)
                selected_indices: List of selected indices for visualization (optional)
            """
            # Open3D windows (even hidden ones) need an X display
            if self.rendering_available is None:
                probe = o3d.visualization.Visualizer()
                self.rendering_available = probe.create_window(visible=False)
                probe.destroy_window()
                if not self.rendering_available:
                    print("Warning: Open3D could not create a window (no display), skipping the 3D visualization videos. "
                          "Run under a virtual display (e.g. `xvfb-run -a`) to record them.")
            if not self.rendering_available:
                return

            # Store camera intrinsics and image size for frustum visualization
            if camera_intrinsics is not None:
                self.camera_intrinsics = camera_intrinsics
            if image_size is not None:
                self.image_size = image_size
            
            # Store active camera pose list
            if active_camera_pose_list is not None:
                self.active_camera_pose_list = active_camera_pose_list
            
            # Store retargeted active camera pose list
            if retargeted_active_camera_pose_list is not None:
                self.retargeted_active_camera_pose_list = retargeted_active_camera_pose_list
            
            # Update current camera pose if following mode is enabled
            if self.follow_camera_view and camera_pose is not None:
                self.set_current_camera_pose(camera_pose)

            # Store current views before updating

            if color_mesh is not None:
                self._visualize_nvblox_mesh(color_mesh)

            if point_cloud is not None:
                self._visualize_nvblox_point_cloud(point_cloud)

            
            # add LOS visualization (when query_points and visibility_results are given)
            if query_points is not None:
                # Handle both [N, 3 or 4] and [horizon, N, 3 or 4] shapes
                query_points_is_3d = query_points.ndim == 3
                if query_points_is_3d:
                    # [horizon, N, 3 or 4] - predicted flow case
                    # Visualize all horizons with different colors
                    num_horizons = query_points.shape[0]
                    if query_points.shape[-1] == 4:  # if it includes visibility
                        query_points_xyz = query_points[..., :3].copy()  # [horizon, N, 3]
                    else:
                        query_points_xyz = query_points.copy()  # [horizon, N, 3]
                elif query_points.ndim == 2:
                    # [N, 3 or 4] - original case
                    num_horizons = 1
                    if query_points.shape[-1] == 4:  # if it includes visibility
                        query_points_xyz = query_points[..., :3].copy()  # [N, 3]
                    else:
                        query_points_xyz = query_points.copy()  # [N, 3]
                    # Add horizon dimension for consistent processing
                    query_points_xyz = query_points_xyz[None, :, :].copy()  # [1, N, 3]
                else:
                    raise ValueError(f"query_points must be 2D or 3D array, got {query_points.shape}")

                # LOS of the current camera (selected indices only)
                if current_camera_pose_list is not None and current_visibility_results_dict is not None and selected_indices is not None:
                    for name, visualizer in self.visualizers.items():
                        for cam_idx, cam_visibility_results in current_visibility_results_dict.items():
                            if cam_idx in selected_indices and cam_idx < len(current_camera_pose_list):
                                current_camera_pose = current_camera_pose_list[cam_idx]
                                current_camera_center = current_camera_pose[:3, 3].cpu().numpy() if isinstance(current_camera_pose, torch.Tensor) else current_camera_pose[:3, 3]
                                
                                # LOS only for the query points of cam_idx
                                # cam_idx comes from selected_indices, so only the query points of that horizon step are used
                                if cam_idx < num_horizons:
                                    # remove the previous current-camera LOS geometry
                                    current_los_key = f"{name}_current_los_{cam_idx}_h{cam_idx}"
                                    if current_los_key in self.los_geometries:
                                        visualizer.remove_geometry(self.los_geometries[current_los_key])
                                    
                                    # use only the query points of cam_idx
                                    query_points_h = query_points_xyz[cam_idx].copy()  # [N, 3]
                                    
                                    # filter by query_point_indices if given
                                    if query_point_indices is not None:
                                        query_points_h = query_points_h[query_point_indices].copy()  # [M, 3]
                                        cam_visibility_results = cam_visibility_results[query_point_indices].copy()  # [M]
                                    
                                    # brightness along the horizon
                                    # normalize by the position of cam_idx in selected_indices
                                    if selected_indices is not None and len(selected_indices) > 1:
                                        # handle numpy arrays
                                        selected_indices_list = selected_indices.tolist() if isinstance(selected_indices, np.ndarray) else list(selected_indices)
                                        cam_idx_position = selected_indices_list.index(cam_idx) if cam_idx in selected_indices_list else 0
                                        normalized_horizon = cam_idx_position / (len(selected_indices_list) - 1)
                                    else:
                                        normalized_horizon = 0.0
                                    
                                    # brightness decreases along the horizon (1.0 ~ 0.3)
                                    # darker towards the last horizon step
                                    brightness = 1.0 - 0.7 * normalized_horizon  # 1.0 (bright, first step) ~ 0.3 (dark, last step)
                                    
                                    # color by visibility: green if visible, red if occluded
                                    # brightness depends on the horizon step
                                    colors = []
                                    for visible in cam_visibility_results:
                                        if visible:
                                            # green with brightness
                                            colors.append([0, brightness, 0])  # green (visible)
                                        else:
                                            # red with brightness
                                            colors.append([brightness, 0, 0])  # red (occluded)
                                    
                                    # create and add the LOS
                                    line_set = self.create_multiple_lines_of_sight(current_camera_center, query_points_h, colors)
                                    current_los_key = f"{name}_current_los_{cam_idx}_h{cam_idx}"
                                    self.los_geometries[current_los_key] = line_set
                                    visualizer.add_geometry(line_set)
                                
                
                # in active_camera mode, LOS for the selected active cameras
                # Collect query_points_h for comparison
                active_query_points_dict = {}  # {cam_idx: query_points_h}
                if self.active_camera and self.active_camera_pose_list is not None and len(self.active_camera_pose_list) > 0 and active_visibility_results_dict is not None and selected_indices is not None:
                    # LOS for each selected active camera
                    for name, visualizer in self.visualizers.items():
                        for cam_idx, cam_visibility_results in active_visibility_results_dict.items():
                            if cam_idx in selected_indices and cam_idx < len(self.active_camera_pose_list):
                                active_camera_pose = self.active_camera_pose_list[cam_idx]
                                active_camera_center = active_camera_pose[:3, 3].cpu().numpy() if isinstance(active_camera_pose, torch.Tensor) else active_camera_pose[:3, 3]
                                
                                # LOS only for the query points of cam_idx
                                # cam_idx comes from selected_indices, so only the query points of that horizon step are used
                                if cam_idx < num_horizons:
                                    # remove the active camera LOS geometry
                                    active_los_key = f"{name}_active_los_{cam_idx}_h{cam_idx}"
                                    if active_los_key in self.los_geometries:
                                        visualizer.remove_geometry(self.los_geometries[active_los_key])
                                    
                                    # use only the query points of cam_idx
                                    query_points_h = query_points_xyz[cam_idx].copy()  # [N, 3]
                                    
                                    # filter by query_point_indices if given
                                    if query_point_indices is not None:
                                        query_points_h = query_points_h[query_point_indices].copy()  # [M, 3]
                                        cam_visibility_results = cam_visibility_results[query_point_indices].copy()  # [M]
                                    
                                    # Store query_points_h for comparison
                                    active_query_points_dict[cam_idx] = query_points_h.copy()
                                    
                                    # brightness along the horizon
                                    # normalize by the position of cam_idx in selected_indices
                                    if selected_indices is not None and len(selected_indices) > 1:
                                        # handle numpy arrays
                                        selected_indices_list = selected_indices.tolist() if isinstance(selected_indices, np.ndarray) else list(selected_indices)
                                        cam_idx_position = selected_indices_list.index(cam_idx) if cam_idx in selected_indices_list else 0
                                        normalized_horizon = cam_idx_position / (len(selected_indices_list) - 1)
                                    else:
                                        normalized_horizon = 0.0
                                    
                                    # brightness decreases along the horizon (1.0 ~ 0.3)
                                    # darker towards the last horizon step
                                    brightness = 1.0 - 0.7 * normalized_horizon  # 1.0 (bright, first step) ~ 0.3 (dark, last step)
                                    
                                    # active camera color by visibility: green if visible, red if occluded
                                    # brightness depends on the horizon step
                                    active_colors = []
                                    for visible in cam_visibility_results:
                                        if visible:
                                            # green with brightness
                                            active_colors.append([0, brightness, 0])  # green (visible)
                                        else:
                                            # red with brightness
                                            active_colors.append([brightness, 0, 0])  # red (occluded)
                                    
                                    # create and add the active camera LOS
                                    active_line_set = self.create_multiple_lines_of_sight(active_camera_center, query_points_h, active_colors)
                                    active_los_key = f"{name}_active_los_{cam_idx}_h{cam_idx}"
                                    self.los_geometries[active_los_key] = active_line_set
                                    visualizer.add_geometry(active_line_set)
                
                # in retargeted_active_camera mode, LOS for the selected retargeted active cameras
                # Collect query_points_h for comparison
                retargeted_query_points_dict = {}  # {cam_idx: query_points_h}
                if self.active_camera and self.retargeted_active_camera_pose_list is not None and len(self.retargeted_active_camera_pose_list) > 0 and retargeted_active_visibility_results_dict is not None and selected_indices is not None:
                    # LOS for each selected retargeted active camera
                    for name, visualizer in self.visualizers.items():
                        for cam_idx, cam_visibility_results in retargeted_active_visibility_results_dict.items():
                            if cam_idx in selected_indices and cam_idx < len(self.retargeted_active_camera_pose_list):
                                retargeted_active_camera_pose = self.retargeted_active_camera_pose_list[cam_idx]
                                retargeted_active_camera_center = retargeted_active_camera_pose[:3, 3].cpu().numpy() if isinstance(retargeted_active_camera_pose, torch.Tensor) else retargeted_active_camera_pose[:3, 3]
                                
                                # LOS only for the query points of cam_idx
                                if cam_idx < num_horizons:
                                    # remove the retargeted active camera LOS geometry
                                    retargeted_active_los_key = f"{name}_retargeted_active_los_{cam_idx}_h{cam_idx}"
                                    if retargeted_active_los_key in self.los_geometries:
                                        visualizer.remove_geometry(self.los_geometries[retargeted_active_los_key])
                                    
                                    # use only the query points of cam_idx
                                    query_points_h = query_points_xyz[cam_idx].copy()  # [N, 3]
                                    
                                    # filter by query_point_indices if given
                                    if query_point_indices is not None:
                                        query_points_h = query_points_h[query_point_indices].copy()  # [M, 3]
                                        cam_visibility_results = cam_visibility_results[query_point_indices].copy()  # [M]
                                    
                                    # Store query_points_h for comparison
                                    retargeted_query_points_dict[cam_idx] = query_points_h.copy()
                                    
                                    # brightness along the horizon
                                    if selected_indices is not None and len(selected_indices) > 1:
                                        selected_indices_list = selected_indices.tolist() if isinstance(selected_indices, np.ndarray) else list(selected_indices)
                                        cam_idx_position = selected_indices_list.index(cam_idx) if cam_idx in selected_indices_list else 0
                                        normalized_horizon = cam_idx_position / (len(selected_indices_list) - 1)
                                    else:
                                        normalized_horizon = 0.0
                                    
                                    # brightness decreases along the horizon (1.0 ~ 0.3)
                                    brightness = 1.0 - 0.7 * normalized_horizon  # 1.0 (bright, first step) ~ 0.3 (dark, last step)
                                    
                                    # retargeted active camera color by visibility: green if visible, red if occluded
                                    # brightness depends on the horizon step
                                    retargeted_active_colors = []
                                    for visible in cam_visibility_results:
                                        if visible:
                                            # green with brightness
                                            retargeted_active_colors.append([0, brightness, 0])  # green (visible)
                                        else:
                                            # red with brightness
                                            retargeted_active_colors.append([brightness, 0, 0])  # red (occluded)
                                    
                                    # create and add the retargeted active camera LOS
                                    retargeted_active_line_set = self.create_multiple_lines_of_sight(retargeted_active_camera_center, query_points_h, retargeted_active_colors)
                                    retargeted_active_los_key = f"{name}_retargeted_active_los_{cam_idx}_h{cam_idx}"
                                    self.los_geometries[retargeted_active_los_key] = retargeted_active_line_set
                                    visualizer.add_geometry(retargeted_active_line_set)
                                
            # Store visibility_info for frame annotation
            if step is not None:
                self.current_step = step
                if step not in self.current_visibility_info:
                    self.current_visibility_info[step] = {}
                
                # Store active camera visibility_info
                if active_visibility_results_info is not None and selected_indices is not None:
                    for cam_idx in selected_indices:
                        # Ensure cam_idx is a valid key in active_visibility_results_info
                        if cam_idx in active_visibility_results_info and cam_idx < len(active_camera_pose_list):  # Add this check
                            if cam_idx not in self.current_visibility_info[step]:
                                self.current_visibility_info[step][cam_idx] = {}
                            if 'active' in self.current_visibility_info[step][cam_idx]:
                                raise NotImplementedError(f"cam_idx: {cam_idx} already exists in self.current_visibility_info[step][cam_idx]['active']")
                            self.current_visibility_info[step][cam_idx]['active'] = active_visibility_results_info[cam_idx].copy()
                        else:
                            raise NotImplementedError(f"cam_idx: {cam_idx} not found in active_visibility_results_info or active_camera_pose_list length: {len(active_camera_pose_list)}")
                # Store retargeted camera visibility_info
                if retargeted_active_visibility_results_info is not None and selected_indices is not None:
                    for cam_idx in selected_indices:
                        # Ensure cam_idx is a valid key in retargeted_active_visibility_results_info
                        if cam_idx in retargeted_active_visibility_results_info and cam_idx < len(retargeted_active_camera_pose_list):  # Add this check
                            if cam_idx not in self.current_visibility_info[step]:
                                self.current_visibility_info[step][cam_idx] = {}
                            if 'retargeted' in self.current_visibility_info[step][cam_idx]:
                                raise NotImplementedError(f"cam_idx: {cam_idx} already exists in self.current_visibility_info[step][cam_idx]['retargeted']")
                            self.current_visibility_info[step][cam_idx]['retargeted'] = retargeted_active_visibility_results_info[cam_idx].copy()
                        else:
                            raise NotImplementedError(f"cam_idx: {cam_idx} not found in retargeted_active_visibility_results_info or retargeted_active_camera_pose_list length: {len(retargeted_active_camera_pose_list)}")
            # Compare query_points_h pairs between active and retargeted cameras
            # Check if query_points_h for each cam_idx are identical between active and retargeted
            print(f"visualize step: {step}")
            if len(active_query_points_dict) > 0 or len(retargeted_query_points_dict) > 0:
                all_cam_indices = set(active_query_points_dict.keys()) | set(retargeted_query_points_dict.keys())
                for cam_idx in all_cam_indices:
                    if cam_idx in active_query_points_dict and cam_idx in retargeted_query_points_dict:
                        active_qp = active_query_points_dict[cam_idx]
                        retargeted_qp = retargeted_query_points_dict[cam_idx]
                        if not np.array_equal(active_qp, retargeted_qp):
                            print(f"WARNING: query_points_h mismatch for cam_idx={cam_idx}")
                            print(f"  Active query_points_h shape: {active_qp.shape}, first point: {active_qp[0] if len(active_qp) > 0 else 'empty'}")
                            print(f"  Retargeted query_points_h shape: {retargeted_qp.shape}, first point: {retargeted_qp[0] if len(retargeted_qp) > 0 else 'empty'}")
                            print(f"  Max difference: {np.max(np.abs(active_qp - retargeted_qp)) if active_qp.shape == retargeted_qp.shape else 'shape mismatch'}")
                    elif cam_idx in active_query_points_dict:
                        print(f"WARNING: cam_idx={cam_idx} only exists in active_query_points_dict")
                    elif cam_idx in retargeted_query_points_dict:
                        print(f"WARNING: cam_idx={cam_idx} only exists in retargeted_query_points_dict")
            

            if feature_mesh is not None:
                self._visualize_nvblox_feature_mesh(feature_mesh)

            # camera frustum visualization
            # current camera poses (selected indices only)
            if current_camera_pose_list is not None and selected_indices is not None:
                self._visualize_current_camera_poses(current_camera_pose_list, selected_indices)
                
            
            # in active_camera mode, frustums of the selected active camera poses
            if self.active_camera and self.active_camera_pose_list is not None and selected_indices is not None:
                self._visualize_active_camera_poses(self.active_camera_pose_list, selected_indices)
            
            # in retargeted_active_camera mode, frustums of the selected retargeted active camera poses
            if self.active_camera and self.retargeted_active_camera_pose_list is not None and selected_indices is not None:
                self._visualize_retargeted_active_camera_poses(self.retargeted_active_camera_pose_list, selected_indices)
            
            # draw a white frustum at self.current_camera_pose
            if self.current_camera_pose is not None and self.camera_intrinsics is not None and self.image_size is not None:
                for name, visualizer in self.visualizers.items():
                    # remove the previous current camera pose frustum geometry
                    if name in self.current_camera_frustum_geometries:
                        visualizer.remove_geometry(self.current_camera_frustum_geometries[name])
                    
                    # create and add the white frustum
                    current_frustum_geometry = self.create_camera_frustum_geometry(
                        self.current_camera_pose, self.camera_intrinsics, self.image_size, color=[1, 1, 1]  # white
                    )
                    self.current_camera_frustum_geometries[name] = current_frustum_geometry
                    visualizer.add_geometry(current_frustum_geometry)

            # Restore views and update
            for name, visualizer in self.visualizers.items():
                if name in self.pending_camera_updates:
                    visualizer.set_view_status(self.pending_camera_updates[name]['view_status'])
                elif self.follow_camera_view:
                    # Follow camera mode: determine which camera to follow
                    if self.current_camera_pose is not None:
                        # default: follow the current camera pose
                        self._set_camera_pose_from_matrix(self.visualizers[name], self.current_camera_pose)
                        
                elif self.initial_camera_pose is not None:
                    # Static mode: use initial camera pose
                    self._set_camera_pose_from_matrix(self.visualizers[name], self.initial_camera_pose)
                    
                self._update_visualization(visualizer, name)

            # Handle pausing
            if self.pause:
                self._loop_while_paused()
        
        def save_video(self, episode_idx, result_save_path=None):
            """Save captured frames as separate MP4 videos for mesh and point cloud
            
            Args:
                episode_idx: Episode index for generating video path
                result_save_path: Path to save videos (if None, uses self.result_save_path)
            """
            if not self.video_save:
                print("Video saving disabled")
                return
            
            # Use provided result_save_path or fall back to stored one
            if result_save_path is None:
                result_save_path = self.result_save_path
            
            if result_save_path is None:
                raise ValueError("result_save_path must be provided either in __init__ or save_video")
            
            # Generate video path based on episode_idx
            video_path = os.path.join(result_save_path, f"visualization_{episode_idx}.mp4")
            
            # Save mesh video
            if len(self.captured_frames_mesh) > 0:
                mesh_video_path = video_path.replace('.mp4', '_mesh.mp4')
                self._save_frames_to_video(self.captured_frames_mesh, mesh_video_path, "mesh")
            else:
                print("No mesh frames captured")
            
            # Save point cloud video
            if len(self.captured_frames_point_cloud) > 0:
                point_cloud_video_path = video_path.replace('.mp4', '_point_cloud.mp4')
                self._save_frames_to_video(self.captured_frames_point_cloud, point_cloud_video_path, "point cloud")
            else:
                print("No point cloud frames captured")
        
        def _save_frames_to_video(self, frames, video_path, video_type):
            """Helper method to save frames to video file"""
            if len(frames) == 0:
                print(f"No {video_type} frames to save")
                return
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
            
            # Write frames with frame numbers
            for frame_idx, frame in enumerate(frames):
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Add frame number to the frame (centered at top)
                frame_text = f"Frame: {frame_idx}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                color = (255, 255, 255)  # White color
                thickness = 2
                # Get text size to position it properly
                (text_width, text_height), baseline = cv2.getTextSize(frame_text, font, font_scale, thickness)
                # Position text at top center
                frame_width = frame_bgr.shape[1]
                text_x = (frame_width - text_width) // 2  # Center horizontally
                text_y = 30  # Top position
                
                # Draw text
                cv2.putText(frame_bgr, frame_text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
                
                out.write(frame_bgr)
            
            # Release everything
            out.release()
            print(f"{video_type.capitalize()} video saved to: {video_path}")
            print(f"Total {video_type} frames: {len(frames)}")
            

def get_scene_mesh(rgb_frame, depth_frame, K_frame, relative_pose, voxel_size, mapper):
    # depth image to a torch tensor (GPU)
    depth_tensor = torch.from_numpy(depth_frame.astype(np.float32)).cuda()
    
    # RGB image to a torch tensor (GPU)
    rgb_tensor = torch.from_numpy(rgb_frame).cuda()
    
    # camera intrinsics to a torch tensor (CPU)
    intrinsics_tensor = torch.from_numpy(K_frame.astype(np.float32)).cpu()
    
    # camera pose
    if relative_pose is None:
        pose_tensor = torch.eye(4, dtype=torch.float32).cpu()
    else:
        pose_tensor = torch.from_numpy(relative_pose).float().cpu()
    
    # integrate the frame into nvblox (as in the nvblox sun3d.py example)
    mapper.add_depth_frame(depth_tensor, pose_tensor, intrinsics_tensor)
    mapper.add_color_frame(rgb_tensor, pose_tensor, intrinsics_tensor)
    
    # update the mesh (every frame)
    mapper.update_color_mesh()
    color_mesh = mapper.get_color_mesh()
    
    # nvblox ColorMesh
    scene_open3d = color_mesh.to_open3d()
    
    
    scene_mesh = scene_open3d

    # clean up the scene mesh (robot meshes don't need this)
    mesh_postprocess_start = time.time()
    
    scene_mesh.remove_duplicated_vertices()
    scene_mesh.remove_duplicated_triangles()
    scene_mesh.remove_degenerate_triangles()
    vprint(f"  Mesh postprocess time: {time.time() - mesh_postprocess_start:.4f} seconds") #  0.04s
    

    triangle_postprocess_start = time.time()
    
    
    # relative area criterion
    tri_clusters, _, cluster_area = scene_mesh.cluster_connected_triangles()
    cluster_area = np.asarray(cluster_area)

    largest = float(cluster_area.max()) if len(cluster_area) else 0.0
    alpha = 0.005   # e.g. remove clusters smaller than 0.5% of the largest one
    abs_floor = 800 * 0.5 * (voxel_size ** 2)  # absolute lower bound
    thresh = max(alpha * largest, abs_floor)

    keep_ids = np.where(cluster_area >= thresh)[0]
    mask = np.isin(tri_clusters, keep_ids)
    scene_mesh.remove_triangles_by_mask(~mask)
    scene_mesh.remove_unreferenced_vertices()

    vprint(f"  In multiframe example, Mesh remove triangles time: {time.time() - triangle_postprocess_start:.4f} seconds") # 0.04s
    return scene_mesh, pose_tensor

    