import numpy as np
import os
import open3d as o3d
from urdf_parser_py.urdf import URDF
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

from egoavflow.visibility import SO3Constraint

# Verbose logging control
VERBOSE = False

def vprint(*args, **kwargs) -> None:
    """Print only when VERBOSE is True."""
    if VERBOSE:
        print(*args, **kwargs)


class WidowXVisualizer:
    def __init__(self, urdf_path, mesh_base_path, ee_bias = None):
        """
        WidowX robot visualizer
        
        Args:
            urdf_path: path to the URDF file
            mesh_base_path: base directory of the mesh files
            
        """
        self.urdf_path = urdf_path
        self.mesh_base_path = mesh_base_path
        self.ee_bias = ee_bias
        assert ee_bias is not None
        self.robot = None
        self.joint_angles = np.zeros(6)  # 6 joint angles
        self.gripper_position = 0.0  # gripper position (prismatic joint)
        self.link_transforms = {}  # transform of each link
        self.meshes = {}  # loaded meshes
        
        # Link colors (silver)
        self.link_colors = {
            'base_link': [0.8, 0.8, 0.8],
            'link_1': [0.8, 0.8, 0.8],
            'link_2': [0.8, 0.8, 0.8],
            'link_3': [0.8, 0.8, 0.8],
            'link_4': [0.8, 0.8, 0.8],
            'link_5': [0.8, 0.8, 0.8],
            'link_6': [0.8, 0.8, 0.8],
            'carriage_left': [0.8, 0.8, 0.8],
            'carriage_right': [0.8, 0.8, 0.8],
            'gripper_left': [0.8, 0.8, 0.8],
            'gripper_right': [0.8, 0.8, 0.8],
        }

        # Joint limits from URDF (joint_0 to joint_5)
        self.joint_limits = [
            (-3.14159, 3.14159),   
            (0.0, 3.14159),     
            (0.0, 2.35619),     
            (-1.57079, 1.57079),   
            (-1.57079, 1.57079),     
            (-3.14159, 3.14159)    
        ]
        
        # Gripper limits (prismatic joint)
        self.gripper_limits = (0.0, 0.044)  # 0 to 44mm

        # environment interface (unused)
        
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
                            mesh_path = mesh_filename.replace('package://trossen_arm_description/', self.mesh_base_path)
                            
                            # STL file path (the URDF already points to .stl files)
                            if os.path.exists(mesh_path):
                                # load the STL file
                                mesh_obj = o3d.io.read_triangle_mesh(mesh_path)
                                if len(mesh_obj.vertices) > 0:
                                    # simplify the mesh (faster raycasting)
                                    original_vertices = len(mesh_obj.vertices)
                                    original_triangles = len(mesh_obj.triangles)
                                    
                                    # reduce the number of triangles (set target_triangle_count)
                                    if link_name in ['base_link', 'link_1', 'link_2', 'link_3', 'link_4']:
                                        target_triangle_count = 100  # keep at least 100 triangles
                                    else:
                                        target_triangle_count = max(100, original_triangles) # // 4  # keep at least 100 triangles
                                    
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
                                        mesh_obj = o3d.io.read_triangle_mesh(mesh_path)
                                        print(f"Mesh over-simplified, using the original: {link_name}")
                                    else:
                                        print(f"Simplified mesh: {link_name} - {original_vertices}->{len(mesh_obj.vertices)} vertices, {original_triangles}->{len(mesh_obj.triangles)} triangles")
                                    
                                    # color: only set when the mesh has no colors
                                    # use the colors from the mesh file if present, otherwise the default color
                                    if link_name in self.link_colors:
                                        # check whether the mesh has vertex colors
                                        has_colors = len(mesh_obj.vertex_colors) > 0 and len(mesh_obj.vertex_colors) == len(mesh_obj.vertices)
                                        if not has_colors:
                                            # set the default color if the mesh has none
                                            mesh_obj.vertex_colors = o3d.utility.Vector3dVector(
                                                np.tile(np.array(self.link_colors[link_name])[None, :], (len(mesh_obj.vertices), 1))
                                            )
                                        else:
                                            print(f"Mesh already has colors: {link_name}, keeping them")
                                    # if commented out, the mesh file colors are always used and meshes without colors are shown uncolored
                                    self.meshes[link_name] = mesh_obj
                                    vprint(f"Loaded STL mesh: {link_name} -> {mesh_path} ({len(mesh_obj.vertices)} vertices)")
                                    
    def set_joint_angles(self, angles, gripper_position=None):
        """Set joint angles"""
        self.joint_angles = np.array(angles)
        if gripper_position is not None:
            self.gripper_position = np.clip(gripper_position, self.gripper_limits[0], self.gripper_limits[1])
        vprint(f"Joint angles set: {self.joint_angles}, gripper position: {self.gripper_position}")
    
    def _get_current_ee_se3(self, joint_pos=None, gripper_pos=None):
        """SE(3) transform of the current end effector"""
        if joint_pos is not None:
            old_angles = self.joint_angles.copy()
            old_gripper = self.gripper_position
            self.joint_angles = np.array(joint_pos)
            if gripper_pos is not None:
                self.gripper_position = gripper_pos
        
        # forward kinematics
        self.compute_forward_kinematics_urdf(self.gripper_position)
        
        # the end effector is link_6 or ee_gripper_link
        if 'ee_gripper_link' in self.link_transforms:
            ee_transform = self.link_transforms['ee_gripper_link'].copy()
        elif 'link_6' in self.link_transforms:
            # use link_6 if ee_gripper_link does not exist
            ee_transform = self.link_transforms['link_6'].copy()
        else:
            # default
            ee_transform = np.eye(4)
        
        if joint_pos is not None:
            self.joint_angles = old_angles
            self.gripper_position = old_gripper
        
        return ee_transform
    
    def solve_inverse_kinematics(self, target_pose):
        """Compute joint angles for a target pose with inverse kinematics"""
            
        try:
            # use the current joint angles as the initial guess
            current_q = self.joint_angles.copy()
            vprint(f"Initial joint angles: {current_q}")
            if (current_q == np.zeros(6)).all():
                initial_guess = None
            else:
                initial_guess = current_q


            T_ge = np.eye(4)
            T_ge[:3, 3] = -self.ee_bias.copy()
            
            T_bg = target_pose.copy()
            
            T_be = T_bg @ T_ge
            target_pose = T_be.copy()
            

            success, q_forward, iter_taken, error, null_obj_val = self.solve_ik_null_space(target_pose, initial_guess=initial_guess, max_iterations=50, tolerance=1e-2, tolerance_null=5e-2, epsilon=1e-6)
            
            
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
            current_T = self._get_current_ee_se3(joint_pos=q)
            
            
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
                
                
                perturb_T = self._get_current_ee_se3(joint_pos=q_perturb)

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
        current_T = self._get_current_ee_se3(joint_pos=q)
        
        final_error = err # np.linalg.norm(final_pos_error)
        
        # Check if both conditions are satisfied for success
        success = (final_error < tolerance and null_obj_val < tolerance_null)
        
        
        return success, q, iter_taken, final_error, null_obj_val

    def jacobian_position(self, q: np.ndarray) -> np.ndarray:
        """Calculate position Jacobian using numerical differentiation."""
        epsilon = 1e-6
        epsilon_inv = 1.0 / epsilon
        T = self._get_current_ee_se3(joint_pos=q)
        p = T[:3, 3]
        jac = np.zeros([3, 6])
        for i in range(6):
            q_ = q.copy()
            q_[i] = q_[i] + epsilon
            T_ = self._get_current_ee_se3(joint_pos=q_)
            p_ = T_[:3, 3]
            jac[:, i] = (p_ - p) * epsilon_inv
        return jac

    def compute_forward_kinematics(self, joint_angles=None, gripper_position=0.0):
        """Forward kinematics from the URDF"""
        if joint_angles is not None:
            self.joint_angles = joint_angles.copy()
            
        self.compute_forward_kinematics_urdf(gripper_position)
        # return the end-effector transform
        if 'ee_gripper_link' in self.link_transforms:
            return self.link_transforms['ee_gripper_link'].copy()
        elif 'link_6' in self.link_transforms:
            return self.link_transforms['link_6'].copy()
        else:
            return np.eye(4)
    
    def compute_forward_kinematics_urdf(self, gripper_position=0.0):
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
        
        # world -> base_link (fixed)
        self.link_transforms['world'] = np.eye(4)
        self.link_transforms['base_link'] = np.eye(4)
        
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
                
            elif joint.type == 'prismatic':
                # Prismatic joint (gripper)
                if joint.name == 'left_carriage_joint':
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
                    
                    # prismatic transform (translation along the axis)
                    translation = axis * gripper_position
                    
                    # build the transform
                    T_joint = np.eye(4)
                    T_joint[:3, :3] = rot_matrix
                    T_joint[:3, 3] = xyz + translation
                    
                    # compose with the parent link transform
                    parent_transform = self.link_transforms.get(joint.parent, np.eye(4))
                    child_transform = parent_transform @ T_joint
                    
                    self.link_transforms[joint.child] = child_transform
                    
                    # right_carriage_joint mimics the left one
                    # mimic joints are handled manually here
                    if 'right_carriage_joint' in joint_info:
                        right_joint = joint_info['right_carriage_joint']
                        right_origin = right_joint.origin
                        if right_origin is not None:
                            right_xyz = np.array(right_origin.xyz) if right_origin.xyz else np.zeros(3)
                            right_rpy = np.array(right_origin.rpy) if right_origin.rpy else np.zeros(3)
                        else:
                            right_xyz = np.zeros(3)
                            right_rpy = np.zeros(3)
                        
                        if np.any(right_rpy):
                            right_rot_matrix = R.from_euler('xyz', right_rpy).as_matrix()
                        else:
                            right_rot_matrix = np.eye(3)
                        
                        # the right carriage moves in the opposite direction
                        right_axis = np.array(right_joint.axis)
                        right_translation = right_axis * gripper_position
                        
                        T_right_joint = np.eye(4)
                        T_right_joint[:3, :3] = right_rot_matrix
                        T_right_joint[:3, 3] = right_xyz + right_translation
                        
                        right_parent_transform = self.link_transforms.get(right_joint.parent, np.eye(4))
                        right_child_transform = right_parent_transform @ T_right_joint
                        self.link_transforms[right_joint.child] = right_child_transform
                
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
            gripper_angle: gripper position (prismatic joint)
            T_target: transform of the target frame (4x4). If None, the world frame is used
        
        Returns:
            transformed_meshes: dict of meshes in the target frame
        """
        # forward kinematics
        self.compute_forward_kinematics(gripper_position=gripper_angle)
        
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

