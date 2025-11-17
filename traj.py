import numpy as np
import mujoco
import mujoco.viewer
import time
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from ikpy.chain import Chain
from scipy.interpolate import CubicSpline, interp1d
from collections import deque
# from pybullet_planning import rrt_connect, birrt

# --------- CONFIG ----------
URDF = "puma560_description/urdf/puma560_robot.urdf"
MJCF = "puma560_description/urdf/puma560_robot.xml"
MESH_DIR = "puma560_description/meshes"

# Which link was the base you used earlier
BASE_ELEMENT = ['link1']

# trajectory params
cart_start = np.array([0.3, 0.2, 0.2])
cart_end   = np.array([0.5, 0.5, 1])
path_type = "arc"         # Options: "linear", "arc", "circular", "parabolic", "rrt"
coarse_points =2     # number of IK solves along path (coarse)
frames = 4000       # total frames to play out (fine)
dt = 0.01                 # seconds per render step (sleep); adjust with viewer rate
max_joint_vel = 1.5       # rad/s (per joint) — tune to reduce jerk
use_pd_control = False    # if True we'll output desired qpos and you can adapt to your actuators
interp_type = "bangbang"     # Options: "cubic", "quintic", "lspb", "bangbang"
show_planned_path = True  # Show the planned path before execution


def simple_rrt_cartesian(start, goal, n_waypoints=15, max_iterations=500, step_size=0.1):
    """
    Simple RRT in Cartesian space to find collision-free path.
    This is a basic implementation - for production use pybullet-planning or OMPL.
    """
    print("Running RRT path planning in Cartesian space...")
    
    # Simple obstacle check (you can extend this)
    def is_valid(point):
        # Check workspace bounds
        if point[2] < 0.1:  # Don't go below z=0.1
            return False
        if np.linalg.norm(point[:2]) > 1.0:  # Stay within radius
            return False
        return True
    
    # RRT Tree
    tree = [start]
    parent = {0: None}
    
    for iteration in range(max_iterations):
        # Random sample (bias toward goal)
        if np.random.random() < 0.1:  # 10% goal bias
            random_point = goal
        else:
            random_point = np.array([
                np.random.uniform(-0.6, 0.6),
                np.random.uniform(-0.6, 0.6),
                np.random.uniform(0.1, 1.0)
            ])
        
        # Find nearest node in tree
        distances = [np.linalg.norm(node - random_point) for node in tree]
        nearest_idx = np.argmin(distances)
        nearest = tree[nearest_idx]
        
        # Extend toward random point
        direction = random_point - nearest
        distance = np.linalg.norm(direction)
        if distance > step_size:
            direction = direction / distance * step_size
        
        new_point = nearest + direction
        
        # Check if valid
        if is_valid(new_point):
            new_idx = len(tree)
            tree.append(new_point)
            parent[new_idx] = nearest_idx
            
            # Check if we reached goal
            if np.linalg.norm(new_point - goal) < step_size:
                # Reconstruct path
                path = [goal]
                current = new_idx
                while current is not None:
                    path.append(tree[current])
                    current = parent[current]
                path.reverse()
                
                # Resample to get desired number of waypoints
                path_array = np.array(path)
                t_original = np.linspace(0, 1, len(path))
                t_new = np.linspace(0, 1, n_waypoints)
                
                resampled_path = np.zeros((n_waypoints, 3))
                for dim in range(3):
                    resampled_path[:, dim] = np.interp(t_new, t_original, path_array[:, dim])
                
                print(f"  RRT found path in {iteration+1} iterations with {len(path)} nodes")
                return resampled_path
    
    print("  RRT failed to find path, using direct interpolation")
    return np.linspace(start, goal, n_waypoints)


def generate_cartesian_path(start, end, n_points, path_type="linear"):
    """
    Generate better Cartesian space paths between start and end points.
    """
    if path_type == "linear":
        print("Using linear path in Cartesian space")
        return np.linspace(start, end, n_points)
    
    elif path_type == "arc":
        print("Using smooth arc path (avoiding obstacles)")
        t = np.linspace(0, 1, n_points)
        linear_path = np.outer(1-t, start) + np.outer(t, end)
        mid_height = 0.2
        vertical_offset = 4 * mid_height * t * (1 - t)
        arc_path = linear_path.copy()
        arc_path[:, 2] += vertical_offset
        return arc_path
    
    elif path_type == "circular":
        print("Using circular path")
        center = (start + end) / 2
        radius = np.linalg.norm(end - start) / 2
        direction = end - start
        direction = direction / np.linalg.norm(direction)
        up = np.array([0, 0, 1])
        perp = np.cross(direction, up)
        if np.linalg.norm(perp) < 0.01:
            perp = np.array([1, 0, 0])
        perp = perp / np.linalg.norm(perp)
        angles = np.linspace(0, np.pi, n_points)
        path = []
        for angle in angles:
            offset = radius * (np.cos(angle) * direction + np.sin(angle) * perp)
            point = center + offset
            path.append(point)
        return np.array(path)
    
    elif path_type == "parabolic":
        print("Using parabolic path with lateral movement")
        t = np.linspace(0, 1, n_points)
        linear_path = np.outer(1-t, start) + np.outer(t, end)
        variation_y = 0.1 * np.sin(np.pi * t)
        variation_z = 0.15 * (4 * t * (1 - t))
        parabolic_path = linear_path.copy()
        parabolic_path[:, 1] += variation_y
        parabolic_path[:, 2] += variation_z
        return parabolic_path
    
    elif path_type == "rrt":
        return simple_rrt_cartesian(start, end, n_points)
    
    else:
        raise ValueError(f"Unknown path type: {path_type}")


def interpolate_trajectory(q_coarse, t_coarse, t_fine, method="cubic"):
    """Interpolate joint-space trajectory using various methods."""
    n_points, n_joints = q_coarse.shape
    n_fine = len(t_fine)
    q_fine = np.zeros((n_fine, n_joints))
    
    if method == "cubic":
        print("Using Cubic Spline interpolation")
        for j in range(n_joints):
            cs = CubicSpline(t_coarse, q_coarse[:, j], bc_type='clamped')
            q_fine[:, j] = cs(t_fine)
    
    elif method == "quintic":
        print("Using Quintic polynomial interpolation")
        for j in range(n_joints):
            q_fine[:, j] = quintic_interpolation(t_coarse, q_coarse[:, j], t_fine)
    
    elif method == "lspb":
        print("Using Linear Segment with Parabolic Blends (LSPB)")
        for j in range(n_joints):
            q_fine[:, j] = lspb_interpolation(t_coarse, q_coarse[:, j], t_fine)
    
    elif method == "bangbang":
        print("Using Bang-Bang control")
        for j in range(n_joints):
            q_fine[:, j] = bangbang_interpolation(t_coarse, q_coarse[:, j], t_fine)
    
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    for j in range(n_joints):
        q_fine[:, j] = np.unwrap(q_fine[:, j])
    
    return q_fine


def quintic_interpolation(t_coarse, q_coarse, t_fine):
    q_fine = np.zeros(len(t_fine))
    for i in range(len(t_coarse) - 1):
        mask = (t_fine >= t_coarse[i]) & (t_fine <= t_coarse[i+1])
        if not np.any(mask):
            continue
        t0, t1 = t_coarse[i], t_coarse[i+1]
        q0, q1 = q_coarse[i], q_coarse[i+1]
        tau = (t_fine[mask] - t0) / (t1 - t0)
        a0, a1, a2 = q0, 0, 0
        a3 = 10 * (q1 - q0)
        a4 = -15 * (q1 - q0)
        a5 = 6 * (q1 - q0)
        q_fine[mask] = a0 + a1*tau + a2*tau**2 + a3*tau**3 + a4*tau**4 + a5*tau**5
    return q_fine


def lspb_interpolation(t_coarse, q_coarse, t_fine):
    q_fine = np.zeros(len(t_fine))
    for i in range(len(t_coarse) - 1):
        mask = (t_fine >= t_coarse[i]) & (t_fine <= t_coarse[i+1])
        if not np.any(mask):
            continue
        t0, t1 = t_coarse[i], t_coarse[i+1]
        q0, q1 = q_coarse[i], q_coarse[i+1]
        T = t1 - t0
        tb = 0.3 * T
        tau = (t_fine[mask] - t0) / T
        q_diff = q1 - q0
        accel_mask = tau < tb / T
        tau_a = tau[accel_mask]
        q_fine[np.where(mask)[0][accel_mask]] = q0 + 0.5 * q_diff / (tb / T) * tau_a**2
        const_mask = (tau >= tb / T) & (tau <= 1 - tb / T)
        tau_c = tau[const_mask]
        v_max = q_diff / (T - tb)
        q_fine[np.where(mask)[0][const_mask]] = q0 + v_max * (tau_c * T - tb/2)
        decel_mask = tau > 1 - tb / T
        tau_d = tau[decel_mask]
        q_fine[np.where(mask)[0][decel_mask]] = q1 - 0.5 * q_diff / (tb / T) * (1 - tau_d)**2
    return q_fine


def bangbang_interpolation(t_coarse, q_coarse, t_fine):
    q_fine = np.zeros(len(t_fine))
    for i in range(len(t_coarse) - 1):
        mask = (t_fine >= t_coarse[i]) & (t_fine <= t_coarse[i+1])
        if not np.any(mask):
            continue
        t0, t1 = t_coarse[i], t_coarse[i+1]
        q0, q1 = q_coarse[i], q_coarse[i+1]
        tau = (t_fine[mask] - t0) / (t1 - t0)
        q_diff = q1 - q0
        accel_mask = tau < 0.5
        tau_a = tau[accel_mask]
        q_fine[np.where(mask)[0][accel_mask]] = q0 + 2 * q_diff * tau_a**2
        decel_mask = tau >= 0.5
        tau_d = tau[decel_mask]
        q_fine[np.where(mask)[0][decel_mask]] = q1 - 2 * q_diff * (1 - tau_d)**2
    return q_fine


def add_planned_path_to_mjcf(mjcf_path, cart_waypoints):
    """Add planned path visualization directly to MJCF file."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    
    worldbody = root.find('worldbody')
    if worldbody is None:
        worldbody = ET.SubElement(root, 'worldbody')
    
    # Add spheres for waypoints
    for i, point in enumerate(cart_waypoints):
        body = ET.SubElement(worldbody, 'body', 
                            name=f'waypoint_{i}',
                            pos=f"{point[0]} {point[1]} {point[2]}")
        ET.SubElement(body, 'geom', 
                     type='sphere', 
                     size='0.01',
                     rgba='1 1 0 0.7',  # Yellow waypoints
                     contype='0', 
                     conaffinity='0')
    
    # Add cylinders connecting waypoints
    for i in range(len(cart_waypoints) - 1):
        p1 = cart_waypoints[i]
        p2 = cart_waypoints[i + 1]
        midpoint = (p1 + p2) / 2
        
        direction = p2 - p1
        length = np.linalg.norm(direction)
        
        if length > 0.001:
            direction = direction / length
            
            # Compute quaternion for orientation
            z_axis = np.array([0, 0, 1])
            axis = np.cross(z_axis, direction)
            axis_len = np.linalg.norm(axis)
            
            if axis_len > 0.001:
                axis = axis / axis_len
                angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
                qw = np.cos(angle / 2)
                qx, qy, qz = axis * np.sin(angle / 2)
                quat_str = f"{qw} {qx} {qy} {qz}"
            else:
                quat_str = "1 0 0 0"
            
            body = ET.SubElement(worldbody, 'body',
                                name=f'path_segment_{i}',
                                pos=f"{midpoint[0]} {midpoint[1]} {midpoint[2]}",
                                quat=quat_str)
            ET.SubElement(body, 'geom',
                         type='cylinder',
                         size=f'0.005 {length/2}',
                         rgba='1 0.5 0 0.5',  # Orange path
                         contype='0',
                         conaffinity='0')
    
    tree.write(mjcf_path)


def fix_mjcf_mesh_paths(original_mjcf, mesh_dir, start_pos=None, end_pos=None, planned_path=None):
    """Create a temporary MJCF file with absolute mesh paths and trajectory markers."""
    tree = ET.parse(original_mjcf)
    root = tree.getroot()
    
    abs_mesh_dir = os.path.abspath(mesh_dir)
    
    for mesh in root.iter('mesh'):
        if 'file' in mesh.attrib:
            filename = os.path.basename(mesh.attrib['file'])
            mesh.attrib['file'] = os.path.join(abs_mesh_dir, filename)
    
    worldbody = root.find('worldbody')
    if worldbody is None:
        worldbody = ET.SubElement(root, 'worldbody')
    
    if start_pos is not None:
        pos_str = f"{start_pos[0]} {start_pos[1]} {start_pos[2]}"
        start_body = ET.SubElement(worldbody, 'body', name='start_marker', pos=pos_str)
        ET.SubElement(start_body, 'geom', type='sphere', size='0.03', 
                     rgba='0 1 0 0.8', contype='0', conaffinity='0')
    
    if end_pos is not None:
        pos_str = f"{end_pos[0]} {end_pos[1]} {end_pos[2]}"
        end_body = ET.SubElement(worldbody, 'body', name='end_marker', pos=pos_str)
        ET.SubElement(end_body, 'geom', type='sphere', size='0.03',
                     rgba='1 0 0 0.8', contype='0', conaffinity='0')
    
    # Add planned path visualization
    if planned_path is not None:
        for i, point in enumerate(planned_path):
            body = ET.SubElement(worldbody, 'body', 
                                name=f'planned_waypoint_{i}',
                                pos=f"{point[0]} {point[1]} {point[2]}")
            ET.SubElement(body, 'geom', 
                         type='sphere', 
                         size='0.012',
                         rgba='1 1 0 0.6',
                         contype='0', 
                         conaffinity='0')
        
        # Connect waypoints with cylinders
        for i in range(len(planned_path) - 1):
            p1 = planned_path[i]
            p2 = planned_path[i + 1]
            midpoint = (p1 + p2) / 2
            direction = p2 - p1
            length = np.linalg.norm(direction)
            
            if length > 0.001:
                direction = direction / length
                z_axis = np.array([0, 0, 1])
                axis = np.cross(z_axis, direction)
                axis_len = np.linalg.norm(axis)
                
                if axis_len > 0.001:
                    axis = axis / axis_len
                    angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
                    qw = np.cos(angle / 2)
                    qx, qy, qz = axis * np.sin(angle / 2)
                    quat_str = f"{qw} {qx} {qy} {qz}"
                else:
                    quat_str = "1 0 0 0"
                
                body = ET.SubElement(worldbody, 'body',
                                    name=f'planned_path_seg_{i}',
                                    pos=f"{midpoint[0]} {midpoint[1]} {midpoint[2]}",
                                    quat=quat_str)
                ET.SubElement(body, 'geom',
                             type='cylinder',
                             size=f'0.006 {length/2}',
                             rgba='1 0.7 0 0.5',
                             contype='0',
                             conaffinity='0')
    
    tmpdir = tempfile.mkdtemp()
    tmp_mjcf = os.path.join(tmpdir, "robot_temp.xml")
    tree.write(tmp_mjcf)
    
    return tmp_mjcf, tmpdir


# --------- load models -----------
print("Loading URDF chain...")
chain = Chain.from_urdf_file(URDF, base_elements=BASE_ELEMENT, base_element_type='link', last_link_vector=[0,0,0])

# Generate planned path first
print(f"\n{'='*60}")
print(f"PLANNING PHASE")
print(f"{'='*60}")
print(f"  Start point (GREEN): {cart_start}")
print(f"  End point (RED): {cart_end}")
print(f"  Path type: {path_type}")

cart_points = generate_cartesian_path(cart_start, cart_end, coarse_points, path_type)

print(f"\nFixing MJCF mesh paths and adding visualizations...")
tmp_mjcf, tmpdir = fix_mjcf_mesh_paths(MJCF, MESH_DIR, cart_start, cart_end, cart_points if show_planned_path else None)

try:
    model = mujoco.MjModel.from_xml_path(tmp_mjcf)
    data = mujoco.MjData(model)
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

active_mask = np.array(chain.active_links_mask, dtype=bool)
active_names = [link.name for i, link in enumerate(chain.links) if active_mask[i]]
n_active = len(active_names)
print(f"Active joint names: {active_names}")
print(f"Number of active joints: {n_active}")


def ik_for_point(pt):
    q_full = chain.inverse_kinematics(pt)
    q_active = np.array([q_full[i] for i, m in enumerate(chain.active_links_mask) if m], dtype=float)
    return q_active


# Solve IK for waypoints
print(f"\nSolving inverse kinematics for {coarse_points} waypoints...")
q_coarse = []
failed_points = []
for i, pt in enumerate(cart_points):
    try:
        q = ik_for_point(pt)
        q_coarse.append(q)
    except Exception as e:
        print(f"  Warning: IK failed for waypoint {i} at {pt}")
        failed_points.append(i)
        if len(q_coarse) > 0:
            q_coarse.append(q_coarse[-1])
        else:
            q_coarse.append(np.zeros(n_active))

q_coarse = np.stack(q_coarse)
if failed_points:
    print(f"  Warning: {len(failed_points)} IK solutions failed")

print("Unwrapping joint angles...")
for j in range(n_active):
    q_coarse[:, j] = np.unwrap(q_coarse[:, j])


def read_current_active_from_mujoco():
    muj_names = [model.joint(i).name for i in range(model.nq)]
    current = []
    for name in active_names:
        if name in muj_names:
            idx = muj_names.index(name)
            current.append(float(data.qpos[idx]))
        else:
            current.append(0.0)
    return np.array(current, dtype=float)


q_now = read_current_active_from_mujoco()
for j in range(n_active):
    diff = q_now[j] - q_coarse[0, j]
    k = np.round(diff / (2*np.pi))
    q_coarse[:, j] += k * 2*np.pi

print(f"Interpolating trajectory to {frames} frames using {interp_type}...")
t_coarse = np.linspace(0.0, 1.0, coarse_points)
t_fine = np.linspace(0.0, 1.0, frames)
q_fine = interpolate_trajectory(q_coarse, t_coarse, t_fine, method=interp_type)


muj_names = [model.joint(i).name for i in range(model.njnt)]

def active_to_full_qpos(active_q):
    qpos = list(data.qpos.copy())
    for name, val in zip(active_names, active_q):
        if name in muj_names:
            idx = muj_names.index(name)
            qpos[idx] = float(val)
    return np.array(qpos, dtype=float)


print(f"\n{'='*60}")
print(f"EXECUTION PHASE")
print(f"{'='*60}")
print(f"Total time: {frames*dt:.1f}s")
print(f"Max joint velocity: {max_joint_vel} rad/s\n")

ee_path_history = deque(maxlen=5000)  # Store more points

with mujoco.viewer.launch_passive(model, data) as viewer:
    prev_active = read_current_active_from_mujoco()
    
    for frame_idx in range(frames):
        desired_active = q_fine[frame_idx]
        delta = desired_active - prev_active
        max_step = max_joint_vel * dt
        too_big = np.abs(delta) > max_step
        
        if np.any(too_big):
            delta[too_big] = np.sign(delta[too_big]) * max_step
        
        new_active = prev_active + delta
        alpha = 0.9
        new_active = alpha * new_active + (1.0 - alpha) * prev_active

        qpos_full = active_to_full_qpos(new_active)
        data.qpos[:] = qpos_full
        mujoco.mj_forward(model, data)
        
        # Get and store end-effector position
        ee_body_id = model.nbody - 1
        ee_pos = data.xpos[ee_body_id].copy()
        ee_path_history.append(ee_pos)
        
        # Draw execution trace (cyan) - draw every point for smooth trail
        if len(ee_path_history) >= 2:
            path_list = list(ee_path_history)
            # Draw recent path with higher opacity
            for i in range(len(path_list) - 1):
                if i % 2 == 0:  # Draw every other segment to reduce geometry count
                    p1 = path_list[i]
                    p2 = path_list[i + 1]
                    
                    # Fade older parts of the trail
                    age_factor = i / len(path_list)
                    alpha_val = 0.3 + 0.5 * age_factor
                    
                    if viewer.user_scn.ngeom < viewer.user_scn.maxgeom - 1:
                        midpoint = (p1 + p2) / 2
                        direction = p2 - p1
                        length = np.linalg.norm(direction)
                        
                        if length > 0.0001:
                            direction = direction / length
                            z_axis = np.array([0, 0, 1])
                            
                            if np.abs(np.dot(direction, z_axis)) < 0.999:
                                rot_axis = np.cross(z_axis, direction)
                                rot_axis_norm = np.linalg.norm(rot_axis)
                                if rot_axis_norm > 0.001:
                                    rot_axis = rot_axis / rot_axis_norm
                                    angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
                                    K = np.array([[0, -rot_axis[2], rot_axis[1]],
                                                 [rot_axis[2], 0, -rot_axis[0]],
                                                 [-rot_axis[1], rot_axis[0], 0]])
                                    rot_mat = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
                                else:
                                    rot_mat = np.eye(3)
                            else:
                                rot_mat = np.eye(3) if direction[2] > 0 else -np.eye(3)
                            
                            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                            mujoco.mjv_initGeom(
                                geom,
                                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                size=[0.003, length/2, 0],
                                pos=midpoint,
                                mat=rot_mat.flatten(),
                                rgba=np.array([0, 1, 1, alpha_val])  # Cyan trace
                            )
                            viewer.user_scn.ngeom += 1
        
        viewer.sync()
        time.sleep(dt)
        prev_active = new_active.copy()
        
        if (frame_idx + 1) % 100 == 0:
            print(f"  Progress: {100*(frame_idx+1)/frames:.1f}% ({frame_idx+1}/{frames} frames)")

ee_path = np.array(list(ee_path_history))
print(f"\n{'='*60}")
print(f"TRAJECTORY STATISTICS")
print(f"{'='*60}")
print(f"Total path length: {np.sum(np.linalg.norm(np.diff(ee_path, axis=0), axis=1)):.4f} m")
print(f"Start position: {ee_path[0]}")
print(f"End position: {ee_path[-1]}")
print(f"Error from target start: {np.linalg.norm(ee_path[0] - cart_start):.6f} m")
print(f"Error from target end: {np.linalg.norm(ee_path[-1] - cart_end):.6f} m")
print(f"\n✓ Trajectory playback finished!")