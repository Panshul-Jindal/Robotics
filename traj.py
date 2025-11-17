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

# --------- CONFIG ----------
URDF = "puma560_description/urdf/puma560_robot.urdf"
MJCF = "puma560_description/urdf/puma560_robot.xml"
MESH_DIR = "puma560_description/meshes"

# Which link was the base you used earlier
BASE_ELEMENT = ['link1']

# trajectory params
cart_start = np.array([0.3, 0.2, 0.2])
cart_end   = np.array([0, 0.0, 1])
path_type = "arc"         # Options: "linear", "arc", "circular", "parabolic"
coarse_points = 20        # number of IK solves along path (coarse)
frames = 3000             # total frames to play out (fine)
dt = 0.01                 # seconds per render step (sleep); adjust with viewer rate
max_joint_vel = 1.5       # rad/s (per joint) — tune to reduce jerk
use_pd_control = False    # if True we'll output desired qpos and you can adapt to your actuators
interp_type = "quintic"     # Options: "cubic", "quintic", "lspb", "bangbang"


def generate_cartesian_path(start, end, n_points, path_type="linear"):
    """
    Generate better Cartesian space paths between start and end points.
    
    Args:
        start: Starting 3D position
        end: Ending 3D position
        n_points: Number of waypoints
        path_type: "linear", "arc", "circular", "parabolic"
    
    Returns:
        cart_points: (n_points, 3) array of Cartesian waypoints
    """
    if path_type == "linear":
        print("Using linear path in Cartesian space")
        return np.linspace(start, end, n_points)
    
    elif path_type == "arc":
        print("Using smooth arc path (avoiding obstacles)")
        # Create an arc that goes up in the middle
        t = np.linspace(0, 1, n_points)
        
        # Linear interpolation
        linear_path = np.outer(1-t, start) + np.outer(t, end)
        
        # Add a vertical offset (parabolic) to create an arc
        mid_height = 0.2  # Height of arc above straight line
        vertical_offset = 4 * mid_height * t * (1 - t)  # Parabola: peaks at t=0.5
        
        arc_path = linear_path.copy()
        arc_path[:, 2] += vertical_offset  # Add to Z component
        
        return arc_path
    
    elif path_type == "circular":
        print("Using circular path")
        # Create a circular arc in 3D space
        center = (start + end) / 2
        radius = np.linalg.norm(end - start) / 2
        
        # Create rotation axis perpendicular to start-end line
        direction = end - start
        direction = direction / np.linalg.norm(direction)
        
        # Choose perpendicular vector (use cross product with up vector)
        up = np.array([0, 0, 1])
        perp = np.cross(direction, up)
        if np.linalg.norm(perp) < 0.01:  # If direction is vertical
            perp = np.array([1, 0, 0])
        perp = perp / np.linalg.norm(perp)
        
        # Generate circular arc
        angles = np.linspace(0, np.pi, n_points)
        path = []
        for angle in angles:
            # Rotate around the perpendicular axis
            offset = radius * (np.cos(angle) * direction + np.sin(angle) * perp)
            point = center + offset
            path.append(point)
        
        return np.array(path)
    
    elif path_type == "parabolic":
        print("Using parabolic path with lateral movement")
        t = np.linspace(0, 1, n_points)
        
        # Base linear interpolation
        linear_path = np.outer(1-t, start) + np.outer(t, end)
        
        # Add parabolic variation in multiple dimensions
        variation_y = 0.1 * np.sin(np.pi * t)  # Lateral swing
        variation_z = 0.15 * (4 * t * (1 - t))  # Vertical arc
        
        parabolic_path = linear_path.copy()
        parabolic_path[:, 1] += variation_y
        parabolic_path[:, 2] += variation_z
        
        return parabolic_path
    
    else:
        raise ValueError(f"Unknown path type: {path_type}")


def interpolate_trajectory(q_coarse, t_coarse, t_fine, method="cubic"):
    """
    Interpolate joint-space trajectory using various methods.
    
    Args:
        q_coarse: (N_coarse, n_joints) array of waypoint joint angles
        t_coarse: (N_coarse,) time values for waypoints (normalized 0 to 1)
        t_fine: (N_fine,) time values for interpolated trajectory
        method: "cubic", "quintic", "lspb", or "bangbang"
    
    Returns:
        q_fine: (N_fine, n_joints) interpolated joint angles
    """
    n_points, n_joints = q_coarse.shape
    n_fine = len(t_fine)
    q_fine = np.zeros((n_fine, n_joints))
    
    if method == "cubic":
        print("Using Cubic Spline interpolation (smooth velocity, discontinuous acceleration)")
        for j in range(n_joints):
            cs = CubicSpline(t_coarse, q_coarse[:, j], bc_type='clamped')
            q_fine[:, j] = cs(t_fine)
    
    elif method == "quintic":
        print("Using Quintic polynomial interpolation (smooth velocity and acceleration)")
        for j in range(n_joints):
            q_fine[:, j] = quintic_interpolation(t_coarse, q_coarse[:, j], t_fine)
    
    elif method == "lspb":
        print("Using Linear Segment with Parabolic Blends (LSPB)")
        for j in range(n_joints):
            q_fine[:, j] = lspb_interpolation(t_coarse, q_coarse[:, j], t_fine)
    
    elif method == "bangbang":
        print("Using Bang-Bang (minimum time with max acceleration)")
        for j in range(n_joints):
            q_fine[:, j] = bangbang_interpolation(t_coarse, q_coarse[:, j], t_fine)
    
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    # Unwrap to ensure continuity
    for j in range(n_joints):
        q_fine[:, j] = np.unwrap(q_fine[:, j])
    
    return q_fine


def quintic_interpolation(t_coarse, q_coarse, t_fine):
    """Quintic polynomial interpolation between waypoints."""
    q_fine = np.zeros(len(t_fine))
    
    for i in range(len(t_coarse) - 1):
        mask = (t_fine >= t_coarse[i]) & (t_fine <= t_coarse[i+1])
        if not np.any(mask):
            continue
        
        t0, t1 = t_coarse[i], t_coarse[i+1]
        q0, q1 = q_coarse[i], q_coarse[i+1]
        tau = (t_fine[mask] - t0) / (t1 - t0)
        
        a0 = q0
        a1 = 0
        a2 = 0
        a3 = 10 * (q1 - q0)
        a4 = -15 * (q1 - q0)
        a5 = 6 * (q1 - q0)
        
        q_fine[mask] = a0 + a1*tau + a2*tau**2 + a3*tau**3 + a4*tau**4 + a5*tau**5
    
    return q_fine


def lspb_interpolation(t_coarse, q_coarse, t_fine):
    """Linear Segment with Parabolic Blends - trapezoidal velocity profile."""
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
    """Bang-bang control - triangular velocity profile (switch at midpoint)."""
    q_fine = np.zeros(len(t_fine))
    
    for i in range(len(t_coarse) - 1):
        mask = (t_fine >= t_coarse[i]) & (t_fine <= t_coarse[i+1])
        if not np.any(mask):
            continue
        
        t0, t1 = t_coarse[i], t_coarse[i+1]
        q0, q1 = q_coarse[i], q_coarse[i+1]
        T = t1 - t0
        
        tau = (t_fine[mask] - t0) / T
        q_diff = q1 - q0
        
        accel_mask = tau < 0.5
        tau_a = tau[accel_mask]
        q_fine[np.where(mask)[0][accel_mask]] = q0 + 2 * q_diff * tau_a**2
        
        decel_mask = tau >= 0.5
        tau_d = tau[decel_mask]
        q_fine[np.where(mask)[0][decel_mask]] = q1 - 2 * q_diff * (1 - tau_d)**2
    
    return q_fine


def fix_mjcf_mesh_paths(original_mjcf, mesh_dir, start_pos=None, end_pos=None):
    """
    Create a temporary MJCF file with absolute mesh paths and trajectory markers.
    """
    tree = ET.parse(original_mjcf)
    root = tree.getroot()
    
    abs_mesh_dir = os.path.abspath(mesh_dir)
    
    for mesh in root.iter('mesh'):
        if 'file' in mesh.attrib:
            filename = os.path.basename(mesh.attrib['file'])
            mesh.attrib['file'] = os.path.join(abs_mesh_dir, filename)
    
    if start_pos is not None or end_pos is not None:
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
    
    tmpdir = tempfile.mkdtemp()
    tmp_mjcf = os.path.join(tmpdir, "robot_temp.xml")
    tree.write(tmp_mjcf)
    
    return tmp_mjcf, tmpdir


# --------- load models -----------
print("Loading URDF chain...")
chain = Chain.from_urdf_file(URDF, base_elements=BASE_ELEMENT, base_element_type='link', last_link_vector=[0,0,0])

print(f"Fixing MJCF mesh paths and adding markers...")
print(f"  Start point (GREEN): {cart_start}")
print(f"  End point (RED): {cart_end}")
tmp_mjcf, tmpdir = fix_mjcf_mesh_paths(MJCF, MESH_DIR, cart_start, cart_end)

try:
    model = mujoco.MjModel.from_xml_path(tmp_mjcf)
    data = mujoco.MjData(model)
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

active_mask = np.array(chain.active_links_mask, dtype=bool)
active_names = [link.name for i, link in enumerate(chain.links) if active_mask[i]]
n_active = len(active_names)
print(f"Active joint names (ikpy order): {active_names}")
print(f"Number of active joints: {n_active}")


def ik_for_point(pt):
    q_full = chain.inverse_kinematics(pt)
    q_active = np.array([q_full[i] for i, m in enumerate(chain.active_links_mask) if m], dtype=float)
    return q_active


# ------------- Build better Cartesian path ----------------
print(f"\nGenerating {coarse_points} waypoints using {path_type} path...")
cart_points = generate_cartesian_path(cart_start, cart_end, coarse_points, path_type)

# Solve IK for each waypoint
print("Solving inverse kinematics for waypoints...")
q_coarse = []
failed_points = []
for i, pt in enumerate(cart_points):
    try:
        q = ik_for_point(pt)
        q_coarse.append(q)
    except Exception as e:
        print(f"  Warning: IK failed for waypoint {i} at {pt}")
        failed_points.append(i)
        # Use previous solution if available, otherwise zeros
        if len(q_coarse) > 0:
            q_coarse.append(q_coarse[-1])
        else:
            q_coarse.append(np.zeros(n_active))

q_coarse = np.stack(q_coarse)
print(f"q_coarse shape: {q_coarse.shape}")
if failed_points:
    print(f"Warning: {len(failed_points)} IK solutions failed")


# ------------- Fix branch discontinuities & unwrap -------------
print("Unwrapping joint angles to remove discontinuities...")
for j in range(n_active):
    q_coarse[:, j] = np.unwrap(q_coarse[:, j])


def read_current_active_from_mujoco():
    muj_names = []
    for qi in range(model.nq):
        muj_names.append(model.joint(qi).name)
    current = []
    for name in active_names:
        if name in muj_names:
            idx = muj_names.index(name)
            current.append(float(data.qpos[idx]))
        else:
            current.append(0.0)
    return np.array(current, dtype=float)


q_now = read_current_active_from_mujoco()
print(f"Current robot pose: {q_now}")

for j in range(n_active):
    diff = q_now[j] - q_coarse[0, j]
    k = np.round(diff / (2*np.pi))
    q_coarse[:, j] += k * 2*np.pi

print("Aligned trajectory to current pose")


# ------------- Interpolate in joint-space -------------
print(f"\nInterpolating trajectory to {frames} frames using {interp_type} interpolation...")
t_coarse = np.linspace(0.0, 1.0, coarse_points)
t_fine = np.linspace(0.0, 1.0, frames)

q_fine = interpolate_trajectory(q_coarse, t_coarse, t_fine, method=interp_type)


# ------------- apply trajectory with velocity clamp -------------
muj_names = [model.joint(i).name for i in range(model.njnt)]

def active_to_full_qpos(active_q):
    qpos = list(data.qpos.copy())
    for name, val in zip(active_names, active_q):
        if name in muj_names:
            idx = muj_names.index(name)
            qpos[idx] = float(val)
    return np.array(qpos, dtype=float)


print(f"\nStarting trajectory playback ({frames} frames at {dt}s each = {frames*dt:.1f}s total)...")
print(f"Max joint velocity: {max_joint_vel} rad/s")

ee_path = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    prev_active = read_current_active_from_mujoco()
    
    # Initialize scene for path visualization
    viewer.user_scn.ngeom = 0
    
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
        
        # Get end-effector position
        ee_body_id = model.nbody - 1
        ee_pos = data.xpos[ee_body_id].copy()
        ee_path.append(ee_pos)
        
        # Draw path with cylinders connecting consecutive points for better visualization
        if frame_idx > 0 and frame_idx % 3 == 0:
            prev_pos = ee_path[-4] if len(ee_path) >= 4 else ee_path[-2]
            
            # Add sphere at current position
            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.008, 0, 0],
                    pos=ee_pos,
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0, 1, 1, 0.8])
                )
                viewer.user_scn.ngeom += 1
            
            # Add cylinder connecting to previous point
            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                midpoint = (ee_pos + prev_pos) / 2
                direction = ee_pos - prev_pos
                length = np.linalg.norm(direction)
                
                if length > 0.001:
                    # Compute rotation matrix to align cylinder with direction
                    direction = direction / length
                    z_axis = np.array([0, 0, 1])
                    
                    if np.abs(np.dot(direction, z_axis)) < 0.999:
                        rot_axis = np.cross(z_axis, direction)
                        rot_axis = rot_axis / np.linalg.norm(rot_axis)
                        angle = np.arccos(np.dot(z_axis, direction))
                        
                        # Rodrigues rotation formula
                        K = np.array([[0, -rot_axis[2], rot_axis[1]],
                                     [rot_axis[2], 0, -rot_axis[0]],
                                     [-rot_axis[1], rot_axis[0], 0]])
                        rot_mat = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
                    else:
                        rot_mat = np.eye(3) if direction[2] > 0 else -np.eye(3)
                    
                    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                        size=[0.004, length/2, 0],
                        pos=midpoint,
                        mat=rot_mat.flatten(),
                        rgba=np.array([0, 0.8, 1, 0.6])
                    )
                    viewer.user_scn.ngeom += 1
        
        viewer.sync()
        time.sleep(dt)
        
        prev_active = new_active.copy()
        
        if (frame_idx + 1) % 50 == 0:
            print(f"  Frame {frame_idx + 1}/{frames} ({100*(frame_idx+1)/frames:.1f}%)")

ee_path = np.array(ee_path)
print(f"\nEnd-effector path statistics:")
print(f"  Total path length: {np.sum(np.linalg.norm(np.diff(ee_path, axis=0), axis=1)):.4f} m")
print(f"  Start position: {ee_path[0]}")
print(f"  End position: {ee_path[-1]}")
print(f"  Error from target start: {np.linalg.norm(ee_path[0] - cart_start):.6f} m")
print(f"  Error from target end: {np.linalg.norm(ee_path[-1] - cart_end):.6f} m")

print("\nTrajectory playback finished.")