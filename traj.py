import numpy as np
import mujoco
import mujoco.viewer
import time
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from ikpy.chain import Chain
from scipy.interpolate import CubicSpline

# --------- CONFIG ----------
URDF = "puma560_description/urdf/puma560_robot.urdf"
MJCF = "puma560_description/urdf/puma560_robot.xml"
MESH_DIR = "puma560_description/meshes"

# Which link was the base you used earlier
BASE_ELEMENT = ['link1']

# trajectory params
cart_start = np.array([0.3, 0.2, 0.2])
cart_end   = np.array([-0.5, 0.0, 0.4])
coarse_points = 12        # number of IK solves along path (coarse)
frames =1000             # total frames to play out (fine)
dt = 0.01                 # seconds per render step (sleep); adjust with viewer rate
max_joint_vel = 1.5       # rad/s (per joint) — tune to reduce jerk
use_pd_control = False    # if True we'll output desired qpos and you can adapt to your actuators


def fix_mjcf_mesh_paths(original_mjcf, mesh_dir, start_pos=None, end_pos=None):
    """
    Create a temporary MJCF file with absolute mesh paths and trajectory markers.
    """
    # Parse the XML
    tree = ET.parse(original_mjcf)
    root = tree.getroot()
    
    # Get absolute path to mesh directory
    abs_mesh_dir = os.path.abspath(mesh_dir)
    
    # Find all mesh elements and update their file paths
    for mesh in root.iter('mesh'):
        if 'file' in mesh.attrib:
            filename = os.path.basename(mesh.attrib['file'])
            mesh.attrib['file'] = os.path.join(abs_mesh_dir, filename)
    
    # Add visual markers for start and end positions
    if start_pos is not None or end_pos is not None:
        # Find or create worldbody
        worldbody = root.find('worldbody')
        if worldbody is None:
            worldbody = ET.SubElement(root, 'worldbody')
        
        # Add start marker (green sphere)
        if start_pos is not None:
            start_body = ET.SubElement(worldbody, 'body', name='start_marker', 
                                      pos=f"{start_pos[0]} {start_pos[1]} {start_pos[2]}")
            ET.SubElement(start_body, 'geom', type='sphere', size='0.03', 
                         rgba='0 1 0 0.8', contype='0', conaffinity='0')
        
        # Add end marker (red sphere)
        if end_pos is not None:
            end_body = ET.SubElement(worldbody, 'body', name='end_marker',
                                    pos=f"{end_pos[0]} {end_pos[1]} {end_pos[2]}")
            ET.SubElement(end_body, 'geom', type='sphere', size='0.03',
                         rgba='1 0 0 0.8', contype='0', conaffinity='0')
    
    # Create temporary file
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
    # Clean up temp directory after loading
    shutil.rmtree(tmpdir, ignore_errors=True)

# active joint names in ikpy order
active_mask = np.array(chain.active_links_mask, dtype=bool)
active_names = [link.name for i, link in enumerate(chain.links) if active_mask[i]]
n_active = len(active_names)
print(f"Active joint names (ikpy order): {active_names}")
print(f"Number of active joints: {n_active}")


# helper: compute ikpy active joint vector for a cartesian point
def ik_for_point(pt):
    q_full = chain.inverse_kinematics(pt)   # full returned vector
    q_active = np.array([q_full[i] for i, m in enumerate(chain.active_links_mask) if m], dtype=float)
    return q_active


# ------------- Build coarse joint solutions ----------------
print(f"\nGenerating {coarse_points} waypoints from {cart_start} to {cart_end}...")
cart_points = np.linspace(cart_start, cart_end, coarse_points)
q_coarse = np.stack([ik_for_point(p) for p in cart_points])    # shape (coarse_points, n_active)
print(f"q_coarse shape: {q_coarse.shape}")


# ------------- Fix branch discontinuities & unwrap -------------
print("Unwrapping joint angles to remove discontinuities...")
# 1) Unwrap along each joint axis to remove 2pi jumps
for j in range(n_active):
    q_coarse[:, j] = np.unwrap(q_coarse[:, j])


# 2) If you have a previous pose (e.g., current robot q), align first coarse point to it:
# get current q_active in mujoco order (map by names)
def read_current_active_from_mujoco():
    # build mapping from mujoco joint names to qpos index
    muj_names = []
    for qi in range(model.nq):
        muj_names.append(model.joint(qi).name)
    # build ikpy active vector from current data.qpos
    current = []
    for name in active_names:
        if name in muj_names:
            idx = muj_names.index(name)
            current.append(float(data.qpos[idx]))
        else:
            # fallback zero
            current.append(0.0)
    return np.array(current, dtype=float)


q_now = read_current_active_from_mujoco()
print(f"Current robot pose: {q_now}")

# compute shift to make q_coarse[0] nearest q_now (add multiples of 2pi)
for j in range(n_active):
    diff = q_now[j] - q_coarse[0, j]
    k = np.round(diff / (2*np.pi))
    q_coarse[:, j] += k * 2*np.pi

print("Aligned trajectory to current pose")


# ------------- Interpolate in joint-space (cubic spline) -------------
print(f"\nInterpolating trajectory to {frames} frames using cubic splines...")
t_coarse = np.linspace(0.0, 1.0, coarse_points)
t_fine = np.linspace(0.0, 1.0, frames)
q_fine = np.zeros((frames, n_active))

for j in range(n_active):
    cs = CubicSpline(t_coarse, q_coarse[:, j], bc_type='clamped')
    q_fine[:, j] = cs(t_fine)

# optional: ensure continuity by unwrapping q_fine per joint
for j in range(n_active):
    q_fine[:, j] = np.unwrap(q_fine[:, j])


# ------------- apply trajectory with velocity clamp -------------
# build name->q mapping helper to create full qpos vector in mujoco order
muj_names = [model.joint(i).name for i in range(model.njnt)]

def active_to_full_qpos(active_q):
    # preserve other joints in data.qpos
    qpos = list(data.qpos.copy())
    for name, val in zip(active_names, active_q):
        if name in muj_names:
            idx = muj_names.index(name)
            qpos[idx] = float(val)
    return np.array(qpos, dtype=float)


# apply incremental smoothing on each step: clamp delta to max_joint_vel * dt
print(f"\nStarting trajectory playback ({frames} frames at {dt}s each = {frames*dt:.1f}s total)...")
print(f"Max joint velocity: {max_joint_vel} rad/s")

with mujoco.viewer.launch_passive(model, data) as viewer:
    prev_active = read_current_active_from_mujoco()
    
    for frame_idx in range(frames):
        desired_active = q_fine[frame_idx]
        
        # clamp per-joint step
        delta = desired_active - prev_active
        max_step = max_joint_vel * dt
        too_big = np.abs(delta) > max_step
        
        if np.any(too_big):
            # scale down only those that exceed
            delta[too_big] = np.sign(delta[too_big]) * max_step
        
        new_active = prev_active + delta

        # optionally low-pass filter: new = alpha*new + (1-alpha)*prev (small alpha gives smoother)
        alpha = 0.9
        new_active = alpha * new_active + (1.0 - alpha) * prev_active

        # convert to mujoco qpos and write
        qpos_full = active_to_full_qpos(new_active)
        data.qpos[:] = qpos_full
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(dt)
        
        prev_active = new_active.copy()
        
        # Progress indicator every 50 frames
        if (frame_idx + 1) % 50 == 0:
            print(f"  Frame {frame_idx + 1}/{frames} ({100*(frame_idx+1)/frames:.1f}%)")

print("\nTrajectory playback finished.")