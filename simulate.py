"""Simulate trajectory playback for the Puma560 robot.

This script plans a Cartesian path, solves inverse kinematics for
coarse waypoints, interpolates a smooth joint-space trajectory, and
plays it back in MuJoCo. It can run headless for faster rendering and
record an offscreen video if mediapy is installed.

Key concepts:
- Path generation: delegated to `path_planning.generate_cartesian_path`.
- IK solving: uses `ikpy.Chain` to solve for joint angles at waypoints.
- Trajectory interpolation: delegated to `trajectory_planning.interpolate_trajectory`.
- Rendering/recording: uses MuJoCo renderer for offscreen capture.

Only comments and docstrings were added; no runtime logic was modified.
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from ikpy.chain import Chain
import argparse
from collections import deque

# mediapy is optional; it is only required when the user requests video
# output. We set a flag so the rest of the script can gracefully disable
# video capture when mediapy isn't present.
try:
    import mediapy as media
    MEDIAPY_AVAILABLE = True
except ImportError:
    print("Warning: mediapy not installed. Video recording will be disabled.")
    print("Install with: pip install mediapy")
    MEDIAPY_AVAILABLE = False

# Project helpers: path generation and trajectory interpolation routines
from path_planning import generate_cartesian_path
from trajectory_planning import interpolate_trajectory

# --------- Parse command-line arguments ----------
# The script supports choosing the path planner, interpolation method,
# output video settings, and whether to run headless (no interactive viewer).
parser = argparse.ArgumentParser(description='Robot trajectory planning and execution')
parser.add_argument('--path-type', '-p', type=str, default='rrt',
                    choices=['linear', 'arc', 'circular', 'parabolic', 'rrt'],
                    help='Path planning method (default: rrt)')
parser.add_argument('--interp-type', '-i', type=str, default='bangbang',
                    choices=['cubic', 'quintic', 'lspb', 'bangbang'],
                    help='Trajectory interpolation method (default: bangbang)')
parser.add_argument('--output', '-o', type=str, default=None,
                    help='Output video filename (default: auto-generated based on path and interp type)')
parser.add_argument('--fps', type=int, default=30,
                    help='Video frames per second (default: 30)')
parser.add_argument('--width', type=int, default=1280,
                    help='Video width in pixels (default: 1280)')
parser.add_argument('--height', type=int, default=720,
                    help='Video height in pixels (default: 720)')
parser.add_argument('--no-display', action='store_true',
                    help='Run headless without viewer (faster rendering)')

args = parser.parse_args()

# --------- Configuration ----------
# File locations for URDF / MJCF and mesh directory. The URDF is used
# for inverse kinematics with ikpy while the MJCF is used for MuJoCo
# visualization and rendering (meshes in MJCF need absolute paths).
URDF = "puma560_description/urdf/puma560_robot.urdf"
MJCF = "puma560_description/urdf/puma560_robot.xml"
MESH_DIR = "puma560_description/meshes"
BASE_ELEMENT = ['link1']

# Trajectory parameters (tweak these to change behavior)
cart_start = np.array([0.3, -0.5, 0.2])
cart_end = np.array([0.4, -0.6, 1])
path_type = args.path_type
interp_type = args.interp_type
coarse_points = 15       # number of IK solves along path (coarse)
frames = 6000            # total frames to play out (simulation steps)
dt = 0.001               # seconds per simulation step (used for sleep rate)
max_joint_vel = 1.5      # rad/s (per joint) — used to cap per-step motion
use_pd_control = False   # placeholder: if True, output qpos to be used by actuators
show_planned_path = True # display planned Cartesian waypoints in MJCF

# Video settings: only enable recording if mediapy is available
save_video = MEDIAPY_AVAILABLE
if not MEDIAPY_AVAILABLE and args.output:
    print("Warning: Cannot save video without mediapy. Install with: pip install mediapy")
    save_video = False

video_fps = args.fps
video_width = args.width
video_height = args.height

if args.output:
    video_filename = args.output
else:
    video_filename = f"videos/trajectory_{path_type}_{interp_type}.mp4"

# Ensure videos directory exists
os.makedirs("videos", exist_ok=True)

# Calculate skip rate to achieve desired video FPS
sim_fps = 1.0 / dt
frame_skip = max(1, int(sim_fps / video_fps))

print(f"\n{'='*60}")
print(f"CONFIGURATION")
print(f"{'='*60}")
print(f"  Path type: {path_type}")
print(f"  Interpolation: {interp_type}")
print(f"  Video output: {video_filename}")
print(f"  Video resolution: {video_width}x{video_height}")
print(f"  Video FPS: {video_fps}")
print(f"  Frame skip: {frame_skip} (capturing every {frame_skip} frames)")
print(f"  Headless mode: {args.no_display}")


def fix_mjcf_mesh_paths(original_mjcf, mesh_dir, start_pos=None, end_pos=None, planned_path=None, offscreen_width=1280, offscreen_height=720):
    """Create a temporary MJCF with absolute mesh paths and optional visual markers.

    MuJoCo's MJCF mesh 'file' attributes are commonly relative; when
    rendering offscreen we prefer absolute paths. This function also
    optionally injects small visual markers for the start/end points and
    the planned Cartesian waypoints (spheres) and connects them with
    cylinders so the path is visible in the scene.

    Returns:
        tmp_mjcf (str): path to a temporary MJCF file to load into MuJoCo.
        tmpdir (str): directory containing the temporary MJCF (caller should cleanup).
    """
    tree = ET.parse(original_mjcf)
    root = tree.getroot()

    # Ensure a <visual><global .../> block exists so we can set offscreen size
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')

    global_visual = visual.find('global')
    if global_visual is None:
        global_visual = ET.SubElement(visual, 'global')

    # Configure offscreen buffer size for renderer
    global_visual.set('offwidth', str(offscreen_width))
    global_visual.set('offheight', str(offscreen_height))

    # Convert mesh file attributes to absolute paths pointing into mesh_dir
    abs_mesh_dir = os.path.abspath(mesh_dir)
    for mesh in root.iter('mesh'):
        if 'file' in mesh.attrib:
            filename = os.path.basename(mesh.attrib['file'])
            mesh.attrib['file'] = os.path.join(abs_mesh_dir, filename)

    # Ensure worldbody exists so we can attach markers
    worldbody = root.find('worldbody')
    if worldbody is None:
        worldbody = ET.SubElement(root, 'worldbody')

    # Add simple sphere geoms for start/end
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

    # Add small spheres for planned waypoints and connect consecutive ones
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

        # Connect consecutive waypoints with thin cylinders for visualization
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

                # Compute quaternion for cylinder orientation (fallback to identity)
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
print(f"\n{'='*60}")
print(f"LOADING MODELS")
print(f"{'='*60}")
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
tmp_mjcf, tmpdir = fix_mjcf_mesh_paths(
    MJCF, MESH_DIR, cart_start, cart_end, 
    cart_points if show_planned_path else None,
    offscreen_width=video_width,
    offscreen_height=video_height
)

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


muj_names = [model.joint(i).name for i in range(model.njnt)]

def active_to_full_qpos(active_q):
    qpos = np.zeros(model.nq)
    for name, val in zip(active_names, active_q):
        if name in muj_names:
            idx = muj_names.index(name)
            qpos[idx] = float(val)
    return qpos


# ========== SET INITIAL CONFIGURATION TO START POINT ==========
print(f"\n{'='*60}")
print(f"SETTING INITIAL CONFIGURATION")
print(f"{'='*60}")

# Calculate IK for the starting point
print(f"Computing IK for start position: {cart_start}")
q_start = ik_for_point(cart_start)
print(f"Initial joint configuration: {q_start}")

# Set the robot to this configuration
data.qpos[:] = active_to_full_qpos(q_start)
mujoco.mj_forward(model, data)

# Verify the end-effector is at the start position
ee_body_id = model.nbody - 1
ee_pos_verify = data.xpos[ee_body_id].copy()
ee_error = np.linalg.norm(ee_pos_verify - cart_start)
print(f"End-effector position after setting: {ee_pos_verify}")
print(f"Error from desired start: {ee_error:.6f} m")

if ee_error > 0.01:
    print(f"⚠ Warning: Large positioning error ({ee_error:.4f} m)")
else:
    print(f"✓ Robot positioned at start point successfully")

# Now the trajectory starts from this configuration (no need to prepend)
# The first waypoint in q_coarse is already the start, so we use it as-is

# Create time vectors
n_coarse = q_coarse.shape[0]
t_coarse = np.linspace(0.0, 1.0, n_coarse)
t_fine = np.linspace(0.0, 1.0, frames)

# Interpolate using chosen method
print(f"\nInterpolating trajectory to {frames} frames using {interp_type}...")
q_fine = interpolate_trajectory(q_coarse, t_coarse, t_fine, method=interp_type)

print(f"\n{'='*60}")
print(f"EXECUTION PHASE")
print(f"{'='*60}")
print(f"Total time: {frames*dt:.1f}s")
print(f"Max joint velocity: {max_joint_vel} rad/s")
if save_video:
    print(f"Recording video: {video_filename}\n")

ee_path_history = deque(maxlen=5000)
video_frames = []

# Setup renderer for offscreen rendering
renderer = None
if save_video:
    try:
        renderer = mujoco.Renderer(model, height=video_height, width=video_width)
        print(f"Renderer initialized: {video_width}x{video_height}")
    except Exception as e:
        print(f"Warning: Could not initialize renderer: {e}")
        print("Video recording will be disabled")
        save_video = False

# Reset robot to start configuration before execution
data.qpos[:] = active_to_full_qpos(q_start)
mujoco.mj_forward(model, data)

# Choose viewer mode
if args.no_display:
    # Headless mode - faster rendering
    print("Running in headless mode...")
    prev_active = q_start.copy()
    
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
        
        # Capture frame for video
        if save_video and renderer is not None and frame_idx % frame_skip == 0:
            try:
                renderer.update_scene(data)
                pixels = renderer.render()
                video_frames.append(pixels.copy())
            except Exception as e:
                if frame_idx == 0:
                    print(f"Warning: Video capture failed: {e}")
                    print(f"Error details: {str(e)}")
                    save_video = False
        
        prev_active = new_active.copy()
        
        if (frame_idx + 1) % 500 == 0:
            print(f"  Progress: {100*(frame_idx+1)/frames:.1f}% ({frame_idx+1}/{frames} frames, {len(video_frames)} video frames captured)")
    
    if renderer is not None:
        renderer.close()

else:
    # Interactive mode with viewer
    print("Running with interactive viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        prev_active = q_start.copy()
        
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
            
            # Draw execution trace (cyan)
            if len(ee_path_history) >= 2:
                path_list = list(ee_path_history)
                for i in range(len(path_list) - 1):
                    if i % 2 == 0:
                        p1 = path_list[i]
                        p2 = path_list[i + 1]
                        
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
                                    rgba=np.array([0, 1, 1, alpha_val])
                                )
                                viewer.user_scn.ngeom += 1
            
            # Capture frame for video
            if save_video and renderer is not None and frame_idx % frame_skip == 0:
                try:
                    renderer.update_scene(data)
                    pixels = renderer.render()
                    video_frames.append(pixels.copy())
                except Exception as e:
                    if frame_idx == 0:
                        print(f"Warning: Video capture failed: {e}")
                        save_video = False
            
            viewer.sync()
            time.sleep(dt)
            prev_active = new_active.copy()
            
            if (frame_idx + 1) % 100 == 0:
                print(f"  Progress: {100*(frame_idx+1)/frames:.1f}% ({frame_idx+1}/{frames} frames, {len(video_frames)} video frames)")
    
    if save_video and renderer is not None:
        renderer.close()

# Save video
if save_video and len(video_frames) > 0:
    print(f"\n{'='*60}")
    print(f"SAVING VIDEO")
    print(f"{'='*60}")
    print(f"Writing {len(video_frames)} frames to {video_filename}...")
    try:
        # Ensure frames directory exists
        os.makedirs(os.path.dirname(video_filename), exist_ok=True)
        
        # Convert to numpy array for mediapy
        video_array = np.array(video_frames, dtype=np.uint8)
        print(f"Video array shape: {video_array.shape}")
        print(f"Video array dtype: {video_array.dtype}")
        
        media.write_video(video_filename, video_array, fps=video_fps)
        print(f"✓ Video saved successfully!")
        
        if os.path.exists(video_filename):
            file_size = os.path.getsize(video_filename) / (1024 * 1024)
            print(f"  File size: {file_size:.2f} MB")
            print(f"  Location: {os.path.abspath(video_filename)}")
        else:
            print(f"⚠ Warning: Video file not found after writing")
    except Exception as e:
        print(f"✗ Error saving video: {e}")
        print(f"  Frames captured: {len(video_frames)}")
        print(f"  Frame shape: {video_frames[0].shape if video_frames else 'N/A'}")
        import traceback
        traceback.print_exc()
elif save_video:
    print(f"\n⚠ Warning: No video frames were captured")
    print(f"  Check that renderer was initialized correctly")

print(f"\n✓ Trajectory playback finished!")