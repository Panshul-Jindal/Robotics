import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
import time

# ==============================================================================
# |                 CORE ROBOTICS MOTION FUNCTIONS                           |
# ==============================================================================

def forward_kinematics(model, data, joint_angles, ee_name):
    """
    Performs forward kinematics to find the end-effector pose.

    Args:
        model: MuJoCo MjModel object.
        data: MuJoCo MjData object.
        joint_angles (np.array): The joint angles for the robot.
        ee_name (str): The name of the end-effector body.

    Returns:
        tuple: A tuple containing:
            - ee_pos (np.array): The Cartesian position (x, y, z).
            - ee_quat (np.array): The orientation as a quaternion (w, x, y, z).
    """
    data.qpos[:] = joint_angles
    mujoco.mj_forward(model, data)
    
    ee_pos = data.body(ee_name).xpos.copy()
    ee_mat = data.body(ee_name).xmat.copy().reshape(3, 3)
    ee_quat_scipy = Rotation.from_matrix(ee_mat).as_quat() # x,y,z,w
    ee_quat_mj = np.array([ee_quat_scipy[3], ee_quat_scipy[0], ee_quat_scipy[1], ee_quat_scipy[2]]) # w,x,y,z
    
    return ee_pos, ee_quat_mj

def inverse_kinematics(model, data, target_pos, target_quat, ee_name, initial_qpos=None):
    """
    Performs inverse kinematics to find the required joint angles.

    Args:
        model: MuJoCo MjModel object.
        data: MuJoCo MjData object.
        target_pos (np.array): The target Cartesian position (x, y, z).
        target_quat (np.array): The target orientation quaternion (w, x, y, z).
        ee_name (str): The name of the end-effector body.
        initial_qpos (np.array, optional): A seed for the IK solver. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - success (bool): True if a solution was found.
            - q_sol (np.array): The solved joint angles.
    """
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
    if initial_qpos is not None:
        data.qpos[:] = initial_qpos

    result = mujoco.mj_ik(model, data, ee_body_id, target_pos, target_quat)
    
    # result is a tuple: (success_flag, err_pos, err_quat, ...)
    # For newer mujoco versions, it's a struct with result.success
    # This check works for both
    success = result[0] == mujoco.mjtIKResult.mjIK_SUCCESS 
    
    return success, data.qpos.copy()


# ==============================================================================
# |                     MAIN SIMULATION SCRIPT                                 |
# ==============================================================================

# --- 1. SETUP: LOAD MODEL AND INITIALIZE ---
xml_path = "puma560_description/urdf/puma560_robot.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
ee_name = "link7"

# --- 2. DEFINE THE CARTESIAN TRAJECTORY ---

# Get the home configuration (q0) - all zeros
q0 = np.zeros(model.nq)

# Use our new FK function to find the initial pose
p0, quat0 = forward_kinematics(model, data, q0, ee_name)
print(f"Initial End-Effector Position (from FK): {p0}")

# Define the target point (a simple straight line)
c = 0.3  # A smaller, safer travel distance
p1 = p0 + np.array([c, -c, c])

N = 200  # Number of waypoints
cartesian_points = np.linspace(p0, p1, N)

# --- 3. RUN THE SIMULATION LOOP ---

print("Starting trajectory simulation...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25
    viewer.cam.distance = 4.0
    viewer.cam.lookat[:] = [0.5, 0, 0.8]

    # Initialize robot at home position
    current_q = q0.copy()
    data.qpos[:] = current_q
    
    # Loop through all the waypoints
    for k in range(N):
        target_pos = cartesian_points[k]
        
        # Use our new IK function to find the joint angles for the next point
        # We use the 'current_q' as a seed for the solver to ensure a smooth path
        success, q_sol = inverse_kinematics(model, data, target_pos, quat0, ee_name, initial_qpos=current_q)
        
        if success:
            # Set the calculated joint positions as the control target
            data.ctrl[:] = q_sol[:model.nu]
            current_q = q_sol # Update the seed for the next iteration
        else:
            print(f"IK failed for waypoint {k}. Halting.")
            break # Stop if IK fails

        # Step the simulation
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01) # Small delay to make animation smoother

    print("Trajectory completed. Viewer will stay open. Close it to exit.")
    # Keep the viewer open at the end
    while viewer.is_running():
        viewer.sync()