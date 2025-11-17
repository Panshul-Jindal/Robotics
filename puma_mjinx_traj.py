import mjinx
import numpy as np
import mujoco
import mujoco.viewer
from mjinx import mjModel
xml_path = "puma560_description/urdf/puma560_robot.xml"

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Load mjinx model wrapper
mjx_model = mjinx.mjModel(xml_path)

# Create IK solver
ik = mjinx.IK(mjx_model)

# End-effector body
ee_name = "link7"

# Forward kinematics to get current pose
mujoco.mj_forward(model, data)
p0 = data.body(ee_name).xpos.copy()
quat0 = data.body(ee_name).xquat.copy()   # w,x,y,z

# Target end-effector movement
target_pos = p0 + np.array([0.2, 0.0, 0.2])

# ---- Solve IK ----
q_sol = ik.solve(
    pos=target_pos,
    quat=quat0,
    body=ee_name,
    q0=data.qpos.copy()
)

print("IK q_sol:", q_sol)

# ---- Apply to simulation ----
data.qpos[:] = q_sol
mujoco.mj_forward(model, data)

# ---- View ----
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()
