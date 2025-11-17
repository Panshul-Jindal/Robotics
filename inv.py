# follow_trajectory_visual.py
import mujoco
import mujoco.viewer
import numpy as np
import time
from ikpy.chain import Chain

URDF = "puma560_description/urdf/puma560_robot.urdf"
MJCF = "puma560_description/urdf/puma560_robot.xml"

# load ikpy chain (use the same base element you used before)
chain = Chain.from_urdf_file(URDF, base_elements=['link1'], base_element_type='link', last_link_vector=[0,0,0])

model = mujoco.MjModel.from_xml_path(MJCF)
data = mujoco.MjData(model)

# task-space waypoints (straight line)
start = np.array([0.3, 0.2, 0.2])
end   = np.array([0.5, 0.0, 0.4])
N = 200
waypoints = np.linspace(start, end, N)

# helper: compute ik -> qpos vector in mujoco order
def ik_to_mujoco_qpos(target_xyz):
    q_full = chain.inverse_kinematics(target_xyz)       # full returned vector
    # extract active values using mask
    q_active = [q_full[i] for i, m in enumerate(chain.active_links_mask) if m]
    # active_joint_names (in ikpy order)
    active_names = [link.name for i, link in enumerate(chain.links) if chain.active_links_mask[i]]
    # build name->value
    name_to_q = dict(zip(active_names, q_active))
    # create qpos in mujoco model order
    qpos = list(data.qpos)   # preserve length
    # model.joint(i).name provides joint names in this mujoco Python API
    # we fill only for the joints that ikpy solved (j1..j6)
    for qi in range(model.nq):
        try:
            jname = model.joint(qi).name
        except Exception:
            # fallback: if model.joint API differs you can read model.joint_names
            jname = None
        if jname and jname in name_to_q:
            qpos[qi] = float(name_to_q[jname])
    return np.array(qpos, dtype=float)

# run viewer and follow waypoints
with mujoco.viewer.launch_passive(model, data) as viewer:
    for point in waypoints:
        qpos_target = ik_to_mujoco_qpos(point)
        data.qpos[:] = qpos_target
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.01)
