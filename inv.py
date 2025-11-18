# joint_interp_visual.py
import numpy as np
import mujoco, mujoco.viewer, time
from ikpy.chain import Chain
from scipy.interpolate import CubicSpline   # pip install scipy

URDF = "puma560_description/urdf/puma560_robot.urdf"
MJCF = "puma560_description/urdf/puma560_robot.xml"
chain = Chain.from_urdf_file(URDF, base_elements=['link1'], base_element_type='link', last_link_vector=[0,0,0])
model = mujoco.MjModel.from_xml_path(MJCF)
data = mujoco.MjData(model)

# compute a coarse set of IK solutions along the Cartesian path
coarse_N = 10
cart_points = np.linspace([0.3,0.2,0.2], [0.5,0.0,0.4], coarse_N)
q_list = []
for p in cart_points:
    q_full = chain.inverse_kinematics(p)
    q_active = [q_full[i] for i,m in enumerate(chain.active_links_mask) if m]
    q_list.append(q_active)
q_array = np.stack(q_list)   # shape (coarse_N, n_joints)

# unwrap angles to avoid 2pi jumps (optional)
for j in range(q_array.shape[1]):
    q_array[:, j] = np.unwrap(q_array[:, j])

# create time parameterization for spline
t_coarse = np.linspace(0, 1, coarse_N)
t_fine = np.linspace(0, 1, 200)  # final # of frames
qs_fine = np.zeros((len(t_fine), q_array.shape[1]))
for j in range(q_array.shape[1]):
    cs = CubicSpline(t_coarse, q_array[:, j], bc_type='clamped')
    qs_fine[:, j] = cs(t_fine)

# helper to map joint vector (ikpy order) to mujoco qpos
active_names = [link.name for i,link in enumerate(chain.links) if chain.active_links_mask[i]]
def apply_joint_vector_to_qpos(qvec_active):
    name_to_q = dict(zip(active_names, qvec_active))
    qpos = list(data.qpos)
    for qi in range(model.nq):
        jname = model.joint(qi).name
        if jname in name_to_q:
            qpos[qi] = float(name_to_q[jname])
    data.qpos[:] = np.array(qpos)

with mujoco.viewer.launch_passive(model, data) as viewer:
    for qvec in qs_fine:
        apply_joint_vector_to_qpos(qvec)
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.01)
