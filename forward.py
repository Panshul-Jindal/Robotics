import mujoco
import mujoco.viewer
import numpy as np
import time

xml_path = "puma560_description/urdf/puma560_robot.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0
    while viewer.is_running():
        # simple joint oscillation
        data.qpos[0] = 1 * np.sin(5 * t)
        data.qpos[1] = 0.4 * np.sin(1.5 * t)
        data.qpos[2] = 0.3 * np.sin(1.2 * t)

        mujoco.mj_forward(model, data)

        viewer.sync()
        time.sleep(0.01)
        t += 0.01
