import mujoco
import os
urdf_path = "puma560_description/urdf/puma560_robot.urdf"  
output_mjcf_path = "puma560_description/urdf/puma560_robot.xml"
try:
    # Load the URDF model
    model = mujoco.MjModel.from_xml_path(urdf_path)
    # Save the model as MJCF
    mujoco.mj_saveLastXML(output_mjcf_path, model)
    print(f"Successfully converted {urdf_path} to {output_mjcf_path}")
except Exception as e:
    print(f"Error converting URDF to MJCF: {e}")