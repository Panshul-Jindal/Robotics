# --- replace ikpy usage: read-only MJCF-based IK helpers ---
import math
import numpy as np
import mujoco
def _safe_sqrt(x):
    return math.sqrt(max(0.0, x))

def _atan2(y,x):
    return math.atan2(y,x)

def infer_params_from_mjcf(mj_model, verbose=False):
    """
    Heuristically infer PUMA-like link parameters from a loaded MjModel.
    Returns dict with a1,a2,a3,d1,d2,d4,d6 (some may be None).
    This uses body positions (model.body_pos) and site positions (model.site_pos)
    if present. These are heuristics and should be checked against your MJCF.
    """
    params = {"a1": None, "a2": None, "a3": None, "d1": None, "d2": None, "d4": None, "d6": None}

    try:
        # model.nbody, model.body_pos exist in mujoco MjModel
        nbody = int(mj_model.nbody)
        # model.body_pos is shape (nbody,3) absolute positions in parent body frame
        body_pos = np.array(mj_model.body_pos).reshape((nbody, 3))
        # model.body_names list
        body_names = [mj_model.body(i).name for i in range(nbody)]

        if verbose:
            print("bodies:", body_names)

        # Try to find typical PUMA bodies by searching names
        # common names: base, link1, link2, link3, link4, link5, link6, wrist, ee
        name_map = {n.lower(): i for i, n in enumerate(body_names)}
        # heuristic: pick the first few movable links as a1,a2,a3 distances
        # choose indices in order of appearance (skip 'world' at index 0)
        body_idxs = [i for i in range(1, min(nbody, 7))]  # up to link6
        if len(body_idxs) >= 3:
            # compute vector from body i to next body (approx link vector)
            # a1: x-offset of link1
            b1, b2, b3 = body_pos[body_idxs[0]], body_pos[body_idxs[1]], body_pos[body_idxs[2]]
            # distances between consecutive body positions (norms)
            dist12 = float(np.linalg.norm(b2 - b1))
            dist23 = float(np.linalg.norm(b3 - b2))
            params["a2"] = dist12
            params["a3"] = dist23
            # estimate d1 as z of first body (if base has z offset)
            params["d1"] = float(body_pos[body_idxs[0]][2])
            # d2 often small; estimate from body1 pos y or x if available
            params["d2"] = float(body_pos[body_idxs[0]][1]) if abs(body_pos[body_idxs[0]][1])>1e-6 else 0.0

        # If MJCF defines sites (wrist or ee) use them for d4,d6
        try:
            nsite = int(mj_model.nsite)
            if nsite > 0:
                site_pos = np.array(mj_model.site_pos).reshape((nsite,3))
                site_names = [mj_model.site(i).name for i in range(nsite)]
                # look for site named 'wrist' or 'ee' to estimate d6
                for i, name in enumerate(site_names):
                    ln = name.lower()
                    if 'wrist' in ln or 'ee' in ln or 'tool' in ln:
                        # take z component as d6 heuristic
                        params["d6"] = float(site_pos[i][2])
                        break
        except Exception:
            pass

        # d4 is often link offset between joint3 and joint4; approximate using body distances
        if params["a3"] is not None and params["a2"] is not None:
            # placeholder: set d4 small if not present
            params["d4"] = params["a3"] * 0.1 if params["d4"] is None else params["d4"]

    except Exception as e:
        if verbose:
            print("infer_params_from_mjcf failed:", e)
        # return partial None-filled dict
    return params


def inverse_kinematics_custom(target_pos, target_R=None, mj_model=None, mj_data=None, params=None, verbose=False):
    """
    Closed-form PUMA-like inverse kinematics that uses only the MuJoCo model (MJCF) if given.
    - target_pos: [x,y,z]
    - target_R: optional 3x3 rotation matrix for orientation (if provided)
    - mj_model: mujoco.MjModel (required for mapping to qpos ordering & to infer params when params None)
    - mj_data: mujoco.MjData (optional)
    - params: dict to override inferred params
    Returns q_full: list of angles matching the active joints order extracted from mj_model.
    """
    p = np.asarray(target_pos, dtype=float).flatten()
    if p.size != 3:
        raise ValueError("target_pos must be length-3")

    # get params
    if params is None and mj_model is not None:
        inferred = infer_params_from_mjcf(mj_model, verbose=verbose)
    else:
        inferred = {}

    # defaults (tweak these to match your robot)
    default_params = {"a1": 0.0, "a2": 0.4318, "a3": 0.0203, "d1": 0.0, "d2": 0.0, "d4": 0.0, "d6": 0.0}
    # merge
    final = dict(default_params)
    final.update({k:v for k,v in inferred.items() if v is not None})
    if params:
        final.update(params)

    a1 = float(final.get("a1",0.0))
    a2 = float(final.get("a2",0.0))
    a3 = float(final.get("a3",0.0))
    d1 = float(final.get("d1",0.0))
    d2 = float(final.get("d2",0.0))
    d4 = float(final.get("d4",0.0))
    d6 = float(final.get("d6",0.0))

    # wrist center
    if target_R is None:
        a_axis = np.array([0.0,0.0,1.0])
    else:
        R = np.asarray(target_R)
        a_axis = R[:,2]
    pc = p - d6 * a_axis
    x_c, y_c, z_c = pc.tolist()

    r_sq = x_c**2 + y_c**2 - d2**2
    r = _safe_sqrt(r_sq)
    denom = (x_c**2 + y_c**2) if (x_c**2 + y_c**2) > 1e-12 else 1e-12
    s1 = (d2 * x_c + r * y_c) / denom
    c1 = (r * x_c - d2 * y_c) / denom
    th1 = _atan2(s1, c1)

    k1 = 2 * a2 * (r - a1)
    k2 = 2 * a2 * (z_c - d1)
    k3 = (r - a1)**2 + (z_c - d1)**2 + a2**2 - d4**2 - a3**2

    denom_k = math.hypot(k1, k2)
    if denom_k < 1e-12:
        denom_k = 1e-12
    cosarg = max(-1.0, min(1.0, k3 / denom_k))
    delta = math.acos(cosarg)
    th2 = _atan2(k2, k1) + delta

    c2 = math.cos(th2); s2 = math.sin(th2)
    Kx = r - a1 - a2*c2
    Ky = z_c - d1 - a2*s2

    if abs(d4) < 1e-12:
        D = math.hypot(Kx, Ky)
        num = D**2 - a2**2 - a3**2
        den = 2*a2*a3 if (2*a2*a3)!=0 else 1e-12
        cos_th3 = max(-1.0, min(1.0, num/den))
        th3_planar = math.acos(cos_th3)
        th23 = th3_planar
    else:
        s23 = -Kx / d4
        c23 = Ky / d4
        mag = math.hypot(s23, c23)
        if mag < 1e-12:
            s23, c23 = 0.0, 1.0
        else:
            s23 /= mag; c23 /= mag
        th23 = _atan2(s23, c23)

    th3 = th23 - th2

    th4 = th5 = th6 = 0.0
    if target_R is not None:
        # assemble R0_3 (heuristic) and compute wrist rotations
        def Ry(a):
            ca, sa = math.cos(a), math.sin(a)
            return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])
        def Rz(a):
            ca, sa = math.cos(a), math.sin(a)
            return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
        R0_1 = Rz(th1)
        R1_2 = Ry(th2)
        R2_3 = Ry(th3)
        R0_3 = R0_1.dot(R1_2).dot(R2_3)
        R3_6 = R0_3.T.dot(np.asarray(target_R))
        m = R3_6
        m11,m12,m13 = m[0,0], m[0,1], m[0,2]
        m21,m22,m23 = m[1,0], m[1,1], m[1,2]
        m31,m32,m33 = m[2,0], m[2,1], m[2,2]
        th5 = _atan2(math.sqrt(max(0.0, 1.0 - m33*m33)), m33)
        th4 = _atan2(m23, m13)
        th6 = _atan2(m32, -m31)

    # Build joint list in same order as model.joints
    # We will place zero for non-revolute/free joints.
    q6 = [th1, th2, th3, th4, th5, th6]

    if mj_model is None:
        return q6

    # Map results onto mj_model.joint order using joint names.
    # We will fill indices for revolute joints sequentially with q6 values and zeros elsewhere.
    nq = int(mj_model.nq)
    q_full = [0.0]*nq
    # iterate mj_model.joint(i) for i in range(mj_model.njnt) to find revolute joints and fill
    # NOTE: mapping depends on your MJCF joint ordering; this is a heuristic
    q_idx = 0
    for ji in range(int(mj_model.njnt)):
        j = mj_model.joint(ji)
        jtype = j.type  # numeric code in mujoco: 0-free, 1-ball, 2-unbounded? (depends on mujoco)
        # we only fill for revolute (hinge) joints; check name for 'joint' substring as heuristic
        jname = j.name.lower()
        if ('joint' in jname) or ('hinge' in jname) or ('revolute' in jname) or (q_idx < len(q6)):
            # place next q6 if available
            if q_idx < len(q6):
                # find qpos index of this joint in mj_model -- joint.qposadr exists in C API via model.joint_qposadr ?
                try:
                    qpadr = int(mj_model.jnt_qposadr[ji])
                except Exception:
                    # fallback: place sequentially
                    qpadr = q_idx
                if qpadr < nq:
                    q_full[qpadr] = float(q6[q_idx])
                q_idx += 1

    return q_full

# ---------------- Example: compute IK for one example position ----------------
# Example end-effector position (meters)
target_pos = [0.5, 0.0, 0.4]

# PUMA-like parameters (adjust these to match your MJCF exactly if you have the true values)
# NOTE: the MJCF-based inverse_kinematics_custom expects params keys:
#   a1, a2, a3, d1, d2, d4, d6
# Earlier notes used d3 in the URDF form — here we place that offset into d2
params_example = {
    "a1": 0.0,        # base x-offset
    "a2": 0.4318,     # typical PUMA a2 (m)
    "a3": 0.0203,     # small a3 (m)
    "d1": 0.0,        # base z-offset
    "d2": 0.15005,    # this corresponds to the earlier d3 / link-offset (tweak if needed)
    "d4": 0.4318,     # wrist offset (example)
    "d6": 0.0         # end-effector tool length (0 if none)
}

# If you don't want to rely on MJCF inference, pass params_example explicitly.
# Call IK (use mj_model=model and mj_data=data if available)





MJCF = "puma560_description/urdf/puma560_robot.xml"
model = mujoco.MjModel.from_xml_path(MJCF)
data = mujoco.MjData(model)

print("Loaded MJCF model successfully!")
print("Number of joints:", model.njnt)
print("Number of qpos:", model.nq)
print("Joint names:")
for i in range(model.njnt):
    print("  ", i, model.joint(i).name)
target_pos = [0.5, 0.0, 0.4]

params_example = {
    "a1": 0.0,
    "a2": 0.4318,
    "a3": 0.0203,
    "d1": 0.0,
    "d2": 0.15005,
    "d4": 0.4318,
    "d6": 0.0
}

q_full = inverse_kinematics_custom(
    target_pos,
    target_R=None,
    mj_model=model,
    mj_data=data,
    params=params_example,
    verbose=True
)

print("\nIK q_full:", q_full)

