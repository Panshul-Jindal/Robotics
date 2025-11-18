import numpy as np
from scipy.interpolate import CubicSpline

def quintic_interpolation(t_coarse, q_coarse, t_fine):
    q_fine = np.zeros(len(t_fine))
    for i in range(len(t_coarse) - 1):
        mask = (t_fine >= t_coarse[i]) & (t_fine <= t_coarse[i+1])
        if not np.any(mask):
            continue
        t0, t1 = t_coarse[i], t_coarse[i+1]
        q0, q1 = q_coarse[i], q_coarse[i+1]
        tau = (t_fine[mask] - t0) / (t1 - t0)
        a0, a1, a2 = q0, 0, 0
        a3 = 10 * (q1 - q0)
        a4 = -15 * (q1 - q0)
        a5 = 6 * (q1 - q0)
        q_fine[mask] = a0 + a1*tau + a2*tau**2 + a3*tau**3 + a4*tau**4 + a5*tau**5
    return q_fine

def lspb_interpolation(t_coarse, q_coarse, t_fine):
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
    q_fine = np.zeros(len(t_fine))
    for i in range(len(t_coarse) - 1):
        mask = (t_fine >= t_coarse[i]) & (t_fine <= t_coarse[i+1])
        if not np.any(mask):
            continue
        t0, t1 = t_coarse[i], t_coarse[i+1]
        q0, q1 = q_coarse[i], q_coarse[i+1]
        tau = (t_fine[mask] - t0) / (t1 - t0)
        q_diff = q1 - q0
        accel_mask = tau < 0.5
        tau_a = tau[accel_mask]
        q_fine[np.where(mask)[0][accel_mask]] = q0 + 2 * q_diff * tau_a**2
        decel_mask = tau >= 0.5
        tau_d = tau[decel_mask]
        q_fine[np.where(mask)[0][decel_mask]] = q1 - 2 * q_diff * (1 - tau_d)**2
    return q_fine

def interpolate_trajectory(q_coarse, t_coarse, t_fine, method="cubic"):
    
    # Interpolate joint-space trajectory using various methods.
    
    n_points, n_joints = q_coarse.shape
    n_fine = len(t_fine)
    q_fine = np.zeros((n_fine, n_joints))
    
    if method == "cubic":
        print("Using Cubic Spline interpolation")
        for j in range(n_joints):
            cs = CubicSpline(t_coarse, q_coarse[:, j], bc_type='clamped')
            q_fine[:, j] = cs(t_fine)
    
    elif method == "quintic":
        print("Using Quintic polynomial interpolation")
        for j in range(n_joints):
            q_fine[:, j] = quintic_interpolation(t_coarse, q_coarse[:, j], t_fine)
    
    elif method == "lspb":
        print("Using Linear Segment with Parabolic Blends (LSPB)")
        for j in range(n_joints):
            q_fine[:, j] = lspb_interpolation(t_coarse, q_coarse[:, j], t_fine)
    
    elif method == "bangbang":
        print("Using Bang-Bang control")
        for j in range(n_joints):
            q_fine[:, j] = bangbang_interpolation(t_coarse, q_coarse[:, j], t_fine)
    
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    for j in range(n_joints):
        q_fine[:, j] = np.unwrap(q_fine[:, j])
    
    return q_fine