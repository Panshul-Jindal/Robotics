import numpy as np
import matplotlib.pyplot as plt
from math import pi
from matplotlib import animation

# --- DH parameters ---
inch = 0.0254
base = 26.45 * inch
d = np.array([base, 0.0, 0.15005, 0.4318, 0.0, 0.0])
a = np.array([0.0, 0.4318, 0.0203, 0.0, 0.0, 0.0])
alpha = np.array([pi/2, 0.0, -pi/2, pi/2, -pi/2, 0.0])
qlim = np.array([
    [-160*pi/180, 160*pi/180],
    [-110*pi/180, 110*pi/180],
    [-135*pi/180, 135*pi/180],
    [-266*pi/180, 266*pi/180],
    [-100*pi/180, 100*pi/180],
    [-266*pi/180, 266*pi/180],
])

def dh_transform(theta, d_i, a_i, alpha_i):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha_i), np.sin(alpha_i)
    return np.array([
        [ct, -st*ca,  st*sa, a_i*ct],
        [st,  ct*ca, -ct*sa, a_i*st],
        [0, sa, ca, d_i],
        [0, 0, 0, 1]
    ])

def fkine(q):
    T = np.eye(4)
    for i in range(6):
        T = T @ dh_transform(q[i], d[i], a[i], alpha[i])
    return T[:3, 3]

def sample_workspace():
    q1 = np.linspace(qlim[0,0], qlim[0,1], 80)
    q2 = np.linspace(qlim[1,0], qlim[1,1], 80)
    q3 = np.linspace(qlim[2,0], qlim[2,1], 80)
    pts = []
    for th1 in q1[::3]:
        for th2 in q2[::3]:
            for th3 in q3[::3]:
                q = np.array([th1, th2, th3, 0, 0, 0])
                pts.append(fkine(q))
    return np.array(pts)

# --- compute workspace ---
pts = sample_workspace()
x, y, z = pts[:,0], pts[:,1], pts[:,2]
r = np.sqrt(x**2 + y**2)

# --- create animation ---
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=r, cmap='viridis', s=1.2, alpha=0.4)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("PUMA 560 Workspace")
ax.set_box_aspect([1,1,1])
ax.view_init(elev=-5, azim=80)
fig.tight_layout()

def rotate(angle):
    ax.view_init(elev=30, azim=angle)

ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 1), interval=80)
ani.save("puma560_workspace-3d.mp4", writer="ffmpeg", fps=30, dpi=400)

plt.close(fig)
