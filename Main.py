import numpy as np
from matplotlib import pyplot as plt
import Renderer as rd
import Utility as ut
import VPCalculator as vp
import Cheirality as ch
import LMTriangulation as lmt

A1s = [2277.20, 1024.74]
A2s = [2684.00, 1165.05]
A3s = [2792.35, 1100.15]
A4s = [2958.68, 947.07]
A5s = [2483.91, 1249.85]
A6s = [2895.50, 1176.09]
A7s = [2456.60, 2079.96]
A8s = [2930.88, 2025.67]
A9s = [2230.8, 2413.67]
A10s = [2698.94, 2466.28]
A11s = [2834.46, 2429.31]
A12s = [3038.57, 2350.80]

A13s = [918.13, 551.97]
A14s = [1521.28, 760.44]
A15s = [1670.51, 812.35]
A16s = [2155.76, 978.77]
A17s = [800.43, 1082.66]
A18s = [1448.04, 1257.32]
A19s = [1611.28, 1299.65]
A20s = [2128.08, 1439.00]
A21s = [721.88, 1444.92]
A22s = [1404.54, 1594.25]
A23s = [1571.98, 1629.82]
A24s = [2113.06, 1748.45]
A25s = [556.69, 2228.79]
A26s = [1313.09, 2312.28]
A27s = [1500.18, 2332.40]
A28s = [2080.42, 2397.65]
A29s = [466.48, 2682.98]
A30s = [1267.67, 2724.45]
A31s = [1460.86, 2729.49]
A32s = [2073.88, 2758.78]
A33s = [274.33, 3565.79]
A34s = [1182.14, 3491.83]
A35s = [1395.07, 3476.04]
A36s = [2049.51, 3432.01]

A37s = [3125.35, 705.89]
A38s = [3617.57, 244.78]
A39s = [3712.62, 157.32]
A40s = [3171.86, 1258.58]
A41s = [3719.26, 855.95]
A42s = [3825.68, 778.54]
A43s = [3193.91, 1567.65]
A44s = [3773.94, 1200.99]
A45s = [3885.93, 1128.69]
A46s = [3249.91, 2270.69]
A47s = [3900.05, 2017.29]
A48s = [4030.99, 1971.61]

A1d = [900.34, 702.53]
A2d = [1403.83, 971.71]
A3d = [1533.98, 922.11]
A4d = [1734.55, 789.95]
A5d = [1135.16, 1014.85]
A6d = [1631.59, 1023.87]
A7d = [949.74, 1939.79]
A8d = [1526.30, 1943.27]
A9d = [569.29, 2294.99]
A10d = [1197.11, 2404.29]
A11d = [1355.74, 2375.47]
A12d = [1582.19, 2311.73]

pre_src = [
    A1s, A2s, A3s, A4s, A5s, A6s, A7s, A8s, A9s, A10s, A11s, A12s,
    A13s, A14s, A15s, A16s, A17s, A18s, A19s, A20s, A21s, A22s, A23s, A24s,
    A25s, A26s, A27s, A28s, A29s, A30s, A31s, A32s, A33s, A34s, A35s, A36s,
    A37s, A38s, A39s, A40s, A41s, A42s, A43s, A44s, A45s, A46s, A47s, A48s
]
pre_dst = [
    A1d, A2d, A3d, A4d, A5d, A6d, A7d, A8d, A9d, A10d, A11d, A12d
]

src = []
dst = []

# --- Convert Cartesian coordinate into homogeneous ---
for point in pre_src:
    new_point = np.array([point[0], point[1], 1.0], dtype = float).reshape(3, 1)
    src.append(new_point)
for point in pre_dst:
    new_point = np.array([point[0], point[1], 1.0], dtype = float).reshape(3, 1)
    dst.append(new_point)

# --- Calculate fundamental matrix ---
fundamental_mat = []
for i, (p1, p2) in enumerate(zip(src, dst)):
    if i <= 11:
        constraints = ut.fundamental_constraints(p1, p2)
        fundamental_mat.append(constraints)
fundamental_mat = np.array(fundamental_mat, dtype=float)
_, _, Vt = np.linalg.svd(fundamental_mat)
f = Vt[-1]
F = f.reshape(3, 3)

# --- Calculate IAC (Image of Absolute Conic) ---
l_13_33 = ut.line(src[12], src[32]).flatten().tolist()
l_14_34 = ut.line(src[13], src[33]).flatten().tolist()
l_15_35 = ut.line(src[14], src[34]).flatten().tolist()
l_16_36 = ut.line(src[15], src[35]).flatten().tolist()
l_1_9 = ut.line(src[0], src[8]).flatten().tolist()
l_2_10 = ut.line(src[1], src[9]).flatten().tolist()
l_3_11 = ut.line(src[2], src[10]).flatten().tolist()
l_4_12 = ut.line(src[3], src[11]).flatten().tolist()
l_37_46 = ut.line(src[36], src[45]).flatten().tolist()
l_38_47 = ut.line(src[37], src[46]).flatten().tolist()
l_39_48 = ut.line(src[38], src[47]).flatten().tolist()
lines_up = [l_13_33, l_14_34, l_15_35, l_16_36,
            l_1_9, l_2_10, l_3_11, l_4_12,
            l_37_46, l_38_47, l_39_48]
v_up = vp.VP_LM(lines_up)

l_13_16 = ut.line(src[12], src[15]).flatten().tolist()
l_17_20 = ut.line(src[16], src[19]).flatten().tolist()
l_21_24 = ut.line(src[20], src[23]).flatten().tolist()
l_25_28 = ut.line(src[24], src[27]).flatten().tolist()
l_29_32 = ut.line(src[28], src[31]).flatten().tolist()
l_33_36 = ut.line(src[32], src[35]).flatten().tolist()
lines_right = [l_13_16, l_17_20, l_21_24,
               l_25_28, l_29_32, l_33_36]
v_right = vp.VP_LM(lines_right)

l_37_39 = ut.line(src[36], src[38]).flatten().tolist()
l_40_42 = ut.line(src[39], src[41]).flatten().tolist()
l_43_45 = ut.line(src[42], src[44]).flatten().tolist()
l_46_48 = ut.line(src[45], src[47]).flatten().tolist()
lines_left = [l_37_39, l_40_42,
              l_43_45, l_46_48]
v_left = vp.VP_LM(lines_left)

omega_mat = [ut.omega_constraints(v_right, v_left),
             ut.omega_constraints(v_left, v_up),
             ut.omega_constraints(v_up, v_right)]
_, _, Vt = np.linalg.svd(omega_mat)
theta = Vt[-1]
theta = theta / theta[-1]
a, d, e, _ = theta
f = 1 / np.sqrt(a)
cx = -d / a
cy = -e / a
K = np.array([
    [f, 0, cx],
    [0, f, cy],
    [0, 0, 1]
])
E = K.T @ F @ K
P1, P2 = ch.test(E, K, src, dst)
P2, X_3D_homogeneous = lmt.triangulate_and_LM(P1, P2, src, dst)
X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12 = X_3D_homogeneous

# ---- 2D canvas ----
fig2d, ax2d = plt.subplots()
ax2d.set_title("2D Canvas")
ax2d.set_xlim(-0, 10000)
ax2d.set_ylim(5000, -8000)
rd.render_image(ax2d, "Sample/Image 1.JPG")

# i1 = [12, 16, 20, 24, 28, 32] # right
# i2 = [32, 33, 34, 35, 45, 46, 47] # up
# i3 = [47, 44, 41, 38] # left
# for i in i1:
#     rd.render_line2D(ax2d, v_right, src[i], color = (255, 0, 0))
# for i in i2:
#     rd.render_line2D(ax2d, v_up, src[i], color = (255, 0, 0))
# for i in i3:
#     rd.render_line2D(ax2d, v_left, src[i], color = (255, 0, 0))

# rd.render_point2D(ax2d, v_up, color = (0, 255, 0))
# rd.render_point2D(ax2d, v_right, color = (0, 255, 0))
# rd.render_point2D(ax2d, v_left, color = (0, 255, 0))
for X in X_3D_homogeneous:
    projection = P1 @ X
    rd.render_point2D(ax2d, projection)
for i, x in enumerate(src):
    if i <= 11: rd.render_point2D(ax2d, x, color = (0, 255, 0))


# ---- 3D canvas ----
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.set_title("3D Canvas")
for X in X_3D_homogeneous:
    rd.render_point3D(ax3d, X, color = (0, 255, 0))

rd.render_line3D(ax3d, X1, X2)
rd.render_line3D(ax3d, X2, X10)
rd.render_line3D(ax3d, X10, X9)
rd.render_line3D(ax3d, X9, X1)
rd.render_line3D(ax3d, X3, X4)
rd.render_line3D(ax3d, X4, X12)
rd.render_line3D(ax3d, X12, X11)
rd.render_line3D(ax3d, X11, X3)

plt.show()

