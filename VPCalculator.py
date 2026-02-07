import numpy as np

def VP_LM(lines: list, iteration: int = 50, ld: float = 1e-3,
          bounds=None, tol: float = 1e-12):
    A = np.array(lines, dtype=float)   # N x 3

    # --- DLT init ---
    _, _, Vt = np.linalg.svd(A)
    vp = Vt[-1].reshape(3, 1)
    if abs(vp[2, 0]) < 1e-12:
        raise ValueError("VP at infinity (vp[2]≈0). This (x,y,1) parameterization won't work.")
    vp = vp / vp[2, 0]

    def clamp_xy(x, y):
        if bounds is None:
            return x, y
        xmin, xmax, ymin, ymax = bounds
        return float(np.clip(x, xmin, xmax)), float(np.clip(y, ymin, ymax))

    # optional clamp initial
    vp[0,0], vp[1,0] = clamp_xy(vp[0,0], vp[1,0])

    def build_J_r(x, y):
        J = []
        r = []
        for a, b, c in A:
            d = np.sqrt(a*a + b*b)
            if d < 1e-12:
                continue
            J.append([a/d, b/d])
            r.append([(a*x + b*y + c)/d])
        J = np.asarray(J, dtype=float)
        r = np.asarray(r, dtype=float)
        return J, r

    # initial cost
    J, r = build_J_r(vp[0,0], vp[1,0])
    prev_cost = float(r.T @ r)

    for _ in range(iteration):
        x, y = float(vp[0,0]), float(vp[1,0])

        # current J, r
        J, r = build_J_r(x, y)
        cost = float(r.T @ r)

        # LM step
        I = np.eye(2, dtype=float)
        H = J.T @ J + ld * I
        g = -J.T @ r
        delta = np.linalg.solve(H, g)          # (2,1)

        x_try = x + float(delta[0,0])
        y_try = y + float(delta[1,0])
        x_try, y_try = clamp_xy(x_try, y_try)

        # evaluate candidate
        _, r_try = build_J_r(x_try, y_try)
        new_cost = float(r_try.T @ r_try)

        if new_cost < cost:   # accept
            vp[0,0], vp[1,0], vp[2,0] = x_try, y_try, 1.0
            ld *= 0.5

            # stop if improvement tiny
            if abs(cost - new_cost) < tol:
                break
        else:                 # reject
            ld *= 2.0

    return vp
