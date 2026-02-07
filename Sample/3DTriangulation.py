import numpy as np

def triangulate_and_LM(P1: np.ndarray, P2: np.ndarray, src, dst,
                       n_corr: int = 12,
                       iteration: int = 100,
                       ld: float = 1e-3):
    """
    src, dst: list of homogeneous image points, each (3x1)
    Use first n_corr correspondences.
    Triangulate initial Xi (4x1), then LM optimize P2 and all Xi.

    Returns:
      P2, points_in_homogeneous (list of 4x1), history (list of residual norms)
    """

    # ---------------- STEP 4: Triangulation ----------------
    points_in_homogeneous = []

    p1t, p2t, p3t = P1[0], P1[1], P1[2]
    p1t_2, p2t_2, p3t_2 = P2[0], P2[1], P2[2]

    N = min(n_corr, len(src), len(dst))
    for x, x_c in zip(src[:N], dst[:N]):
        # src/dst are (3x1) homogeneous -> use inhomogeneous u,v
        if abs(x[2, 0]) <= 1e-12 or abs(x_c[2, 0]) <= 1e-12:
            continue

        u   = x[0, 0] / x[2, 0]
        v   = x[1, 0] / x[2, 0]
        u_c = x_c[0, 0] / x_c[2, 0]
        v_c = x_c[1, 0] / x_c[2, 0]

        B = np.array([
            u   * p3t   - p1t,
            v   * p3t   - p2t,
            u_c * p3t_2 - p1t_2,
            v_c * p3t_2 - p2t_2
        ], dtype=float)

        _, _, VtB = np.linalg.svd(B)
        Xh = VtB[-1].reshape(4, 1)  # keep homogeneous for LM
        points_in_homogeneous.append(Xh)

    # if some points were skipped, update N
    N = len(points_in_homogeneous)
    if N == 0:
        raise ValueError("No valid correspondences for triangulation (check homogeneous inputs).")

    # ---------------- LM optimization (P2 and all Xi) ----------------
    history = []
    I3 = np.eye(3, dtype=float)

    for it in range(iteration):
        J = np.zeros((4 * N, 12 + 4 * N), dtype=float)
        r = np.zeros((4 * N, 1), dtype=float)

        # Build residual and Jacobian
        for i in range(N):
            Xi = points_in_homogeneous[i]  # (4,1)

            y1 = P1 @ Xi  # (3,1)
            y2 = P2 @ Xi  # (3,1)

            # guard against division by ~0
            if abs(y1[2, 0]) <= 1e-12 or abs(y2[2, 0]) <= 1e-12:
                continue

            # projections
            pi1 = np.array([[y1[0, 0] / y1[2, 0]],
                            [y1[1, 0] / y1[2, 0]]], dtype=float)
            pi2 = np.array([[y2[0, 0] / y2[2, 0]],
                            [y2[1, 0] / y2[2, 0]]], dtype=float)

            # observed (inhomogeneous) 2D points
            x1h = src[i]
            x2h = dst[i]
            if abs(x1h[2, 0]) <= 1e-12 or abs(x2h[2, 0]) <= 1e-12:
                continue
            x1 = np.array([[x1h[0, 0] / x1h[2, 0]],
                           [x1h[1, 0] / x1h[2, 0]]], dtype=float)
            x2 = np.array([[x2h[0, 0] / x2h[2, 0]],
                           [x2h[1, 0] / x2h[2, 0]]], dtype=float)

            ri = np.vstack([x1 - pi1, x2 - pi2])  # (4,1)
            r0, r1 = 4 * i, 4 * i + 4
            r[r0:r1, 0:1] = ri

            # d(pi)/d(y)
            dpi_dy1 = np.array([
                [1.0 / y1[2, 0], 0.0, -y1[0, 0] / (y1[2, 0] ** 2)],
                [0.0, 1.0 / y1[2, 0], -y1[1, 0] / (y1[2, 0] ** 2)]
            ], dtype=float)

            dpi_dy2 = np.array([
                [1.0 / y2[2, 0], 0.0, -y2[0, 0] / (y2[2, 0] ** 2)],
                [0.0, 1.0 / y2[2, 0], -y2[1, 0] / (y2[2, 0] ** 2)]
            ], dtype=float)

            # vec(P2) Jacobian (column-major)
            # y2 = P2 @ Xi, so dy2/dvec(P2) = kron(Xi^T, I3)
            dy2_dvecP2 = np.kron(Xi.T, I3)         # (3,12)
            dr2_dvecP2 = -dpi_dy2 @ dy2_dvecP2     # (2,12)

            # residual is [r1; r2], but only r2 depends on P2
            dRi_dP2 = np.vstack([np.zeros((2, 12), dtype=float), dr2_dvecP2])  # (4,12)

            # d r / d Xi
            dRi_dXi = np.vstack([
                -dpi_dy1 @ P1,   # (2,4)
                -dpi_dy2 @ P2    # (2,4)
            ])  # (4,4)

            # fill Jacobian blocks
            J[r0:r1, 0:12] = dRi_dP2
            c0, c1 = 12 + 4 * i, 12 + 4 * i + 4
            J[r0:r1, c0:c1] = dRi_dXi

        # current residual norm
        r_norm = float(np.linalg.norm(r))
        history.append(r_norm)

        # solve damped normal equations
        dim = 12 + 4 * N
        H = J.T @ J + ld * np.eye(dim, dtype=float)
        g = -J.T @ r

        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            ld *= 10.0
            continue

        # update P2
        dP2 = delta[0:12, 0].reshape(3, 4, order='F')
        P2_new = P2 + dP2

        # scale-fix (keeps projective matrix from drifting in magnitude)
        s = np.linalg.norm(P2_new[:, :3])
        if s > 1e-12:
            P2_new = P2_new / s

        # update Xi
        X_new = []
        for i in range(N):
            dXi = delta[12 + 4 * i: 12 + 4 * i + 4, 0].reshape(4, 1)
            Xi_new = points_in_homogeneous[i] + dXi

            # normalize homogeneous scale for numerical stability (same projective point)
            if abs(Xi_new[3, 0]) > 1e-12:
                Xi_new = Xi_new / Xi_new[3, 0]

            X_new.append(Xi_new)

        # compute new residual norm
        r_new = np.zeros((4 * N, 1), dtype=float)
        for i in range(N):
            Xi = X_new[i]
            y1 = P1 @ Xi
            y2 = P2_new @ Xi
            if abs(y1[2, 0]) <= 1e-12 or abs(y2[2, 0]) <= 1e-12:
                continue

            pi1 = np.array([[y1[0, 0] / y1[2, 0]],
                            [y1[1, 0] / y1[2, 0]]], dtype=float)
            pi2 = np.array([[y2[0, 0] / y2[2, 0]],
                            [y2[1, 0] / y2[2, 0]]], dtype=float)

            x1h = src[i]
            x2h = dst[i]
            if abs(x1h[2, 0]) <= 1e-12 or abs(x2h[2, 0]) <= 1e-12:
                continue
            x1 = np.array([[x1h[0, 0] / x1h[2, 0]],
                           [x1h[1, 0] / x1h[2, 0]]], dtype=float)
            x2 = np.array([[x2h[0, 0] / x2h[2, 0]],
                           [x2h[1, 0] / x2h[2, 0]]], dtype=float)

            ri_new = np.vstack([x1 - pi1, x2 - pi2])
            r0, r1 = 4 * i, 4 * i + 4
            r_new[r0:r1, 0:1] = ri_new

        if np.linalg.norm(r_new) < r_norm:
            P2 = P2_new
            points_in_homogeneous = X_new
            ld *= 0.5
            if r_norm - float(np.linalg.norm(r_new)) < 1e-12:
                break
        else:
            ld *= 2.0

    return P2, points_in_homogeneous
