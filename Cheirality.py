import numpy as np

def triangulate_point(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Linear (DLT) triangulation.
    x1, x2: homogeneous image points (3x1)
    Returns X: homogeneous 3D point (4x1)
    """
    A = np.vstack([
        x1[0, 0] * P1[2, :] - P1[0, :],
        x1[1, 0] * P1[2, :] - P1[1, :],
        x2[0, 0] * P2[2, :] - P2[0, :],
        x2[1, 0] * P2[2, :] - P2[1, :],
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1].reshape(4, 1)
    X = X / X[-1, 0]
    return X

def cheirality_count(P1: np.ndarray, P2: np.ndarray, src, dst, n_corr: int = 12) -> int:
    """
    Count how many of the first n_corr correspondences triangulate in front of BOTH cameras.
    src, dst: lists of (3x1) homogeneous points
    """
    count = 0
    for i in range(min(n_corr, len(src), len(dst))):
        x1 = src[i]
        x2 = dst[i]
        X = triangulate_point(P1, P2, x1, x2)

        # Depth in each camera: z of projected camera coordinates (up to positive scale)
        z1 = (P1 @ X)[2, 0]
        z2 = (P2 @ X)[2, 0]

        if z1 > 0 and z2 > 0:
            count += 1
    return count

def test(E: np.ndarray, K: np.ndarray, src, dst, n_corr: int = 12):
    """
    Build P1 and 4 candidates of P2 from E, then pick the best P2 using cheirality on first n_corr points.

    src, dst: list[np.ndarray], each point is homogeneous (3x1)
    Returns:
      P1 (3x4),
      P2_candidates (list of 4x (3x4)),
      best_idx (0..3),
      best_P2 (3x4),
      scores (list of 4 ints)
    """
    # Optional but recommended: force E to have singular values (s, s, 0)
    U, S, Vt = np.linalg.svd(E)
    s = 0.5 * (S[0] + S[1])
    E = U @ np.diag([s, s, 0.0]) @ Vt

    # Decompose E
    U, _, Vt = np.linalg.svd(E)

    # Make U,Vt proper (det = +1)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]], dtype=float)

    R1 = U @ W  @ Vt
    R2 = U @ W.T @ Vt

    # Ensure det(R)=+1
    if np.linalg.det(R1) < 0: R1 *= -1
    if np.linalg.det(R2) < 0: R2 *= -1

    t = U[:, 2].reshape(3, 1)  # translation direction (up to scale)

    # Camera matrices
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

    P2_1 = K @ np.hstack([R1,  t])
    P2_2 = K @ np.hstack([R1, -t])
    P2_3 = K @ np.hstack([R2,  t])
    P2_4 = K @ np.hstack([R2, -t])

    P2_candidates = [P2_1, P2_2, P2_3, P2_4]

    # Cheirality selection using first n_corr correspondences
    scores = [cheirality_count(P1, P2, src, dst, n_corr=n_corr) for P2 in P2_candidates]
    best_idx = int(np.argmax(scores))
    best_P2 = P2_candidates[best_idx]

    return P1, best_P2
