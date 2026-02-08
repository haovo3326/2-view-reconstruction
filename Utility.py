import numpy as np

def line(p1: np.ndarray, p2: np.ndarray):
    """
    :param p1: homogeneous point 3 x 1
    :param p2: homogeneous point 3 x 1
    :return: line ax + by + c = 0 (3 x 1)
    """
    l = np.cross(p1.flatten(), p2.flatten()).reshape(3, 1)
    return l

def enforce_rank2(m: np.ndarray):
    assert m.shape == (3, 3)
    U, S, Vt = np.linalg.svd(m)
    S[-1] = 0
    return U @ S @ Vt

def omega_constraints(v1: np.ndarray, v2: np.ndarray):
    x1, y1, _ = v1.flatten()
    x2, y2, _ = v2.flatten()
    return [x1*x2 + y1*y2, x1 + x2, y1 + y2, 1]

def fundamental_constraints(x: np.ndarray, x_c: np.ndarray):
    u, v, _ = x.flatten()
    u_c, v_c, _ = x_c.flatten()
    return [u * u_c, u_c * v, u_c, v_c * u, v * v_c, v_c, u, v, 1]

