import matplotlib.pyplot as plt
import numpy as np


def render_image(axe, path: str):
    image = plt.imread(path)
    axe.imshow(image)


# ---------- helpers ----------
def _normalize2D(p):
    return p / p[2, 0]


def _normalize3D(p):
    return p / p[3, 0]


# ---------- 2D ----------
def render_point2D(axe, point: np.ndarray, color = (255, 0, 0)):
    point = _normalize2D(point)

    x = point[0, 0]
    y = point[1, 0]

    color = np.array(color) / 255
    axe.scatter(x, y, s=10, color=color)


def render_line2D(axe, p1: np.ndarray, p2: np.ndarray, color = (255, 0, 0)):
    p1 = _normalize2D(p1)
    p2 = _normalize2D(p2)

    x = [p1[0, 0], p2[0, 0]]
    y = [p1[1, 0], p2[1, 0]]

    color = np.array(color) / 255
    axe.plot(x, y, color=color, linewidth=1)


# ---------- 3D ----------
def render_point3D(axe, point: np.ndarray, color = (255, 0, 0)):
    point = _normalize3D(point)

    x = point[0, 0]
    y = point[1, 0]
    z = point[2, 0]

    color = np.array(color) / 255
    axe.scatter(x, y, z, s=20, color=color)


def render_line3D(axe, p1: np.ndarray, p2: np.ndarray, color = (255, 0, 0)):
    p1 = _normalize3D(p1)
    p2 = _normalize3D(p2)

    x = [p1[0, 0], p2[0, 0]]
    y = [p1[1, 0], p2[1, 0]]
    z = [p1[2, 0], p2[2, 0]]

    color = np.array(color) / 255
    axe.plot(x, y, z, color=color, linewidth=1)
