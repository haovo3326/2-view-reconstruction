import matplotlib.pyplot as plt
import numpy as np

def render_image(axe, path: str):
    image = plt.imread(path)
    axe.imshow(image)

def render_point2D(axe, point: np.ndarray, color):
    x = point[0, 0]
    y = point[1, 0]
    color = np.array(color) / 255
    axe.scatter(x, y, s = 10, color = color)

def render_line2D(axe, p1: np.ndarray, p2: np.ndarray, color):
    x = [p1[0, 0], p2[0, 0]]
    y = [p1[1, 0], p2[1, 0]]
    color = np.array(color) / 255
    axe.plot(x, y, color = color, linewidth = 1)
