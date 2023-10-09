import numpy as np
from numpy.linalg import norm


def Area(x1, x2, x3, y1, y2, y3):
    a = x2 * y3 - y2 * x3 + x3 * y1 - y3 * x1 + x1 * y2 - y1 * x2  # % mass
    a = a / 2
    return a


def Area_r(r_0: np.array, r_1: np.array, r_2: np.array):
    # vectors of sides
    r_01 = np.array(r_1) - np.array(r_0)
    r_02 = np.array(r_2) - np.array(r_0)
    # r_12 = r_2 - r_1
    cross = np.abs(np.cross(r_01, r_02)) / 2.0

    return cross


def Area_r_short(triangles, radiusvector):
    r_0, r_1, r_2 = radiusvector[triangles[:, 0]], radiusvector[triangles[:, 1]], radiusvector[triangles[:, 2]]
    # vectors of sides
    # r_01 = r_1 - r_0
    # r_02 = r_2 - r_0
    # r_12 = r_2 - r_1
    cross = np.abs(np.cross(r_1 - r_0, r_2 - r_1, axis=1)) / 2

    return cross
