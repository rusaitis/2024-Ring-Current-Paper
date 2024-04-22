import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if path not in sys.path:
    sys.path.append(path)

def cubic_interpolation(points, t=None, t_val=None, tangents=None):
    """ Cubic spline interpolation for 3D points """
    N = len(points)
    if t is None:
        t = np.linspace(0, 1, N)
    if t_val is None:
        t_val = np.linspace(0, 1, N*5)
    if tangents is None:
        # Cubic spline interpolation for each dimension
        cs_x = CubicSpline(t, points[:, 0])
        cs_y = CubicSpline(t, points[:, 1])
        cs_z = CubicSpline(t, points[:, 2])
    else:
        # tangents = tangents / np.linalg.norm(tangents, axis=1)[:, None]
        # Cubic spline interpolation for each dimension with tangents
        cs_x = CubicSpline(t, points[:, 0], bc_type=((1, tangents[0, 0]), (1, tangents[-1, 0])))
        cs_y = CubicSpline(t, points[:, 1], bc_type=((1, tangents[0, 1]), (1, tangents[-1, 1])))
        cs_z = CubicSpline(t, points[:, 2], bc_type=((1, tangents[0, 2]), (1, tangents[-1, 2])))

    points_int = [np.array([cs_x(_), cs_y(_), cs_z(_)]) for _ in t_val]
    return np.array(points_int)


def cubic_interpolation1D(points, t=None, t_val=None):
    """ Cubic spline interpolation for 1D """
    N = len(points)
    if t is None:
        t = np.linspace(0, 1, N)
    if t_val is None:
        t_val = np.linspace(0, 1, N*5)
    cs = CubicSpline(t, points)
    points_int = [cs(_) for _ in t_val]
    return np.array(points_int)

if __name__ == "__main__":
    from plot import *
    # Sample points
    points = np.array([
        [0, 0, 0],    # P1
        [1, 2, 3],    # P2
        [4, 3, 2],    # P3
        [7, 5, 5],    # P4
        [10, 8, 7]    # P5
    ])

    # Tangent vectors at each point
    tangents = np.array([
        [2, -5, -1],   # Tangent at P1
        [2, -5, -1],   # Tangent at P2
        [2, -5, -1],   # Tangent at P3
        [2, -5, -1],   # Tangent at P4
        [1, -1, -1]    # Tangent at P5
    ])

    t = np.linspace(0, 1, len(points))
    # t = [0, 1, 2, 8, 9]
    t_vals = np.linspace(0, 1, 100, endpoint=True)
    interpolated_points = cubic_interpolation(points, t, t_vals)
    interpolated_points_tangent = cubic_interpolation(points, t, t_vals, tangents)

    # use dark backtround
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*points.T, 'ro', label='Original Points')
    ax.plot(*interpolated_points.T, 'b-', label='Cubic Spline')
    ax.plot(*interpolated_points_tangent.T, 'g--', label='Cubic Spline with Tangents')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    ax.set_aspect('equal')
    # remove axes frames
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.title("3D Cubic Spline Interpolation")
    plt.show()
    pass
