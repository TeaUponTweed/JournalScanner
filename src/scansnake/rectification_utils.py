import math

import numpy as np
import scipy as sp
from shapely.geometry import Point, Polygon


def two_norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1])


def find_normals(P):
    # find pointing unit vectors assuming clock-wise points
    r10 = (P[1] - P[0]) / two_norm(P[1] - P[0])
    r21 = (P[2] - P[1]) / two_norm(P[2] - P[1])
    r32 = (P[3] - P[2]) / two_norm(P[3] - P[2])
    r03 = (P[0] - P[3]) / two_norm(P[0] - P[3])
    # rotate by -90 deg to get normal vectors inward
    N0 = np.array([r03[1], -r03[0]])
    N1 = np.array([r10[1], -r10[0]])
    N2 = np.array([r21[1], -r21[0]])
    N3 = np.array([r32[1], -r32[0]])

    return (N0, N1, N2, N3)


def map_uv_to_xy(u, v, P, N):
    A = np.zeros((2, 2))
    b = np.zeros(2)
    A[0, :] = u * N[2] - (1 - u) * N[0]
    A[1, :] = v * N[3] - (1 - v) * N[1]
    b[0] = u * P[2] @ N[2] - (1 - u) * P[0] @ N[0]
    b[1] = v * P[3] @ N[3] - (1 - v) * P[0] @ N[1]
    return np.linalg.solve(A, b)


def get_sample_xy_points(points, normals, npoints, height_pixels, width_pixels):
    out_x = np.zeros((npoints, npoints))
    out_y = np.zeros((npoints, npoints))
    u = np.linspace(0, 1, npoints)
    v = np.linspace(0, 1, npoints)
    for i in range(npoints):
        for j in range(npoints):
            xy = map_uv_to_xy(u[j], v[i], points, normals)
            r = int(xy[1])
            c = int(xy[0])
            if r < height_pixels:
                out_y[i, j] = r
            else:
                out_y[i, j] = height_pixels - 1

            if c < width_pixels:
                out_x[i, j] = c
            else:
                out_x[i, j] = width_pixels - 1

    return u, v, out_x, out_y


def get_square_impl(xx, yy, gray):
    out = np.empty(gray.shape, gray.dtype)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            out[i, j] = gray[yy[i, j], xx[i, j]]
    return out


def get_square_image(gray, width_pixels, height_pixels, points):
    normals = find_normals(points)
    out = np.zeros((height_pixels, width_pixels))
    u, v, x_map, y_map = get_sample_xy_points(
        points, normals, 30, gray.shape[0], gray.shape[1]
    )

    x_func = sp.interpolate.interp2d(u, v, x_map)
    y_func = sp.interpolate.interp2d(u, v, y_map)
    u = np.linspace(0, 1, width_pixels)
    v = np.linspace(0, 1, height_pixels)

    xx = x_func(u, v)
    np.rint(xx, out=xx)
    np.clip(xx, a_min=0, a_max=None, out=xx)
    xx = xx.astype(np.int)

    yy = y_func(u, v)
    np.rint(yy, out=yy)
    np.clip(yy, a_min=0, a_max=None, out=yy)
    yy = yy.astype(np.int)
    return gray[yy, xx]


_TEST_POINTS = [
    np.array([1, -2]),
    np.array([10, -3]),
    np.array([12, -9]),
    np.array([2, -7]),
    np.array([1, -2]),
]


def test_normals():
    normals = find_normals(_TEST_POINTS)
    # make sure they are unit vectors
    np.testing.assert_array_almost_equal(np.linalg.norm(normals, axis=1), np.ones(4))
    # n1, slope = -1/9
    np.testing.assert_array_almost_equal(
        (np.array([-1, -9]) / math.sqrt(9 * 9 + 1 * 1)), normals[1]
    )
    # n2, slope = -6/2
    np.testing.assert_array_almost_equal(
        (np.array([-6, -2]) / math.sqrt(2 * 2 + 6 * 6)), normals[2]
    )
    # n3, slope = 2/-10
    np.testing.assert_array_almost_equal(
        (np.array([2, 10]) / math.sqrt(2 * 2 + 10 * 10)), normals[3]
    )
    # n3, slope = 5/-1
    np.testing.assert_array_almost_equal(
        (np.array([5, 1]) / math.sqrt(1 * 1 + 5 * 5)), normals[0]
    )


def test_to_xy():
    # visual inspection
    points = _TEST_POINTS
    normals = find_normals(_TEST_POINTS)
    debug_plot = False
    if debug_plot:
        import matplotlib.pyplot as plt

        plt.plot(*zip(*points), color="k")
        for i in range(4):
            p = (points[i] + points[(i + 1) % 4]) / 2
            normal = normals[(i + 1) % 4]
            plt.plot(*zip(*[p, p + normal]), label=f"n{(i+1)%4}")

    poly = Polygon(points)

    for u in np.linspace(0, 1, 10):
        for v in np.linspace(0, 1, 10):
            (x, y) = p = map_uv_to_xy(u, v, points, normals)
            a = (p - points[0]) @ normals[0]
            b = (p - points[2]) @ normals[2]
            _u = a / (a + b)
            c = (p - points[0]) @ normals[1]
            d = (p - points[3]) @ normals[3]
            _v = c / (c + d)
            np.testing.assert_almost_equal(_v, v)
            np.testing.assert_almost_equal(_u, u)
            if u != 0 and u != 1 and v != 0 and v != 1:
                assert poly.contains(Point(x, y))
            if debug_plot:
                plt.plot([x], [y], marker="x", color="r")
    if debug_plot:
        plt.xlim([0, 15])
        plt.ylim([-15, 0])
        plt.legend()
        plt.show()

    np.testing.assert_array_almost_equal(map_uv_to_xy(0, 0, points, normals), points[0])
    np.testing.assert_array_almost_equal(map_uv_to_xy(0, 1, points, normals), points[3])
    np.testing.assert_array_almost_equal(map_uv_to_xy(1, 0, points, normals), points[1])
    np.testing.assert_array_almost_equal(map_uv_to_xy(1, 1, points, normals), points[2])


if __name__ == "__main__":
    test_normals()
    test_to_xy()
