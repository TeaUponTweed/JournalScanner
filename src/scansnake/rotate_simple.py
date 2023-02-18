import itertools
import math
import sys

import cv2
import imutils
import matplotlib
import numpy as np
import scipy as sp
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.morphology import binary_opening

matplotlib.use("Qt5Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from numba import jit, njit, prange
from PIL import Image
from scipy import signal
from skimage.draw import line_aa
from skimage.feature import peak_local_max
from skimage.filters import scharr
from skimage.measure import block_reduce, label, regionprops
from skimage.restoration import denoise_tv_chambolle
from skimage.segmentation import mark_boundaries, slic


def resize(im):
    desired_size = max(im.shape)
    delta_w = desired_size - im.shape[1]
    delta_h = desired_size - im.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im


def find_intercept(l1, l2):
    (x1, y1, dx1, dy1) = l1
    (x2, y2, dx2, dy2) = l2
    A = np.array([[dx1, -dx2], [dy1, -dy2]])
    b = np.array([x2 - x1, y2 - y1])
    try:
        m, n = np.linalg.solve(A, b)
    except np.linalg.linalg.LinAlgError:
        return -1, -1
    else:
        assert np.isclose(x1 + m * dx1, x2 + n * dx2)
        assert np.isclose(y1 + m * dy1, y2 + n * dy2)
        return x1 + m * dx1, y1 + m * dy1


def two_norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1])


# @njit
def find_normals(P):
    # find pointing unit vectors assuming clock-wise points
    r10 = (P[1] - P[0]) / two_norm(P[1] - P[0])
    r21 = (P[2] - P[1]) / two_norm(P[2] - P[1])
    r32 = (P[3] - P[2]) / two_norm(P[3] - P[2])
    r03 = (P[0] - P[3]) / two_norm(P[0] - P[3])
    # rotate by 90deg to get normal vectors inward
    N0 = [-r03[1], r03[0]]
    N1 = [-r10[1], r10[0]]
    N2 = [-r21[1], r21[0]]
    N3 = [-r32[1], r32[0]]
    return (N0, N1, N2, N3)


# @njit
def map_uv_to_xy(u, v, P, N):
    """
    A[0, :] = u*N[2]-(1-u)*N[0]
    A[1, :] = v*N[3]-(1-v)*N[1]
    b[0] = u*P[2]@N[2]-(1-u)*P[0]@N[0]
    b[1] = v*P[3]@N[3]-(1-v)*P[0]@N[1]
    return np.linalg.solve(A, b)
    """
    nu = 1 - u
    nv = 1 - v
    """
    A =
    u*N[2][0]-nu*N[0][0]  u*N[2][1]-nu*N[0][1]
    v*N[3][0]-nv*N[1][0]  v*N[3][1]-nv*N[1][1]

    A_inv =
     v*N[3][1]-nv*N[1][1]  -u*N[2][1]+nu*N[0][1]
    -v*N[3][0]+nv*N[1][0]   u*N[2][0]-nu*N[0][0]
    
    """
    b_0 = u * (P[2][0] * N[2][0] + P[2][1] * N[2][1]) - nu * (
        P[0][0] * N[0][0] + P[0][1] * N[0][1]
    )
    b_1 = v * (P[3][0] * N[3][0] + P[3][1] * N[3][1]) - nv * (
        P[0][0] * N[1][0] + P[0][1] * N[1][1]
    )
    x = b_0 * (v * N[3][1] - nv * N[1][1]) + b_1 * (-u * N[2][1] + nu * N[0][1])
    y = b_0 * (-v * N[3][0] + nv * N[1][0]) + b_1 * (u * N[2][0] - nu * N[0][0])
    # print(u,v,'->',int(x),int(y))
    return x, y


# @njit
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


@njit
def get_square_impl(xx, yy, gray):
    out = np.empty(gray.shape, gray.dtype)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            out[i, j] = gray[yy[i, j], xx[i, j]]
    return out


# @profile
def get_square_image(gray, width_pixels, height_pixels, points):
    normals = find_normals(points)
    out = np.zeros((height_pixels, width_pixels))
    u, v, x_map, y_map = get_sample_xy_points(
        points, normals, 100, height_pixels, width_pixels
    )

    x_func = sp.interpolate.interp2d(u, v, x_map)
    y_func = sp.interpolate.interp2d(u, v, y_map)
    u = np.linspace(0, 1, width_pixels)
    v = np.linspace(0, 1, height_pixels)

    # uu, vv = np.meshgrid(u, v)

    xx = x_func(u, v)
    np.rint(xx, out=xx)
    np.clip(xx, a_min=0, a_max=None, out=xx)
    xx = xx.astype(np.int)

    yy = y_func(u, v)
    np.rint(yy, out=yy)
    np.clip(yy, a_min=0, a_max=None, out=yy)
    yy = yy.astype(np.int)
    return gray[yy, xx]
    # return gray[yy.flatten(), xx.flatten()].reshape((height_pixels, width_pixels))
    # return get_square_impl(xx, yy, gray)

    # print(xx.shape)
    # print(yy.shape)
    # for i in range(height_pixels):
    #     for j in range(width_pixels):
    #         # TODO this function does not vary quicky. Can I interpolate between sample points?
    #         xy = map_uv_to_xy(j/width_pixels, i/height_pixels, points, normals)
    #         if xy[0] > width_pixels:
    #             val = 0
    #         elif xy[1] > height_pixels:
    #             val = 0
    #         else:
    #             r = int(xy[1])
    #             c = int(xy[0])
    #             val = gray[r,c]
    #             out[i,j] = val

    # return out


# TODO
# * Segment out edges and set them to white
# * adaptive threshold that is consistent between nearby superpixels
@njit
def mad(x):
    return np.median(np.abs(x - np.median(x)))


@njit
def find_le_ix(a, x):
    "Find rightmost value less than or equal to x"
    i = np.searchsorted(a, x)
    if i:
        return i - 1
    raise ValueError


@njit
def moving_average(a, n):
    cumsum = np.cumsum(a)
    return (cumsum[n:] - cumsum[:-n]) / n


@njit
def estimate_threshold(x):
    x = np.sort(x)
    # x = sorted(vals)
    # x.sort()
    y = np.linspace(0, 1, len(x))
    a, b = [], []
    dy = 1 / len(x)
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        if dx == 0:
            dy += 1 / len(x)
        else:
            a.append(x[i])
            b.append(dy / dx)
            dy = 1 / len(x)

    a = np.array(a)
    b = np.array(b)
    bdiff = np.diff(b)

    s = mad(bdiff)
    bdiff = moving_average(bdiff, 5)

    medx = x[int(len(x) / 2)]
    ix = find_le_ix(a, medx)
    was_above = False
    maxb = np.max(b)
    while ix > 0:
        if abs(bdiff[ix]) < 3 * s:
            if was_above and b[ix] < maxb / 10:
                break
        else:
            was_above = True

        ix -= 1
    thresh = a[ix]
    return thresh


# @njit
# def project_down(rotated, data, data_row):
#     # projection = rotated.sum(axis=0)
#     for i in range(rotated.shape[0]):
#         for j in range(rotated.shape[1]):
#             data[data_row, j] += rotated[i, j]


# def mad(x):
# return np.median(np.abs(x-np.median(x)))
# @profile
def main():
    image = cv2.imread(sys.argv[1])
    original_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # .astype(np.float64)
    # TODO scale down image intelligently
    SHRINK_FACTOR = 8
    gray = block_reduce(original_gray, (SHRINK_FACTOR, SHRINK_FACTOR))
    im_rows, im_cols = gray.shape
    edged = scharr(gray)
    original_edged = edged
    edged = resize(edged)

    rows, cols = edged.shape
    angles = np.arange(0, 185, 0.5)

    data = np.zeros((len(angles), cols))
    for i, angle in enumerate(angles):
        rotated = imutils.rotate(edged, angle)
        assert rotated.shape[1] == cols
        # rotated =
        data[i, :] = rotated.sum(axis=0)
        # project_down(rotated, data, i)

    # TODO scale threshold dynamically to get a reasonable number of detections
    flat_data = data.flatten()
    peak_thresh = np.median(flat_data) + 3 * 1.4826 * mad(flat_data)
    xy = peak_local_max(data, min_distance=5, threshold_abs=peak_thresh)

    lines = []
    for rotation, extent in xy:
        score = data[rotation, extent]
        theta = np.radians(angles[rotation])
        dx = np.cos(theta)
        dy = np.sin(theta)
        a = extent - cols / 2

        x = a * dx + cols / 2 - (cols - im_cols) / 2
        y = a * dy + rows / 2 - (rows - im_rows) / 2
        r = np.array([dx, dy])
        R = np.array([[0, -1.0], [1.0, 0]])
        dx, dy = R @ r
        lines.append((x, y, dx, dy))

    intersection_points = []
    for l1, l2 in itertools.combinations(lines, 2):
        x, y = find_intercept(l1, l2)
        if x < 0 or x >= im_cols:
            continue
        if y < 0 or y >= im_rows:
            continue
        intersection_points.append((x, y))

    def gen_scored_points():
        for points in itertools.combinations(intersection_points, 4):
            points = list(map(np.array, sorted(points, key=lambda x: -sum(x))))
            sorted_points = [points.pop()[:2]]
            while len(points) > 0:
                points.sort(key=lambda x: -min(np.abs(np.array(x) - sorted_points[-1])))
                sorted_points.append(points.pop())

            sorted_points = [np.round(p).astype(int) for p in sorted_points]
            score = 0
            deltas = [sorted_points[0] - sorted_points[-1]]
            for i in range(4):
                p1 = sorted_points[i]
                p2 = sorted_points[(i + 1) % 4]

                delta = p2 - p1
                # filter out small edges
                # print('|delta|=',np.linalg.norm(delta))
                if np.linalg.norm(delta) < min(im_rows, im_cols) / 2:
                    score = 0
                    break

                # filter out acute angles
                angle = np.degrees(
                    np.arccos(
                        delta
                        @ deltas[-1]
                        / np.linalg.norm(delta)
                        / np.linalg.norm(deltas[-1])
                    )
                )
                # print('angle=', angle)
                if angle < 60 or angle > 120:
                    score = 0
                    break

                rr, cc, vals = line_aa(p1[1], p1[0], p2[1], p2[0])
                ix_1 = rr < im_rows - 1
                ix_2 = cc < im_cols - 1
                ix = ix_1 * ix_2
                score += np.sum(vals[ix] * original_edged[rr[ix], cc[ix]])
                deltas.append(delta)

            yield score, sorted_points, deltas

    score, sorted_points, deltas = max(gen_scored_points(), key=lambda x: x[0])
    print(sorted_points)
    # plt.imshow(gray, cmap=cm.gray, interpolation='nearest')
    # plt.show()
    print("score=", score)

    # undistort image
    width_over_height = 11 / 8.5
    # SHRINK_FACTOR = 1
    width_pixels = SHRINK_FACTOR * int(
        np.round(max(map(np.linalg.norm, deltas)))
    )  # assumes wide image
    height_pixels = SHRINK_FACTOR * int(
        np.round(max(map(np.linalg.norm, deltas)) / width_over_height)
    )
    sorted_points = [p.astype(float) * SHRINK_FACTOR for p in sorted_points]

    # TODO make this optional
    # color = ['r','b','g','k']
    # for i in range(4):
    #     base = wakka[i]+deltas[(i+1)%3]/2
    #     # print(normals[i])
    #     # xy =  # + normals[i]*np.array([0, 10])
    #     # print(xy)
    #     plt.plot([sorted_points[i][0]], [sorted_points[i][1]], linestyle='', marker='x', color=color[i])

    #     plt.plot(
    #         [base[0], base[0] + normals[i][0]*10],
    #         [base[1], base[1] + normals[i][1]*10],
    #         color=color[i]
    #     )
    # plt.show()

    # TODO make this a test
    # print(sorted_points[0])
    # print(sorted_points[1])
    # print(sorted_points[2])
    # print(sorted_points[3])
    # print(map_uv_to_xy(0, 0, *wakka, *normals))
    # print(map_uv_to_xy(1, 0, *wakka, *normals))
    # print(map_uv_to_xy(1, 1, *wakka, *normals))
    # print(map_uv_to_xy(0, 1, *wakka, *normals))

    out = get_square_image(original_gray, width_pixels, height_pixels, sorted_points)
    # f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    # ax1.imshow(out, cmap=cm.gray, interpolation='nearest')
    # out = denoise_tv_chambolle(out, weight=10, multichannel=False).astype(np.float64)

    # ax2.imshow(denoised, cmap=cm.gray, interpolation='nearest')
    # plt.show()
    # return
    segments = slic(out, multichannel=False, n_segments=30)
    segment_nums = list(range(segments.min(), segments.max() + 1))
    thresh_arr = np.array(out.shape, out.dtype)
    threshed = np.zeros(out.shape, np.bool)

    for seg in segment_nums:
        # print(seg)
        mask = segments == seg
        flattened_arr = out[mask]
        thresh = estimate_threshold(flattened_arr)
        print(seg, thresh)
        mask = (segments == seg) & (out > thresh)
        threshed[mask] = True

    # threshed = binary_opening(threshed, structure=np.ones((3,3))).astype(np.int)
    label_img = label(threshed, connectivity=1, background=1)

    # plt.imshow(label_img, cmap=cm.prism)
    # plt.show()
    props = regionprops(label_img)

    all_dark_area = sum(p.area for p in props)
    most_dark_areas = sorted(p.area for p in props)[-4:]
    for prop in props:
        if (prop.area > 0.10 * all_dark_area) and (prop.area in most_dark_areas):
            for row, col in prop.coords:
                threshed[row, col] = 1

    # plt.imshow(threshed, cmap=cm.gray, interpolation='nearest')
    rescaled = (255.0 * threshed).astype(np.uint8)

    im = Image.fromarray(rescaled)
    im.save("out.png")
    # plt.savefig('out.png')
    # plt.show()


if __name__ == "__main__":
    main()
