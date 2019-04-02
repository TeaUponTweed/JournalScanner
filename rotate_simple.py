import sys
import itertools

import numpy as np
from scipy.ndimage.interpolation import map_coordinates

import imutils
import cv2
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

from skimage.filters import scharr
from skimage.draw import line_aa
from skimage.feature import peak_local_max
from skimage.measure import block_reduce

from numba import njit, jit

def resize(im):
    desired_size = max(im.shape)
    delta_w = desired_size - im.shape[1]
    delta_h = desired_size - im.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im

def find_intercept(l1, l2):
    (x1, y1, dx1, dy1) = l1
    (x2, y2, dx2, dy2) = l2
    A = np.array([[dx1, -dx2],[dy1,-dy2]])
    b = np.array([x2-x1, y2-y1])
    try:
        m, n = np.linalg.solve(A, b)
    except np.linalg.linalg.LinAlgError:
        return -1, -1
    else:
        assert np.isclose(x1+m*dx1, x2+n*dx2)
        assert np.isclose(y1+m*dy1, y2+n*dy2)
        return x1+m*dx1, y1+m*dy1

@njit
def find_normals(P):
    # find pointing unit vectors assuming clock-wise points
    r10 = (P[1] - P[0])/np.linalg.norm(P[1] - P[0])
    r21 = (P[2] - P[1])/np.linalg.norm(P[2] - P[1])
    r32 = (P[3] - P[2])/np.linalg.norm(P[3] - P[2])
    r03 = (P[0] - P[3])/np.linalg.norm(P[0] - P[3])
    # rotate by 90deg to get normal vectors inward
    N0 = np.array([-r03[1],r03[0]])
    N1 = np.array([-r10[1],r10[0]])
    N2 = np.array([-r21[1],r21[0]])
    N3 = np.array([-r32[1],r32[0]])
    return [N0, N1, N2, N3]

@njit
def map_uv_to_xy(u, v, P, N):
    A = np.zeros((2,2))
    b = np.zeros(2)
    A[0, :] = u*N[2]-(1-u)*N[0]
    A[1, :] = v*N[3]-(1-v)*N[1]
    b[0] = u*P[2]@N[2]-(1-u)*P[0]@N[0]
    b[1] = v*P[3]@N[3]-(1-v)*P[0]@N[1]
    return np.linalg.solve(A, b)

@njit(parallel=True)
def get_square_image(gray, width_pixels, height_pixels, points):
    normals = find_normals(points)
    out = np.zeros((height_pixels, width_pixels))
    for i in range(height_pixels):
        for j in range(width_pixels):
            # TODO this function does not vary quicky. Can I interpolate between sample points?
            xy = map_uv_to_xy(j/width_pixels, i/height_pixels, points, normals)
            if xy[0] > width_pixels:
                val = 0
            elif xy[1] > height_pixels:
                val = 0
            else:
                r = int(xy[1])
                c = int(xy[0])
                val = gray[r,c]
                out[i,j] = val

    return out


def main():
    image = cv2.imread(sys.argv[1])
    original_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # TODO scale down image intelligently
    SHRINK_FACTOR=8
    gray = block_reduce(original_gray, (SHRINK_FACTOR, SHRINK_FACTOR))
    im_rows,im_cols = gray.shape
    edged = scharr(gray)
    original_edged = edged
    edged = resize(edged)

    rows,cols = edged.shape
    angles = np.arange(0, 185, 0.5)

    data = np.zeros((len(angles), cols))
    for i, angle in enumerate(angles):
        rotated = imutils.rotate(edged, angle)
        projection = rotated.sum(axis=0)
        assert projection.size == cols
        for j in range(cols):
            data[i, j] += projection[j]

    # TODO scale threshold dynamically to get a reasonable number of detections
    xy = peak_local_max(data, min_distance=5,threshold_abs=np.max(data)*.4)

    lines = []
    for (rotation, extent) in xy:
        score = data[rotation, extent]
        theta = np.radians(angles[rotation])
        dx = np.cos(theta)
        dy = np.sin(theta)
        a = extent - cols/2

        x = a*dx + cols/2-(cols-im_cols)/2
        y = a*dy + rows/2-(rows-im_rows)/2
        r = np.array([dx, dy])
        R = np.array([
            [0, -1.0],
            [1.0, 0]
        ])
        dx,dy = R @ r
        lines.append((x,y,dx,dy))

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
                points.sort(key=lambda x: -min(np.abs(np.array(x)-sorted_points[-1])))
                sorted_points.append(points.pop())

            sorted_points = [np.round(p).astype(int) for p in sorted_points]
            score = 0
            deltas = [sorted_points[0] - sorted_points[-1]]
            for i in range(4):
                p1 = sorted_points[i]
                p2 = sorted_points[(i+1)%4]

                delta = p2 - p1
                # filter out small edges
                # print('|delta|=',np.linalg.norm(delta))
                if np.linalg.norm(delta) < min(im_rows, im_cols)/2:
                    score = 0
                    break

                # filter out acute angles
                angle = np.degrees(np.arccos(delta@deltas[-1]/np.linalg.norm(delta)/np.linalg.norm(deltas[-1])))
                # print('angle=', angle)
                if angle < 60 or angle > 120:
                    score = 0
                    break

                rr,cc,vals =line_aa(p1[1], p1[0], p2[1], p2[0])
                ix_1 = rr < im_rows-1
                ix_2 = cc < im_cols-1
                ix = ix_1 * ix_2
                score += np.sum(vals[ix] * original_edged[rr[ix], cc[ix]])
                deltas.append(delta)

            yield score, sorted_points, deltas


    score, sorted_points, deltas = max(gen_scored_points(), key=lambda x: x[0])
    print(sorted_points)
    plt.imshow(gray, cmap=cm.gray, interpolation='nearest')
    print('score=',score)

    # undistort image
    width_over_height = 11/8.5
    width_pixels = SHRINK_FACTOR*int(np.round(max(map(np.linalg.norm, deltas)))) # assumes wide image
    height_pixels = SHRINK_FACTOR*int(np.round(max(map(np.linalg.norm, deltas))/width_over_height))
    sorted_points = [p.astype(float)*SHRINK_FACTOR for p in sorted_points]

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

    plt.imshow(out, cmap=cm.gray, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    main()
