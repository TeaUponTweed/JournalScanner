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

# @njit
def map_uv_to_xy(u, v, P0, P1, P2, P3):
    # find pointing unit vectors assuming clock-wise points
    r0 = (P1 - P0)/np.linalg.norm(P1 - P0)
    r1 = (P2 - P1)/np.linalg.norm(P2 - P1)
    r2 = (P3 - P2)/np.linalg.norm(P3 - P2)
    r3 = (P0 - P3)/np.linalg.norm(P0 - P3)
    # rotate by 90deg to get normal vectors inward
    N0 = np.array([-r0[1],r0[0]])
    N1 = np.array([-r1[1],r1[0]])
    N2 = np.array([-r2[1],r2[0]])
    N3 = np.array([-r3[1],r3[0]])
    # do math https://math.stackexchange.com/questions/13404/mapping-irregular-quadrilateral-to-a-rectangle
    A=N0[0]
    B=N0[1]
    C=-P0@N0
    D=N0[0]+N2[0]
    E=N0[1]+N2[1]
    F=-P0@N0-P2@N2

    G=N1[0]
    H=N1[1]
    I=-P0@N1
    J=N1[0]+N3[0]
    K=N1[1]+N3[1]
    L=-P0@N1-P2@N3

    uDA=u*(D-A)
    uEB=u*(E-B)
    uFC=u*(F-C)

    vJG=v*(J-G)
    vKH=v*(K-H)
    vLI=v*(L-I)
    print(vJG*uEB)
    print(vKH*uDA)
    print('***')
    if np.isclose(vJG*uEB-vKH*uDA, 0):
        return 0, 0
    x=(vKH*uFC-vLI*uEB)/(vJG*uEB-vKH*uDA)
    y=(vLI*uDA-uFC*vJG)/(vJG*uEB-vKH*uDA)
    return x, y

# @njit
def map_uv_to_xy(u, v, P0, P1, P2, P3):
    # find pointing unit vectors assuming clock-wise points
    r0 = (P1 - P0)/np.linalg.norm(P1 - P0)
    r1 = (P2 - P1)/np.linalg.norm(P2 - P1)
    r2 = (P3 - P2)/np.linalg.norm(P3 - P2)
    r3 = (P0 - P3)/np.linalg.norm(P0 - P3)
    # rotate by 90deg to get normal vectors inward
    N0 = -np.array([-r0[1],r0[0]])
    N1 = -np.array([-r1[1],r1[0]])
    N2 = -np.array([-r2[1],r2[0]])
    N3 = -np.array([-r3[1],r3[0]])

    # x = (P0[0]*N0[0] - u*(P0[0]*N0[0]+P2[0]*N2[0]))/(N0[0]-u*(N0[0]+N2[0])) + (P0[0]*N1[0] - v*(P0[0]*N1[0]+P2[0]*N3[0]))/(N1[0]-v*(N1[0]+N3[0]))
    # y = (P0[1]*N0[1] - u*(P0[1]*N0[1]+P2[1]*N2[1]))/(N0[1]-u*(N0[1]+N2[1])) + (P0[1]*N1[1] - v*(P0[1]*N1[1]+P2[1]*N3[1]))/(N1[1]-v*(N1[1]+N3[1]))
    A = np.array([
        [N0[0]-u*(N0[0]+N2[0]), N1[0]-v*(N1[0]+N3[0])],
        [N0[1]-u*(N0[1]+N2[1]), N1[1]-v*(N1[1]+N3[1])]
    ])
    b = np.array([
        P0[0]*N0[0] - u*(P0[0]*N0[0]+P2[0]*N2[0]) + P0[0]*N1[0] - v*(P0[0]*N1[0]+P2[0]*N3[0]),
        P0[1]*N0[1] - u*(P0[1]*N0[1]+P2[1]*N2[1]) + P0[1]*N1[1] - v*(P0[1]*N1[1]+P2[1]*N3[1]),
    ])
    return np.linalg.solve(A, b)
    # return x, y

def main():
    image = cv2.imread(sys.argv[1])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # TODO scale down image intelligently
    gray = block_reduce(gray, (8, 8))
    print(gray.shape)
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
    plt.plot(*zip(*(sorted_points+[sorted_points[0]])), linestyle=':', marker='', color='r')
    plt.show()

    # undistort image
    width_over_height = 11/8.5
    width_pixels = int(np.round(max(map(np.linalg.norm, deltas)))) # assumes wide image
    height_pixels = int(np.round(max(map(np.linalg.norm, deltas))/width_over_height))
    # print(width_pixels)
    # print(height_pixels)
    # to_interp = np.zeros((height_pixels, width_pixels, 2))
    wakka = [p.astype(float) for p in sorted_points]
    out = np.zeros((height_pixels, width_pixels))
    for i in range(height_pixels):
        # print(i)
        to_interp = []

        for j in range(width_pixels):
            x, y = map_uv_to_xy(i/height_pixels, j/width_pixels, *wakka)
            # x, y = blarg(j/width_pixels, i/height_pixels, *wakka)
            # print(x, y)
            to_interp.append([x, y])
            # to_interp[i, j, :] = (x, y)
        # print(to_interp)
        # to_interp = np.array(to_interp)
        # print(to_interp.shape)
        # print(gray.shape)

        out[i,:] = map_coordinates(gray, list(zip(*to_interp)), order=1)
    plt.imshow(out)
    plt.show()


if __name__ == '__main__':
    main()
