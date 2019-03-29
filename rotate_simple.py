import itertools
import argparse

import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt 
from skimage.filters import scharr
from skimage.draw import line_aa
from skimage.feature import peak_local_max

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


def map_uv_to_xy(u, v, P0, P1, P2, P3):
    # find pointing unit vectors assuming CCW points
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
    D=N0[0]+𝑁2[0]
    E=N0[1]+𝑁2[1]
    F=-P0@N0-P2@N2
    G=N1[0]
    H=N1[1]
    I=-P0@N1
    J=N1[0]+N3[0]
    K=N1[1]+N3[1]
    J=-P0@N1-P2@N3

    uDA=u*(D-A)
    uEB=u*(E-B)
    uFC=u*(F-C)
    vJG=v*(J-G)
    vKH=v*(K-H)
    vJG=v*(J-G)

    x=vKH@uFC-vLI@uEB - vJG@uEB-vKH@uDA
    y=vLI@uDA-uFC@vJG - vJG@uEB-vKH@uDA
    return x, y

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to the image file")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    im_rows,im_cols = gray.shape
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # edged = cv2.Canny(gray)
    edged = scharr(gray)
    # equivalent to scharr https://stackoverflow.com/questions/46140823/combined-scharr-derivatives-in-opencv
    # kernel = np.array([[-6, -10, 0],
    #                    [-10, 0, 10],
    #                    [0, 10, 6]])

    # edged = np.abs(cv2.filter2D(gray, -1, kernel))

    plt.imshow(edged)
    plt.show()
    original_edged = edged
    edged = resize(edged)

    rows,cols = edged.shape
    angles = np.arange(0, 185, 1.0)
    data = np.zeros((len(angles), cols))
    for i, angle in enumerate(angles):
        rotated = imutils.rotate(edged, angle)
        projection = rotated.sum(axis=0)
        assert projection.size == cols
        for j in range(cols):
            data[i, j] += projection[j]

    # TODO make thresh dependednt on image size?
    xy = peak_local_max(data, min_distance=5,threshold_abs=np.max(data)*.4)
    # plt.imshow(data)
    # plt.plot(*reversed(list(zip(*xy))),marker='x', linestyle='', color='r')
    # plt.show()

    asd = np.arange(-150, 150, 1)
    plt.imshow(image)
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
        # plt.plot(asd * dx + x, asd * dy + y, linestyle=':', color='r')
        # plt.plot([x], [y], linestyle='', marker='o', color='y')

    # plt.show()
    intersection_points = []
    scores = []
    for l1, l2 in itertools.combinations(lines, 2):
        x, y = find_intercept(l1, l2)
        if x < 0 or x > im_cols:
            continue
        if y < 0 or y > im_rows:
            continue
        # scores.append()
        intersection_points.append((x, y))

    # plt.plot(*zip(*intersection_points), linestyle='', marker='x', color='k')

    print(len(intersection_points))
    def gen_scored_points():
        for points in itertools.combinations(intersection_points, 4):

            # points = [np.array(p) for p in points]
            # print(points)
            # score = sum(p[2] for p in points)
            points = set(points)
            sorted_points = [np.array(points.pop()[:2])]
            # print(sorted_points)
            while len(points) > 0:
                next_point = min(points, key=lambda x: (x[0]-sorted_points[-1][0])**2+(x[1]-sorted_points[-1][1])**2)
                points.remove(next_point)
                # points.sort(key=lambda x: -np.linalg.norm(x, sorted_points[-1]))
                sorted_points.append(next_point)
            sorted_points = [np.round(p).astype(int) for p in sorted_points]
            # print(line(*sorted_points[0], *sorted_points[1]))
            score = 0
            last_delta = None
            # for p1, p2 in ():
            for i in range(4):
                p1 = sorted_points[i]
                p2 = sorted_points[(i+1)%4]
                # if np.linalg.norm(p2 - p1) < 10:
                #     score = 0
                #     break
                delta = p2 - p1
                print('|delta|=',np.linalg.norm(delta))
                if last_delta is not None:
                    angle = np.degrees(np.arccos(delta@last_delta/np.linalg.norm(delta)/np.linalg.norm(last_delta)))
                    print('angle=', angle)
                    if angle < 70 or angle > 110:
                        score = 0
                        break

                last_delta = delta
                # p1 = np.minimum(p1, )
                # p1 = (min(p1[0], im_cols-1), min(p1[1], im_rows-1))
                # p2 = (min(p2[0], im_cols-1), min(p2[1], im_rows-1))
                rr,cc,vals =line_aa(*p1, *p2)
                ix_1 = rr < im_rows-1
                ix_2 = cc < im_cols-1
                ix = ix_1 * ix_2
                # print(rr)
                # print(cc)
                # print(vals)
                score += np.sum(vals[ix] * original_edged[rr[ix], cc[ix]])

            # score += np.sum(edged[line(*sorted_points[0], *sorted_points[1])])
            # score += np.sum(edged[line(*sorted_points[1], *sorted_points[2])])
            # score += np.sum(edged[line(*sorted_points[2], *sorted_points[3])])
            # score += np.sum(edged[line(*sorted_points[3], *sorted_points[0])])
            # plt.plot(*zip(*(sorted_points+[sorted_points[0]])), linestyle=':', marker='', color='r')
            # print(score)
            yield score, sorted_points

    wakka = sorted(gen_scored_points(), key=lambda x: x[0])
    for score, sorted_points in wakka[-10:]:
        if score == 0:
            continue
        plt.imshow(original_edged)
        plt.plot(*zip(*(sorted_points+[sorted_points[0]])), linestyle=':', marker='', color='r')
        print('score=',score)
        plt.show()
    # plt.show()
if __name__ == '__main__':
    main()