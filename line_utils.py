import itertools

import numpy as np
from skimage.draw import line_aa
from skimage.transform import hough_line, hough_line_peaks


def _find_intercept(l1, l2):
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

def find_quadrilateral(edges, debug_plotting=False):
    # find lines in edge image
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360*4)
    h, theta, d = hough_line(edges, theta=tested_angles)
    origin = np.array([0, edges.shape[1]])
    peaks = hough_line_peaks(h, theta, d)
    if debug_plotting:
        plt.imshow(edges)
        for _, angle, dist in zip(*peaks):
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            plt.plot(origin, (y0, y1), '-r')
        plt.xlim(0, edges.shape[1])
        plt.ylim(0, edges.shape[0])
        plt.show()
    # use lines to find document
    return _find_quadrilateral(peaks, origin, edges)


def _find_quadrilateral(peaks, origin, score_image):
    lines = []
    # plt.imshow(score_image)
    for (score, angle, dist) in zip(*peaks):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        x0, x1 = origin
        dx = x1 - x0
        dy = y1 - y0
        lines.append((x1,y1,dx,dy))
    #     plt.plot(origin, (y0, y1), '-r')
    # plt.xlim(0, score_image.shape[1])
    # plt.ylim(0, score_image.shape[0])
    # plt.show()
    # TODO add the edges of the score_image as candidate lines
    intersection_points = []
    for l1, l2 in itertools.combinations(lines, 2):
        x, y = _find_intercept(l1, l2)
        if x < -score_image.shape[1] or x >= 2*score_image.shape[1]:
            continue
        if y < -score_image.shape[0] or y >= 2*score_image.shape[0]:
            continue
        intersection_points.append((x, y))
    # print(intersection_points)
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
                if np.linalg.norm(delta) < min(score_image.shape)/4:
                    score = 0
                    break

                # filter out acute angles
                angle = np.degrees(np.arccos(delta@deltas[-1]/np.linalg.norm(delta)/np.linalg.norm(deltas[-1])))
                if angle < 60 or angle > 120:
                    score = 0
                    break
                # TODO make sure quadrilateral is convex

                rr,cc,vals =line_aa(p1[1], p1[0], p2[1], p2[0])
                ix_1 = rr < score_image.shape[0]-1
                ix_1 *= rr >= 0
                ix_2 = cc < score_image.shape[1]-1
                ix_2 *= cc >= 0
                ix = ix_1 * ix_2
                score += np.sum(vals[ix] * score_image[rr[ix], cc[ix]])
                deltas.append(delta)

            yield score, sorted_points, deltas

    score, sorted_points, deltas = max(gen_scored_points(), key=lambda x: x[0])

    return sorted_points
