import time
import math

import numpy as np

from scipy.misc import imrotate

from skimage import data
from skimage.feature import canny
from skimage.filters import scharr, gaussian
from skimage.transform import integral_image, probabilistic_hough_line, hough_line, hough_line_peaks

import statsmodels.api as sm
from scipy import stats
from scipy.signal import argrelextrema

import matplotlib.pyplot as plt
# import matplotlib
# from matplotlib import cm
from PIL import Image

from bresenham import bresenham

from cem import CEMMinimizer, make_uniform, infer_uniform, make_normal, infer_normal
from h2 import hough






def get_rotated_integral_image(im, deg):
    # TODO this might break at the edges as the true image is rotated off
    return integral_image(imrotate(im, deg, 'nearest'))

def compute_integral_beween(im, x1, y1, x2, y2):
    # print(x1)
    # print(y1)
    theta = math.atan2(y2 - y1, x2-x1)
    theta_deg = round(math.degrees(theta))
    #TODO cache this result
    # im = get_rotated_integral_image(im, theta_deg)
    return sum(im[i, j] for j, i in bresenham(x1, y1, x2, y2))

def find_perp_distance(p0, p1, parr, perp, sign):
    d01 = p1 - p0
    # d01 = np.linalg.norm(p1 - p0)
    # n01 = n01 / d01
    d01perp = np.array([[0., -1.], [1., 0.]]) @ d01
    # assert n01perp.T @ n01 == 0
    return p0 + d01*parr + sign*d01perp*perp

def test():

    # im = Image.open("pentagon.png").convert("L")
    im = np.array(Image.open("/Users/MichaelMason/Desktop/original-small.jpg").convert("L"))
    edges = gaussian(scharr(im))
    # edges = scharr(im)
    nrows, ncols = edges.shape
    def get_points(x):
        x0, y0, x1, y1, x2, y2, x3, y3 = x
        x0 = math.floor(ncols * x0)
        x1 = math.floor(ncols * x1)
        y0 = math.floor(nrows * y0)
        y1 = math.floor(nrows * y1)

        p0 = np.array([x0, y0], dtype=np.int64)
        p1 = np.array([x1, y1], dtype=np.int64)
        x2, y2 = find_perp_distance(p0, p1, x2, y2,  1)
        x3, y3 = find_perp_distance(p0, p1, x3, y3, -1)
        p2 = np.array([x2, y2], dtype=np.int64)
        p3 = np.array([x3, y3], dtype=np.int64)

        return p0, p2, p1, p3

    def f(x):
        p0, p1, p2, p3 = get_points(x)
        for p in (p0, p1, p2, p3):
            if not ((0 <= p[0] < ncols) and (0 <= p[1] < nrows)):
                return float('inf')

        edgesum = 0
        for a, b in ((p0, p1), (p1, p2), (p2, p3), (p3, p0)):
            edgesum += compute_integral_beween(edges, *a, *b)
        return -edgesum

    minimizer = CEMMinimizer(distribution=make_uniform(np.zeros(8), np.ones(8)), niter=50, infer_distribtion=infer_uniform, verbose=True, nsamples=500, relite=0.1)
    # minimizer = CEMMinimizer(distribution=make_normal(np.ones(8)*.5, np.eye(8)), niter=50, infer_distribtion=infer_normal, verbose=True, nsamples=500, relite=0.1)
    x = minimizer.minimize(f)
    plt.imshow(edges)
    points = get_points(x)
    plt.plot(*zip(*[*points, points[0]]))
    # plt.plot(*zip(*points))
    plt.show()

def test2():
    # im = np.array(Image.open("pentagon.png").convert("L"))
    im = np.array(Image.open("/Users/MichaelMason/Desktop/original-small.jpg").convert("L"))

    # edges = canny(im, 2, 1, 25)
    edges = scharr(im)
    # him = hough(edges)
    # him = hough_line(edges)
    h, theta, d = hough_line(edges)
    # print(np.min(edges))
    # print(np.max(edges))
    # for thresh in (0.1, 0.2, 0.3, 0.4):
    #     plt.imshow(edges > thresh)
    #     plt.show()
    # plt.imshow(h)
    # plt.show()

    plt.imshow(edges)
    for acc, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=1, min_angle=1, num_peaks=16)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - im.shape[1] * np.cos(angle)) / np.sin(angle)
        plt.plot((0, im.shape[1]), (y0, y1), '-r')
    plt.show()
    # lines = probabilistic_hough_line(edges, threshold=5, line_length=200,
    #                              line_gap=0)
    # plt.imshow(edges)
    # for line in lines:
    #     p0, p1 = line
    #     plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
    # plt.show()


def test3():
    nobs = 300
    s = np.array(sorted(np.array(list(np.random.normal(size=nobs)) + list(np.random.normal(size=nobs)+10))))
    dens = sm.nonparametric.KDEUnivariate(s)
    dens.fit()
    e = np.array([dens.evaluate(ss) for ss in s])
    ma = argrelextrema(e, np.greater)[0]

    plt.plot(s, e)
    plt.plot(s, stats.norm.pdf(s) + stats.norm.pdf(s, 10))

    for ix in ma:
        plt.plot(s[ix], e[ix], 'ro')
    plt.plot()

    plt.show()


def kdecluster(x):
    x = np.array(sorted(x))
    dens = sm.nonparametric.KDEUnivariate(x)
    dens.fit()
    y = np.array([dens.evaluate(xx) for xx in x])
    f, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()
    extrema_ixs = argrelextrema(y, np.greater)[0]
    return [x[ix] for ix in extrema_ixs]


def test4():
    # im = np.array(Image.open("pentagon.png").convert("L"))
    im = np.array(Image.open("/Users/MichaelMason/Desktop/original-small.jpg").convert("L"))
    n, m = im.shape
    edges = scharr(im)
    # edges = gaussian(scharr(im))
    plt.imshow(edges)

    ixs = list(bresenham(0, n//2, m-1,  n//2))
    line = [float(ix) for ix, (j, i) in enumerate(ixs) if edges[i, j] > 0.1]
    # line = [edges[i,j] for ix, (j, i) in enumerate(ixs)]
    cluster_ixs = kdecluster(line)
    print(cluster_ixs)
    for i in cluster_ixs:
        plt.plot(*ixs[int(np.round(i))], marker='x', color='k')

    # plt.plot(*zip(*bresenham(0, n//2, m,  n//2)))
    plt.show()


if __name__ == '__main__':
    # test()
    # test2()
    # test3()
    test4()
