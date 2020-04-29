import sys

import cv2
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt 
import numpy as np
import scipy as sp
from skimage.feature import canny
from skimage.morphology import binary_dilation

import line_utils
import rectification_utils
import threshold_utils

PLOT_EXTRACTED_DOCUMENT = False
DOCUMENT_ASPECT = 11/8.5

def decimate_image(image, max_side_length=256):
    shrink_factor = int(max(image.shape)/max_side_length)
    if shrink_factor > 1:
        image = image[::shrink_factor, ::shrink_factor]
    else:
        image = image.copy()
    return image, shrink_factor

def find_document(image):
    # find edges in image
    edges = canny(image, sigma=2)
    # widen edges so smooth out small kinks in paper
    edges = binary_dilation(edges, selem=np.ones((3,3), np.bool))
    # find quadrilateral which overlaps with edges the most
    document_corners = line_utils.find_quadrilateral(edges)

    return document_corners

def rectify_document(image, document_corners):
    min_x = np.min(document_corners[:, 0])
    max_x = np.max(document_corners[:, 0])
    min_y = np.min(document_corners[:, 1])
    max_y = np.max(document_corners[:, 1])
    dy = max_y - min_y
    dx = max_x - min_x
    if dy > dx: # tall skinny image
        width_over_height = 1/DOCUMENT_ASPECT
    else: # short fat image
        width_over_height = DOCUMENT_ASPECT
    # print(document_corners)
    deltas = np.linalg.norm(document_corners[:-1] - document_corners[1:], axis=1)
    # print(deltas)
    width_pixels = int(np.round(max(map(np.linalg.norm, deltas)))) # assumes wide image
    height_pixels = int(np.round(max(map(np.linalg.norm, deltas))/width_over_height))
    # sorted_points = [p.astype(float)*SHRINK_FACTOR for p in sorted_points]
    rectified_image = rectification_utils.get_square_image(image, width_pixels, height_pixels, document_corners[:4])
    return rectified_image

def threshold_image(image):
    return threshold_utils.threshold_image(image)

def main(image_file):
    # load in image
    image = cv2.imread(sys.argv[1])
    # convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # decimate image for performance reasons
    downsampled_image, decimation_factor = decimate_image(image)
    # find document in image
    document_corners = find_document(downsampled_image)
    document_corners = np.array([list(xy) for xy in document_corners] + [list(document_corners[0])])
    if PLOT_EXTRACTED_DOCUMENT:
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        x, y = zip(*(decimation_factor*document_corners))
        # x, y = [*x, x[0]], [*y, y[0]]
        plt.plot(x, y, color='r', marker='x')
        plt.xlim(0, image.shape[1])
        plt.ylim(0, image.shape[0])
        plt.show()

    document_image = rectify_document(image, decimation_factor*document_corners)
    plt.imshow(document_image, cmap='gray', vmin=0, vmax=255)
    plt.show()
    document_image = threshold_image(document_image)
    plt.imshow(document_image, cmap='gray', vmin=0, vmax=255)
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1])
