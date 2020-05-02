import math
import sys

import cv2
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt 
import numpy as np
import scipy as sp
from skimage.feature import canny
from skimage.filters import unsharp_mask, threshold_sauvola
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters.rank import median
from skimage.morphology import disk
from shapely.geometry import Polygon

import line_utils
import rectification_utils
import threshold_utils

PLOT_EXTRACTED_DOCUMENT = True
DOCUMENT_ASPECT = 11/8.5

def decimate_image(image, max_side_length=256):
    shrink_factor = int(min(image.shape)/max_side_length)
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
    # if dy > dx: # tall skinny image
    #     image = np.fliplr(image.T)

        # width_over_height = 1/DOCUMENT_ASPECT
    # else: # short fat image
    width_over_height = DOCUMENT_ASPECT
    # print(document_corners)
    doc_poly = Polygon(document_corners)

    if doc_poly.bounds[0] > doc_poly.bounds[1]:
        image = np.fliplr(image.T)
        document_corners = np.array([[b,a] for a, b in document_corners])

    doc_poly = Polygon(document_corners)
    if not doc_poly.exterior.is_ccw:
        document_corners = np.array(list(reversed(document_corners)))

    doc_area = doc_poly.area

    # deltas = np.linalg.norm(document_corners[:-1] - document_corners[1:], axis=1)
    # print(deltas)
    # width_pixels = int(np.round(max(map(np.linalg.norm, deltas)))) # assumes wide image
    # height_pixels = int(np.round(max(map(np.linalg.norm, deltas))/width_over_height))
    # width_pixels = doc_area
    dpi = math.sqrt(doc_area/(8.5*11))
    # print('wakka dpi', dpi)
    # transmorms into short fat image
    width_pixels = int(dpi * 11)
    height_pixels = int(dpi * 8.5)
    # sorted_points = [p.astype(float)*SHRINK_FACTOR for p in sorted_points]
    rectified_image = rectification_utils.get_square_image(image, width_pixels, height_pixels, document_corners[:4])
    if rectified_image.shape[0] < rectified_image.shape[1]:
        rectified_image = np.fliplr(rectified_image.T)
    return rectified_image, dpi

def threshold_image(image):
    dpi = np.mean([image.shape[0]/8.5, image.shape[1]/11])
    window_size = int(dpi/30)

    # if window_size % 2 == 0:
    #     window_size += 1
    print(f'dpi = {dpi}, window_size = {window_size}')

    # remove some noise
    image = median(image, disk(2))
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)

    # sharpen document to make text stand out
    image = unsharp_mask(image, radius=window_size, amount=5)
    # convert back to 8 bit grayscale image
    image*=255
    image = image.astype(np.uint8)
    thresh = threshold_sauvola(image, window_size = 2*window_size+1)
    image = ((image > thresh) * 255).astype(np.uint8)
    image = median(image, disk(2))
    return image

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
    # extract document into an approximate rectangle
    document_image, _ = rectify_document(image, decimation_factor*document_corners)
    plt.imshow(document_image, cmap='gray', vmin=0, vmax=255)
    plt.show()
    document_image = threshold_image(document_image)
    plt.imshow(document_image, cmap='gray', vmin=0, vmax=255)

    plt.show()


if __name__ == '__main__':
    main(sys.argv[1])
