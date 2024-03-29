import math
import sys

import cv2
import numpy as np
import scipy as sp
from PIL import Image
from shapely.geometry import Polygon
from skimage.feature import canny
from skimage.filters import threshold_niblack, threshold_sauvola, unsharp_mask
from skimage.filters.rank import median
from skimage.morphology import binary_dilation, binary_opening, disk
from skimage.restoration import denoise_tv_chambolle

from scansnake import line_utils, rectification_utils, threshold_utils

PLOT_EXTRACTED_DOCUMENT = False


def decimate_image(image, max_side_length=256):
    shrink_factor = int(max(image.shape) / max_side_length)
    if shrink_factor > 1:
        image = image[::shrink_factor, ::shrink_factor]
    else:
        image = image.copy()
    return image, shrink_factor


def find_document(image):
    # find edges in image
    edges = canny(image, sigma=2)
    # widen edges so smooth out small kinks in paper
    edges = binary_dilation(edges, selem=np.ones((3, 3), np.bool))
    # find quadrilateral which overlaps with edges the most
    document_corners = line_utils.find_quadrilateral(edges)

    return document_corners


def rectify_document(image, document_corners):
    assert document_corners.shape == (5, 2)
    # make sure corners are ordered clockwise, assumed in get_square_image
    doc_poly = Polygon(document_corners)
    if doc_poly.exterior.is_ccw:
        document_corners = np.array(list(map(list, reversed(document_corners))))

    # estimate dpi and use that to determine width and height
    doc_area = doc_poly.area
    dpi = math.sqrt(doc_area / (8.5 * 11))

    # determine if the found coordinate system has the image as "thin" or "wide"
    u_mag1 = np.linalg.norm(document_corners[1] - document_corners[0])
    v_mag1 = np.linalg.norm(document_corners[2] - document_corners[1])
    u_mag2 = np.linalg.norm(document_corners[3] - document_corners[2])
    v_mag2 = np.linalg.norm(document_corners[0] - document_corners[3])
    u_mag = (u_mag1 + u_mag2) / 2
    v_mag = (v_mag1 + v_mag2) / 2
    if u_mag > v_mag:  # wide
        width_pixels = int(dpi * 11)
        height_pixels = int(dpi * 8.5)
    else:  # thin
        width_pixels = int(dpi * 8.5)
        height_pixels = int(dpi * 11)

    rectified_image = rectification_utils.get_square_image(
        image, width_pixels, height_pixels, document_corners
    )
    # transform to thin image, might be upside down
    if rectified_image.shape[0] < rectified_image.shape[1]:
        rectified_image = rectified_image.T

    return rectified_image


def threshold_image(image):
    # scale filters based on estimated pixel density
    dpi = np.mean([image.shape[0] / 8.5, image.shape[1] / 11])
    window_size = int(dpi / 30)

    # threshold
    ## 73.6 ~= (1/12 (255 - 0)**2)**.5, the standard deviation of a uniform distribion with range [0,255]
    thresh = threshold_sauvola(
        image, window_size=2 * window_size + 1, k=0.1, r=73.6121593217
    )
    # niblack looks better in the region of the text, and handles color better, but can't handle the background
    # thresh = threshold_niblack(image, window_size = 2*2*window_size+1)

    image = ((image > thresh) * 255).astype(np.uint8)

    return image


def main(image_file, outfile=None):
    # load in image
    image = cv2.imread(sys.argv[1])
    # convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # decimate image for performance reasons
    downsampled_image, decimation_factor = decimate_image(image)
    # find document in image
    document_corners = find_document(downsampled_image)
    # append first point as last to make a closed shape
    document_corners = np.array(
        [list(xy) for xy in document_corners] + [list(document_corners[0])]
    )

    if PLOT_EXTRACTED_DOCUMENT:
        plt.imshow(image, cmap="gray", vmin=0, vmax=255)
        x, y = zip(*(decimation_factor * document_corners))
        plt.plot(x, y, color="r", marker="x")
        plt.xlim(0, image.shape[1])
        plt.ylim(0, image.shape[0])
        plt.show()

    # extract document into an approximate rectangle
    document_image = rectify_document(image, decimation_factor * document_corners)

    # threshold into a "scanned" image
    document_image = threshold_image(document_image)
    if outfile is not None:
        im = Image.fromarray(document_image)
        im.save(outfile)
    else:
        import matplotlib.pyplot as plt

        plt.imshow(image, cmap="gray", vmin=0, vmax=255)
        plt.show()


def cli():
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
