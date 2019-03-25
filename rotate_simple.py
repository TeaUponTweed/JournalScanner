# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt 
from skimage.filters import scharr

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

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to the image file")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_rows,im_cols = gray.shape
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # edged = cv2.Canny(gray)
    edged = scharr(gray)
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
    xy = peak_local_max(data, min_distance=5,threshold_abs=np.max(data)*.5)
    plt.imshow(data)
    plt.plot(*reversed(list(zip(*xy))),marker='x', linestyle='', color='r')
    plt.show()

    asd = np.arange(-150, 150, 1)
    plt.imshow(image)
    for (rotation, extent) in xy:
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
        plt.plot(asd * dx + x, asd * dy + y, linestyle=':', color='r')
        plt.plot([x], [y], linestyle='', marker='o', color='y')

    plt.show()


if __name__ == '__main__':
    main()