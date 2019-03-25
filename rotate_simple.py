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
    # print(gray.shape)
    # gray = resize(gray)
    # plt.imshow(gray)
    # plt.show()

    # edged = cv2.Canny(gray)
    edged = scharr(gray)
    edged = resize(edged)
    # plt.imshow(edged)
    # plt.show()

    rows,cols = edged.shape
    angles = np.arange(0, 185, 1.0)
    data = np.zeros((len(angles), cols))
    for i, angle in enumerate(angles):
        # rotated = imutils.rotate_bound(edged, angle)
        rotated = imutils.rotate(edged, angle)
        # print(rotated.shape)
        wakka = rotated.sum(axis=0)
        # print(wakka.size)
        assert wakka.size == cols
        for j in range(cols):
            # TODO figure out extent dimension for scaling and aliasing
            data[i, j] += wakka[j]
    # plt.hist(data.flatten())
    # plt.show()
    # TODO make thresh dependednt on image size?
    xy = peak_local_max(data, min_distance=5,threshold_abs=np.max(data)*.5)
    # print(xy)
    plt.imshow(data)
    plt.plot(*reversed(list(zip(*xy))),marker='x', linestyle='', color='r')
    plt.show()

    asd = np.arange(-150, 150, 1)
    plt.imshow(image)
    for (rotation, extent) in xy:
        # s_c = im_cols/cols
        # s_r = im_rows/rows
        # extent = extent * cols / len(angles)
        print(extent, rotation)
        theta = np.radians(angles[rotation])
        # print(theta)
        dx = np.cos(theta)
        dy = np.sin(theta)
        a = extent - cols/2

        x = a*dx + cols/2-(cols-im_cols)/2
        y = a*dy + rows/2-(rows-im_rows)/2
        # plt.plot([x], [y], linestyle='', marker='o', color='y')
        # plt.plot(asd *dx + x, asd * dy + y, linestyle=':', color='r')
        r = np.array([dx, dy])
        # r = r / np.linalg.norm(r)
        R = np.array([
            [0, -1.0],
            [1.0, 0]
        ])
        dx,dy = R @ r
        plt.plot(asd * dx + x, asd * dy + y, linestyle=':', color='r')
        plt.plot([x], [y], linestyle='', marker='o', color='y')

        '''
        dx = np.sin(theta)
        dy = np.cos(theta)
        a = extent - cols/2
        b = a * sin(theta)
        x = extent
        y = rows/2 + (extent-cols/2) / np.sin(theta) * np.cos(theta)

        plt.plot([x], [y], linestyle='', marker='o', color='y')
        plt.plot(asd * dx + x, asd * dy + y, linestyle=':', color='r')
        '''
    plt.show()

    if False:
        for angle in np.arange(0, 100, 10):
            # rotated = imutils.rotate_bound(edged, angle)
            rotated = imutils.rotate(edged, angle)
            wakka = rotated.sum(axis=0)
            a = sum(([i/len(wakka)]*wakka[i] for i in range(len(wakka))), [])
            plt.hist(a, density=True, label="angle = " + str(angle))


        plt.legend()
        plt.show()



def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

if __name__ == '__main__':
    main()