import numpy as np
import time
from math import hypot, pi, cos, sin
from PIL import Image
 
 
def hough(im, ntx=460, mry=360):
    "Calculate Hough transform."
    nimx, mimy = im.shape
    mry = int(mry/2)*2          #Make sure that this is even
    him = np.ones((ntx, mry), np.uint8)*255

    rmax = hypot(nimx, mimy)
    dr = rmax / (mry/2)
    dth = pi / ntx
 
    for jx in range(nimx):
        for iy in range(mimy):
            col = im[jx, iy]
            if col == 255: continue
            for jtx in range(ntx):
                th = dth * jtx
                r = jx*cos(th) + iy*sin(th)
                iry = mry/2 + int(r/dr+0.5)
                him[int(jtx), int(iry)] -= 1
    return him
 
 
def test():
    "Test Hough transform with pentagon."
    im = Image.open("pentagon.png").convert("L")
    start = time.time()
    him = hough(im)
    end = time.time()
    print(end - start)
    him.save("ho5.bmp")
 
 
if __name__ == "__main__":
    test()
 