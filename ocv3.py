import numpy as np
import cv2
imgpath="/Users/Shruti/Downloads/standard_test_images/lena_color_256.tif"
img = cv2.imread(imgpath)
cv2.namedWindow('lena', cv2.WINDOW_NORMAL)
cv2.imshow('lena',img)
cv2.waitKey(0)
cv2.destroyWindow('lena')