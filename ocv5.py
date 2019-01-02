import numpy as np
import cv2
imgpath="/Users/Shruti/Documents/standard_test_images/lena_color_256.tif"
img = cv2.imread(imgpath)
outpath="/Users/Shruti/Documents/output/lena_color_256.jpg"

cv2.imshow('lena', img)
cv2.imwrite(outpath, img)
cv2.waitKey(0)
cv2.destroyAllWindows()