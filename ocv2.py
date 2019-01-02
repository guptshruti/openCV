import numpy as np
import cv2
imgpath="/Users/Shruti/Downloads/standard_test_images/house.tif"
img = cv2.imread(imgpath)
cv2.imshow('house',img)
cv2.waitKey(0)
cv2.destroyAllWindows()