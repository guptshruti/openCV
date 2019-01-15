import numpy as np
import cv2
imgpath="/Users/Shruti/Documents/datasets/standard_test_images/lena.png"
img = cv2.imread(imgpath,0)
a=np.shape(img)
print(a)
result=np.array(a).flatten()
print(result)