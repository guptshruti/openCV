
import numpy as np
import cv2

imgpath="/Users/Shruti/Documents/datasets/att_faces/s1/1.pgm"
img = cv2.imread(imgpath)
cv2.line(img,(0,0),(90, 1),(255,255,0),5) #showing colored line (image,1st coordinate, 2nd, color, length)
cv2.rectangle(img,(0, 74),(90, 68),(0,0,255),2) # rectangle(image, left bottom corner, right bottom corner, color, width)
cv2.circle(img,(20,25), 15, (0,255,0),-1) #circle(image, centre, radius, color, )
poly = np.array([[15,6],[25,40],[45,80],[20,10]], np.int32)
cv2.polylines(img, [poly], True, (0,255,255), 3) #polygon
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Shruti',(1, 60), font, 1, (0,0,0), 2, cv2.LINE_AA)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()