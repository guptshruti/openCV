import numpy as np
import cv2
import glob
n = 10304
C = np.zeros((n,n))
   
for img in glob.glob("/Users/Shruti/Documents/datasets/att_faces/s1/*.pgm"):
   cv_img = cv2.imread(img,0)
   #print(cv_img)
   a=np.shape(cv_img) #dimension checking
   #print(a)
   Ivec=cv_img.flatten() #conversion  of image in vector form
   #print(Ivec)
   #print(np.shape(Ivec))
   b=np.average(cv_img)
   Avec=b.flatten() #conversion of avg in vector form
   #print(Avec)
   #print(np.shape(Avec))
   Vec=Ivec-Avec
   #print(Vec)
   #print(np.shape(Vec))
   d=np.array([Vec])
   #print(d)
   #print(np.shape(d))
   d_t=d.transpose()
   #print(d_t)
   #print(np.shape(d_t))
   C=d*d_t+C
   #print(C)
   print(np.shape(C))

eigval, eigvec = np.linalg.eig(C)  
#print(eigval)
#print(eigvec)
print(np.shape(eigval))
print(np.shape(eigvec))
   








