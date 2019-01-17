import numpy as np
import cv2
import glob

for img in glob.glob("/Users/Shruti/Documents/datasets/att_faces/s1/*.pgm"):
   cv_img = cv2.imread(img,0)
   print(cv_img)
   a=np.shape(cv_img) #dimension checking
   print(a)
   Ivec=np.array(a).flatten() #conversion  of image in vector form
   print(Ivec)
   b=np.average(cv_img)
   Avec=np.array(b).flatten() #conversion of avg in vector form
   print(Avec)
   Vec=Ivec-Avec
   print(Vec)

   n = 400
   matrix = np.zeros((n,2)) # Pre-allocate matrix
   for i in range(1,n):
     matrix[i,:]=Vec #FILLING MATRIX'S column with vec
     print(matrix)

A=np.cov(matrix)  #covariance matrix
print(A)
L=A.transpose()
print(L)
eigval, eigvec = np.linalg.eig(L)
print(eigval)
V=eigvec
print(eigvec)
U=A*V
print(U)
U_T=U.transpose()

