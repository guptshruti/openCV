import numpy as np
import cv2
import glob
from sklearn import preprocessing
n = 300
d=307200
k=100

for img in glob.glob("/Users/Shruti/Documents/datasets/att_faces/d/*.jpg"):
   cv_img = cv2.imread(img,0)
   #print(cv_img) 
   a=np.shape(cv_img) #dimension checking
   #print(a) #480*640
   Ivec=cv_img.flatten() #conversion  of image in vector form
   #print(Ivec)   #307200*1
   b=np.average(cv_img)
   Avec=b.flatten() #conversion of avg in vector form
   #print(Avec)
   Vec=Ivec-Avec
   #print(Vec)
   #print(np.shape(Vec)) #d*1
   matrix = np.zeros((n,d)) # Pre-allocate matrix
   for i in range(1,n):
     matrix[i,:]=Vec #FILLING MATRIX'S column with vec
     #print(matrix)
     #print(np.shape(matrix)) #n*d

L=np.cov(matrix)  #covariance matrix
#print(L) 
#print(np.shape(L))  #n*n
X=matrix.transpose() 
#print(np.shape(X))   #d*n

eigval, eigvec = np.linalg.eig(L)
#print(eigval)
T=eigval
#print(np.shape(T)) #n*1
W=eigvec  #N*N
#print(W)
V=np.dot(X, W) #d*N
#print(V)
#print(np.shape(V))
# V_normalized = preprocessing.normalize(V, norm='l2')
#print(V_normalized)

sorted_T=np.sort(T)
reverse=sorted_T[::-1]
#print(reverse)

index_vector=np.zeros((k,))
for i in range(1,k):
  i_eigval=reverse[i]
  for j in range(1,n):
    if i_eigval==T[j]:
      index_vector[i]=j

#print(index_vector)
projected_V=np.zeros((d,k))
for i in range(1,k):
  index=index_vector[i]
  eig_vecs=V[:,int(index)]
  projected_V[:,i]=eig_vecs

#print(projected_V)
#print(np.shape(projected_V)) #d*k
projected_t=projected_V.transpose()  #k*d 

eigcoeff_matrix=np.zeros((k,n))
for i in range(1,n):
  eigcoeff_matrix[:,i]=np.dot(projected_t,X[:,i])
print(eigcoeff_matrix)
print(np.shape(eigcoeff_matrix))  #k*n









