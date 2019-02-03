import numpy as np
import cv2
import glob
from numpy import linalg as LA
import math
n = 300 #total number of images in dataset
d=307200 #480*640 which is dimension of an image
k=100 # taking k, i.e. 100 eigen vectors corresponding to 100 largest eigen values
Avec= np.zeros((d,))
for img in glob.glob("/Users/Shruti/Documents/datasets/att_faces/d/*.jpg"): #to read all the images in dataset
   cv_img = cv2.imread(img,0)
   #print(cv_img)  # converted image in matrix form
   a=np.shape(cv_img) #check the dimension of matrix of image
   #print(a) #480*640
   Ivec=cv_img.flatten() #conversion  of above image matrices in vector form
   #print(Ivec) 
   #print(np.shape(Ivec))  #307200*1
   Avec+=Ivec #average (mean) of images in the database #(307200,)
   #print(Avec)

Avec/=n 
i=0
matrix = np.zeros((n,d)) # Pre-allocate matrix of dimension n*d
for img in glob.glob("/Users/Shruti/Documents/datasets/att_faces/d/*.jpg"):
  cv_img = cv2.imread(img,0)
  #print(cv_img)  # converted image in matrix form
  a=np.shape(cv_img) #check the dimension of matrix of image
  #print(a) #480*640
  Ivec=cv_img.flatten()
  Vec=Ivec-Avec #subtraction of average faces from the training faces
  #print(Vec) #d*1
  matrix[i,:]=Vec #FILLING MATRIX'S column with vec
  i=i+1
  #print(matrix) #dimension= n*d
     
# we have to calculate a similar matrix L as covariance matrix to reduce the computational complexity so we consider the covariance of 'matrix'(n*d) instead of 'X'(d*n)
# where 'X' is transpose of 'matrix' (basically X is actual image space whose covariance we have to find)

X=matrix.transpose()  #d*n

L=np.cov(matrix)  #similar as covariance matrix to reduce coputational complexity
#print(L)  #dimension= n*n

eigval, eigvec = np.linalg.eig(L) #calculated eigen values and eigen vector corresponding to matrix L
#print(eigval)
T=eigval # copied eigenvalues to T  #n*1
W=eigvec  # W eigenvectors of matrix L having dimension n*n
#print(W)

V=np.dot(X, W) #d*n #eigenvectors of covariance matrix by calculating linear combination of image space with eigenvectors of L(similar matrix)
#print(V)

sorted_T=np.sort(T)
reverse=sorted_T[::-1] #sorted eigenvalues in descending order
#print(reverse)

index_vector=np.zeros((k,)) #pre allocate a matrix with dimension k*1 for some largest eigenvalues
for i in range(1,k):
  i_eigval=reverse[i]     #selected k largest eigen values from sorted eigenvalues and stored it in i_eigval
  for j in range(1,n):
    if i_eigval==T[j]: #found the index of largest eigenvalues stored in T (all eigenvalues)
      index_vector[i]=j
#print(index_vector) #stored the indexes of those largest eigenvalues in matrix index_vector

projected_V=np.zeros((d,k)) #initialised a matrix with dimension d*k to store the projection of images
for i in range(1,k):
  index=index_vector[i] 
  eig_vecs=V[:,int(index)] #find the eigenvector corresponding the index value from V(where eigenvectors are stored)
  projected_V[:,i]=eig_vecs # stored those eigenvectors in projection of matrix
#print(projected_V)  #d*k


projected_t=projected_V.transpose()  #k*d #transpose of the projected matrix

eigcoeff_matrix=np.zeros((k,n)) #initialize eigencoefficient matrix with dimension(k*n)
for i in range(1,n):
  eigcoeff_matrix[:,i]=np.dot(projected_t,X[:,i]) #find the eigencoefficients of linear combination and stored it in eigencoefficient matrix
  #print(eigcoeff_matrix[:,i])
#print(eigcoeff_matrix) #k*n

#Testing phase 

imgpath="/Users/Shruti/Documents/datasets/testing_dataset/test5.jpg" # reading the image
test_image = cv2.imread(imgpath,0)
#print(test_image) #dimension of testing image= 480*640

TestImg_vec=test_image.flatten() #dimension= d*1 i.e. (307200,)

mean_image= TestImg_vec-Avec #deducted mean image from test image with dimension d*1 =(307200,)
eigcoeff_test= np.dot(projected_t,mean_image) #k*1

minimum=math.inf #initialise a function minimum wih infinity to store minimum of frobenius 
matched=0 #to store the index of minimum of frobenius
for i in range(1,n):
  J= eigcoeff_test - eigcoeff_matrix[:,i] #difference between eigencoeff of testing image and eigencoeff matrix of training dataset 
  Frobenius=LA.norm(J) #frobenius of difference calculated above
  if minimum>Frobenius:
    minimum=Frobenius  #stored the minimum of frobenius 
    matched=i       
print(matched)  #index of eigencoeff (closest match in terms of the squared distance(==squared root of distance) between the eigen-coefficients)


  









