# Define the function: reading data 

import os, struct
import matplotlib as plt
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros as np
import pandas as pd
from pandas import Series,DataFrame
#datasets form this url (New York University LeCun Professor)
#Change the var on your environment
DATA_PATH = '/Users/anumeha/Downloads/MNIST'
TRAIN_IMG_NAME = 'train-images-idx3-ubyte'
TRAIN_LBL_NAME = 'train-labels-idx1-ubyte'
TEST_IMG_NAME = 't10k-images-idx3-ubyte'
TEST_LBL_NAME = 'train-labels-idx1-ubyte'
# Set the path to save data
"""
Adapted from: http://cvxopt.org/applications/svm/index.html?highlight=mnist
"""
def load_mnist(dataset="training", digits=range(10), path=DATA_PATH):
#Set the filename
    if dataset == "training":
        fname_img = os.path.join(path, TRAIN_IMG_NAME)
        fname_lbl = os.path.join(path, TRAIN_LBL_NAME)
    elif dataset == "testing":
        fname_img = os.path.join(path, TEST_IMG_NAME)
        fname_lbl = os.path.join(path, TEST_LBL_NAME)
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels
# Read Digit data from training dataset
#Read the training and test dataset as numpy array
from pylab import *
from numpy import *
import scipy.sparse as sparse
import scipy.linalg as linalg

# Set the digit to read
images, labels = load_mnist('training', digits=[0,1,2,3,4,5,6,7,8,9])

# converting from NX28X28 array into NX784 array
flatimages = list()
for i in images:
    flatimages.append(i.ravel())
Xtr= np.asarray(flatimages)

# Create the Label like 0, 1, 2, 3...9
flatlabels = list()
for ii in labels:
    flatlabels.append(ii.ravel())
Ttr= np.asarray(flatlabels)
#import seaborn as sns
#fig = plt.figure(figsize=(12,12))
#sns.heatmap(pd.DataFrame(Xtr), annot=False, cmap='PuOr')
#plt.show()

#check the data by using imshow
#Print the digit
print("The shape of matrix is : ", Xtr.shape)
print("Label is : ", Ttr.shape)
plt.imshow(Xtr[7].reshape(28, 28),interpolation='None', cmap=cm.gray)
show()

#Read Digit data from test dataset
images, labels = load_mnist('testing', digits=[0,1,2,3,4,5,6,7,8,9])
# converting from NX28X28 array into NX784 array
flatimages = list()
for i in images:
    flatimages.append(i.ravel())
Xte= np.asarray(flatimages)
# Create the Label like 0, 1, 2, 3...9
flatlabels = list()
for ii in labels:
    flatlabels.append(ii.ravel())
Tte= np.asarray(flatlabels)
#Print the digit
print("The shape of matrix is : ", Xte.shape)
print("Label is : ", Tte.shape)
plt.imshow(Xte[7].reshape(28, 28),interpolation='None', cmap=cm.gray)
show()

#Calculate Principal Component from the dataset(training and test dataset separately)
# Data-preprocessing
#Mean normalization i.e. computing a vector containing the mean value of each feature among our training examples,
#and subtracting that from each example in our training set prior to processing
import numpy as np;

# Find average_face_vector, sum(all image vectors)/number(images) for both training and test data.
#average_face_vector
a= np.mean(Xtr,axis=0)
a1= np.mean(Xte,axis=0)

# Subtract average_face_vector from every image vector
#sub_face_vector Z
Z=Xtr-a
Z1=Xte-a1;
#Find the co-variance matrix which is : A^T * A
#Compute the covariance matrix of the flattened mean normalized input data (Z) for set of both training data nd test data

C=np.cov(Z,rowvar=False)
C1=np.cov(Z1,rowvar=False)
#import seaborn as sns
#fig = plt.figure(figsize=(12,12))
#sns.heatmap(pd.DataFrame(C), annot=False, cmap='PuOr')
#plt.show()
Z1.shape
#import seaborn as sns
#fig = plt.figure(figsize=(12,12))
#sns.heatmap(pd.DataFrame(C1), annot=False, cmap='PuOr')
#plt.show()
#Find the eigenvectors and eigenvalues of C. 
#create eigenvector from feature vector by using linear algebra technique
from numpy import linalg as La
[E_val, E_vec] = La.eigh(C);
E_val1,E_vec1= La.eigh(C1)
E_val.shape
Sorting Eigen values in descending order
together = zip(E_val,E_vec)
together = sorted(together, key=lambda t: t[0], reverse=True)
E_val[:], E_vec[:] = zip(*together)
together1 = zip(E_val1,E_vec1)
together1 = sorted(together1, key=lambda t: t[0], reverse=True)
E_val1[:], E_vec1[:] = zip(*together1)
#Project data onto the top eigenspaces having largest eigenvalues

PCs = E_vec[:70]
projections = np.dot(Z, PCs.T)
#print (projections.shape)
#projections = np.matmul(PCs, Z.T)
print (projections.shape)

#Project test data onto the two eigenvectors having largest eigenvalues
#First I chose 2 so that itsprojected to 2-d space for visualization
PCs1 = E_vec1[:70]
projectionsTest = np.dot(Z1, PCs1.T)
print (projectionsTest.shape)
#projectionsTest = np.matmul(PCs1, Z1.T)
#print (projectionsTest.shape)
#calculate the similarity of the input to each training image
#assess how good the model is by using back-projection, 
#to transform the low-dimensional inputs back into high-dimensional space, 
#with some loss of information
#Reconstruct train data matrix and test data matrix
P=np.dot(Z,E_vec.T)
R=np.dot(P,E_vec);
Xrec1=R+a;
Xrec1=(np.dot(P[:,0:2],E_vec[0:2,:]))+a; #Reconstruction using 2 components
Xrec2=(np.dot(P[:,0:20],E_vec[0:20,:]))+a; #Reconstruction using 20 components
Xrec3=(np.dot(P[:,0:40],E_vec[0:40,:]))+a; #Reconstruction using 70 components

plt.imshow(Xrec1[1].reshape(28, 28),interpolation='None', cmap=cm.gray);
show()
plt.imshow(Xrec2[1].reshape(28, 28),interpolation='None', cmap=cm.gray);
show()
plt.imshow(Xrec3[1].reshape(28, 28),interpolation='None', cmap=cm.gray);
show()
#calculate the similarity of the input to each test image
P1=np.dot(Z1,E_vec1)
R1=np.dot(P1,E_vec1);
Xrec4=R1+a1;
Xrec4=(np.dot(P[:,0:2],E_vec[0:2,:]))+a; #Reconstruction using 2 components
Xrec5=(np.dot(P[:,0:20],E_vec[0:20,:]))+a; #Reconstruction using 20 components
Xrec6=(np.dot(P[:,0:70],E_vec[0:70,:]))+a; #Reconstruction using 70 components

plt.imshow(Xrec4[1].reshape(28, 28),interpolation='None', cmap=cm.gray);
show()
plt.imshow(Xrec5[1].reshape(28, 28),interpolation='None', cmap=cm.gray);
show()
plt.imshow(Xrec6[1].reshape(28, 28),interpolation='None', cmap=cm.gray);
show()
#Preparing data for prediction over reduced dimentions

x_train = projections
x_test = projectionsTest
y_train = Ttr
y_test = Tte
print (x_train.shape)
print (x_test.shape)
print (y_train.shape)
print (y_test.shape)
# take 10% of the training data and use that for validation 
#from sklearn.model_selection import train_test_split
#(x_train, x_val, y_train, y_val) = train_test_split(x_train, y_train, test_size=0.1, random_state=84)
#step5. Build GaussianNB model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
#build image recognition model when we use 70 eigenvectors--> B=70, and taking all training data --> A=60,000
projections.shape
from sklearn import metrics
A = 59000
B = 70
model.fit(x_train[0:A,0:B],y_train[0:A])
predicted=model.predict(P[A:60001,0:B])
expected = y_train[A:60001,]
print(expected)
print ('The accuracy is : ', metrics.accuracy_score(expected, predicted)*100, '%')
cm = metrics.confusion_matrix(expected, predicted)
plt.figure(figsize=(9, 6))
sns.heatmap(cm, linewidths=.9,annot=True,fmt='g')
plt.suptitle('MNIST Confusion Matrix (GaussianNativeBayesian)')
#plt.title()
plt.show()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train.ravel())
y_pred = knn.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))
#Print confusion matrix
#check Classification Report and Confusion Matrix.
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
print ('          === Classification Report for KNN model ===')
print (metrics.classification_report(y_test, y_pred))
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(9, 6))
sns.heatmap(cm, linewidths=.9,annot=True,fmt='g')
plt.suptitle('MNIST Confusion Matrix (KNN)')
#plt.title()
plt.show()
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train.ravel())
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(x_train, y_train.ravel())
# Predict for One Observation (image)
pred_log = logisticRegr.predict(x_test)
print(metrics.accuracy_score(y_test, pred_log))
