#imports
#importing the digits dataset
from sklearn import datasets
digits = datasets.load_digits()
#importing the knn function
from sklearn.neighbors import KNeighborsClassifier
k = int(input('Please enter a value for k: '))
if k<3 or k>20:
    print('k will now be set to 10.')
    k=10
else:
    print('k will now be set to ', k, '.')
knn = KNeighborsClassifier(n_neighbors=k)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import time
start_time = time.time()

# defining our x and y, x being the data our knn sees and y being the target which it needs to indentify
y = digits.target
# flatten the image to 1D, making our 8x8 image just an array of 64 pixels
x = digits.images.reshape((len(digits.images), -1))
x.shape


# split our data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1, stratify=y)
#implementing the knn library function
model = knn.fit(x_train, y_train)
#using knn.predict function
predictions = knn.predict(x_test)

#output
#print number of classes
print ('Number of classes: %d' %len(np.unique(y)))
#print number of data points
print ('Number of data points: %d' %len(y))
#print number of samples from each class
for i in range(len(np.unique(y))):
    a = len(x[y == i,:])
    print ('\nSamples from class',i, ':',a)
#print the classification report including min and max features
print('Classification report.')
print(classification_report(y_test, predictions))
#printing train test split
print('Train, Test split.')
print('Training data images (x): ', x_train.shape,', Testing data images (x): ', x_test.shape,)
print('Training data labels (y): ', y_train.shape,', Testing data labels (y): ', y_test.shape)
#printing accuracy of function based on training and testing datasets
print('Training accuracy score')
print(knn.score(x_train, y_train))
print('Testing accuracy score')
print(knn.score(x_test, y_test))
print('My program took', time.time() - start_time, 'seconds to run')
