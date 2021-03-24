#imports
from sklearn import datasets
#loading out dataset
digits = datasets.load_digits()
import numpy as np
import matplotlib
import time
from sklearn.metrics import accuracy_score
start_time = time.time()

#creating a function to calculate the euclidean distance
def Euclidean(row1, row2):
    #making a function for length of row1 in order to improve efficiency
    lengthOfRow1 = len(row1)-1
    distance = 0
    #using the euclidean distance rather than the minkowski distance as it is more accurate for the size of our data sample
    for i in range(lengthOfRow1):
        distance += abs(row1[i] - row2[i])**2
    return np.sqrt(distance)

#creating a function to find the knn after sorting distance
def getNeighbors(train, testRow, k):
    distance = list()
    data = []
    for i in train:
        dist = Euclidean(testRow, i)
        distance.append(dist)
        data.append(i)
    distance = np.array(distance)
    data = np.array(data)
    #finding the index of the min distance in ascending order
    indexDist = distance.argsort()
    #arranging data according to index
    data = data[indexDist]
    #splitting the data according to the k number of neighbours
    neighbors = data[:k]
    return neighbors

#creating a function to predict the class of the new data point
def predictClass(train, testRow, k):
    Neighbors = getNeighbors(train, testRow, k)
    Classes = []
    for i in Neighbors:
        Classes.append(i[-1])
        #placing the data in the class with majority votes
        prediction = max(Classes, key= Classes.count)
    return prediction

#checking the column names and preprocessing target values in standard format (int8)
digits.keys()
digits.target = digits.target.astype(np.int8)
#finding the independent/dependent variables and finding the shape
#using a 1 liner in order to improve efficiency
x, y = np.array(digits.data), np.array(digits.target)
x.shape, y.shape
#output ((1797, 64), (1797,))

#shuffling the values of x and y to avoid overfitting
randomise = np.random.permutation(x.shape[0])
x, y = x[randomise], y[randomise]

#Used for testing our data sample

##import matplotlib.pyplot as plt
###reshaping our 64 columns into a 8*8 pixel image
##rndDigit = x[12]
##rndDigitImg = rndDigit.reshape(8, 8)
##plt.imshow(rndDigitImg, cmap=matplotlib.cm.binary)
##plt.axis(“off”)
##plt.show()
#output digit 0

#spliting our dataset
#doing this without using the train_test_split function,
# in order to take a smaller sample,
# as well as reducing our imports,
# making out program more efficient.
#done as a 1 liner in order to improve efficiency
x_train, y_train, x_test, y_test = x[:1000], y[:1000], x[50:], y[50:]
#inserting y_train into x_train
train = np.insert(x_train, 64, y_train, axis = 1)
#inserting y_test into x_test
test = np.insert(x_train, 64, y_train, axis = 1)


##prediction = predictClass(train, train[800], 10)
##print(prediction)
#checking the last column in our train set against the true value
##print(train[800][-1])
#output 6.0
#output 6.0
#both the true and predicted values match

#determine k
k = int(input('Please enter a value for k between 3 and 20: '))
if k<3 or k>20:
    print('k will now be set to 10. Please wait...')
    k=10
else:
    print('k will now be set to ', k, '. Please wait...')

y_pred=[]
y_true=train[:,-1]
#making a funcion for length of the training sample in order to increase efficiency
lengthOfTrain = len(train)
for i in range(lengthOfTrain):
    #using 10 as k, as we have 10 possible nodes
    prediction = predictClass(train, train[i], k)
    y_pred.append(prediction)

y_test=test[:,-1]
lengthOfTest = len(train)
for i in range (lengthOfTest):
    prediction = predictClass(test, test[i], k)

#display the accuracy and time taken
print('Training set accuracy: ', accuracy_score(y_true, y_pred))
print('Testing accuracy: ', accuracy_score(y_test, y_pred))
print('My program took', time.time() - start_time, 'seconds to run')
