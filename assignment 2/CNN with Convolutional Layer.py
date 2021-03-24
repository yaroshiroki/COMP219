#base imports
import numpy as np
import matplotlib.pyplot as plt
import time
#dataset import
from sklearn import datasets
#keras imports
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils

start_time = time.time()
#Loading and defining the dataset
digits = datasets.load_digits()
digits_X, digits_y = digits.data, digits.target
#Reshaping the data to apply our convolutional layer on
digits.data[0].reshape((-1,8,8,1))
#Creating a variable which is simply the length of the dataset
count = len(digits.images)

#Splitting and reshaping our dataset so that we can apply out convolutional layer
X_train = digits.data[: int(0.7*count)].reshape((-1,8, 8,1))
y_train = np_utils.to_categorical(digits.target[: int(0.7*count)])
X_test = digits.data[int(0.7*count): int(0.9*count)].reshape((-1,8, 8,1))
y_test = np_utils.to_categorical(digits.target[int(0.7*count): int(0.9*count)])

#Creating a function for our convolutional neural network
def CNN():
    #Using the sequential model
    model = Sequential()
    #Applying a number of convolutional functions to our layer such ash Conv2D and MaxPooling2D
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1)))
    model.add(Conv2D(64,(3,3), activation = 'relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    #transforming our matrix into a vector
    model.add(Flatten())
    #using the dot product considering weight, bias and activation layer
    model.add(Dense(128, activation='relu'))
    #randomly selecting neurons to dropout (simple overvfitting technique)
    model.add(Dropout(0.5))
    #again using dense but here considering the output of 10 possible classes and using
    # softmax to classify our data
    model.add(Dense(10, activation='softmax'))

    # Compile the classification model
    model.compile(loss='categorical_crossentropy', optimizer = 'adam',
                   metrics=['accuracy'])
    return model

#Keeping to the name model in the gobal sphere
model = CNN()
#Fit/train the classification model
history = model.fit(X_train, y_train, validation_data = (X_test, y_test),  batch_size = 64, epochs = 50, verbose = 2)
#Print a summary of the model, displaying the layers, output shape, number of parameters
# and model type
model.summary()

print(count)
#Saving the model
model.save('CNN_digit_ConvLyr.hdf5')

#Creating labels for the true values of y and what classes the model predicts the y
# values belong to
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis = 1)
y_true = np.argmax(y_test, axis = 1)

#Function for working out the accuracy given a dataset and the predictions
#This function had to be made as the sklearn accuracy_score function did not work for
# my crossVal function
def accuracy(dataset, predictions):
    right = 0
    for i in range (len(dataset)):
        if dataset[i] == predictions[i]:
            right = right + 1
    return (right/float(len(dataset)))

#Function for doing the kfolds cross validation given y_pred_classes and y_true
def crossVal(pred, true):
    folds = 5
    X_folds = []
    y_folds = []
    X_folds = np.array_split(digits_X,folds)
    y_folds = np.array_split(digits_y,folds)
    predicts = []
    y_val = []
    print('K-FOLDS CROSS VALIDATION - ERROR RATE')

    for i in range(folds):#split X_train and X_test datasets
        X_train =np.vstack(X_folds[:i] + X_folds[i+1:]) 
        X_test = X_folds[i]
        y_train = np.hstack(y_folds[:i] + y_folds[i+1:])
        y_test = y_folds[i]
        for j in range(len(y_pred_classes)):
            temp_predicts = np.argmax(y_pred_classes[j])
            predicts.append(temp_predicts)
        y_val.extend(y_test)
        #need to get the accuracy of predicts and y_val then print this
    prediction = accuracy(y_val, predicts)
    print(prediction)

cross = crossVal(y_pred_classes,y_true)

#Creating a function for displaying the confusion matrix given y_true and y_pred_classes
def confusion_matrix(true, pred):
    k = len(np.unique(true))
    result = np.zeros((k,k))

    true_positive=0;
    false_positive=0;
    true_negative=0;
    false_negative=0;
    for i in range(len(true)):
        result[true[i]][pred[i]] += 1
        
    return result

#printing the confusion matrix
confusion_matrix_printing = confusion_matrix(y_true, y_pred_classes)

print('CONFUSION MATRIX')
print(confusion_matrix_printing)

#working out the number of TP, FP, TN, FN given the true value of y and the predicted
# classes
def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_actual[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
           FP += 1
        if y_actual[i]==y_pred[i]==0:
           TN += 1
        if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
           FN += 1

    return(TP, FP, TN, FN)

#Creating a function to work out the true positive rate and false positive rate
def TPR_FPR(TP, FP, TN, FN):
    FPR = FP / (FP + TN)
    TPR = TP / (TP + FN)
    return (FPR, TPR)

#Displaying the number of TP, FP, TN, FN as well as the FPR nad TPR so that we can
# work on the ROC curve
TP, FP, TN, FN = perf_measure(y_true,y_pred_classes)
print('Number of True Positives, False Positives, True Negatives and False Negatives - respectively')
print(TP, FP, TN, FN)
FPR, TPR = TPR_FPR(TP, FP, TN, FN)
print('The False Positive Rate and True Positive Rate - respectively')
print(FPR, TPR)

#Function for plotting the ROC curve -- however, this function is incomplete as it does
# not plot the ROC curve, due to the lack of ability to work out the threshold/AUC
def results(x, y):
    # x = false_positive_rate
    # y = true_positive_rate

    #As this line of code doesn't work, my ROC curve cannot propperly function
    #AUC = np.trapz(y, dx = x)

    # This is the ROC curve
    plt.title('Receiver Operating Characteristic')
    plt.plot(x, y, color='r', linewidth=2.0)
    #plt.plot(x, y, 'b', label = 'AUC = %0.2f' % AUC)
    #plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('FALSE POSITIVE RATE')
    plt.xlabel('TRUE POSITIVE RATE')
    plt.show()

results(FPR, TPR)
print('My program took', time.time() - start_time, 'seconds to run')
