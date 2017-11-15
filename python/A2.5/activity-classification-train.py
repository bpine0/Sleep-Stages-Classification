# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:02:58 2016

@author: cs390mb

Assignment 2 : Activity Recognition

This is the starter script used to train an activity recognition 
classifier on accelerometer data.

See the assignment details for instructions. Basically you will train 
a decision tree classifier and vary its parameters and evalute its 
performance by computing the average accuracy, precision and recall 
metrics over 10-fold cross-validation. You will then train another 
classifier for comparison.

Once you get to part 4 of the assignment, where you will collect your 
own data, change the filename to reference the file containing the 
data you collected. Then retrain the classifier and choose the best 
classifier to save to disk. This will be used in your final system.

Make sure to chek the assignment details, since the instructions here are 
not complete.

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import svm
from features import extract_features # make sure features.py is in the same directory
from util import slidingWindow, reorient, reset_vars
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle


# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = os.path.join('data', 'activity-data.csv')
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i,1], data[i,2], data[i,3]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:1],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)


# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

# you may want to play around with the window and step sizes
window_size = 20
step_size = 20

# sampling rate for the sample data should be about 25 Hz; take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,0] - data[0,0]) / 1000
sampling_rate = n_samples / time_elapsed_seconds

feature_names = ["Mean", "Variance", "Local Mimimum Count", "Local Maximum Count", "Range", "Magnitude of Dominant Frequency","Distance Travelled"]
class_names = ["Sitting","Walking","Standing","Stairs"]

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

n_features = len(feature_names)

X = np.zeros((0,n_features))
y = np.zeros(0,)

for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    # omit timestamp and label from accelerometer window for feature extraction:
    window = window_with_timestamp_and_label[:,1:-1]  
    # extract features over window:
    x = extract_features(window)
    # append features:
    X = np.append(X, np.reshape(x, (1,-1)), axis=0)
    # append label:
    y = np.append(y, window_with_timestamp_and_label[10, -1])
    
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(y)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Plot data points
#
# -----------------------------------------------------------------------------

# We provided you with an example of plotting two features.
# We plotted the mean X acceleration against the mean Y acceleration.
# It should be clear from the plot that these two features are alone very uninformative.
"""print("Plotting data points...")
sys.stdout.flush()
plt.figure()
formats = ['bo', 'go']
for i in range(0,len(y),10): # only plot 1/10th of the points, it's a lot of data!
    plt.plot(X[i,2], X[i,3], formats[int(y[i])])
    
plt.show()"""

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------

n = len(y)
n_classes = len(class_names)

# TODO: Train and evaluate your decision tree classifier over 10-fold CV.
# Report average accuracy, precision and recall metrics.

clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, max_features = 5 )
# clfl=LogisticRegression(C=1)

cv = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=None)

def compute_accuracy(conf):
    r0c0 = conf[0][0]
    r1c1 = conf[1][1]
    r2c2 = conf[2][2]
    r3c3 = conf[3][3]
    accuracy = float(r0c0+r1c1+r2c2+r3c3)/np.sum(conf)
    print("accuracy: {}".format(accuracy))
    return accuracy

def compute_recall(conf, col):
    #actual = column, predicted = row
    #TP/(TP+FN), col-wise
    #col = 0,1,2
    row_tp = col
    if col == 0:
        row2 = col+1
        row3 = col+2
        row4 = col+3
    if col == 1:
        row2 = col-1
        row3 = col+1
        row4 = col+2
    if col == 2:
        row2 = col-2
        row3 = col-1
        row4 = col+1
    if col == 3:
        row2 = col-3
        row3 = col-2
        row4 = col-1


    TP = float(conf[row_tp][col])
    FN = float(conf[row2][col])+float(conf[row3][col])+float(conf[row4][col])
    recall = (TP)/(TP + FN) if (TP+FN !=0) else -5
    #print("recall ",conf[col_t][row], conf[col2][row], conf[col3][row])
    print("recall {}: {}").format(col, recall)
    return recall

def compute_precision(conf, row):
    #TP/(TP+FP), row-wise
    col_tp = row
    if row == 0:
        col2 = row+1
        col3 = row+2
        col4 = row+3
    if row == 1:
        col2 = row-1
        col3 = row+1
        col4 = row+2
    if row == 2:
        col2 = row-2
        col3 = row-1
        col4 = row+1
    if row == 3:
        col2 = row-3
        col3 = row-2
        col4 = row-1


    TP = float(conf[row][col_tp])
    FP = float(conf[row][col2])+float(conf[row][col3])+float(conf[row][col4])
    # print(conf[var][0], conf[var][1], conf[var][2])
    precision = (TP)/(TP + FP) if (TP+FP !=0) else -5
    # print("precision: ", precision)
    print("precision {}: {}").format(row, precision)
    return precision
    #not correct because always left column = target

fold = np.zeros([9,10])
#rows:
#acc
#pre 1
#pre 2
#pre 3
#pre 4
#rec 1
#rec 2
#rec 3
#rec 4

for i, (train_indexes, test_indexes) in enumerate(cv):
    X_train = X[train_indexes, :]
    y_train = y[train_indexes]
    X_test = X[test_indexes, :]
    y_test = y[test_indexes]
    clf.fit(X_train, y_train)
    # clfl.fit(X_train, y_train)

    # predict the labels on the test data
    y_pred = clf.predict(X_test)
    # y_pred = clfl.predict(X_test)

    # show the comparison between the predicted and ground-truth labels
    conf = confusion_matrix(y_test, y_pred, labels=[0,1,2,3])
    
    print("Fold {} : The confusion matrix is :".format(i))
    print conf
    acc = compute_accuracy(conf)
    fold[0,i] = acc
    for j in range(4):
        pre = compute_precision(conf, j)
        rec = compute_recall(conf,j)
        fold[1+j, i] = pre
        fold[5+j, i] = rec

    
    print("\n")

avg_conf = np.mean(fold, axis = 1)
print(fold)
print()
print(avg_conf)
    
# TODO: Evaluate another c = lassifier, i.e. SVM, Logistic Regression, k-NN, etc.
    
# TODO: Once you have collected data, train your best model on the entire 
# dataset. Then save it to disk as follows:

# when ready, set this to the best model you found, trained on all the data:
best_classifier = clf
with open('classifier.pickle', 'wb') as f: # 'wb' stands for 'write bytes'
    pickle.dump(best_classifier, f)