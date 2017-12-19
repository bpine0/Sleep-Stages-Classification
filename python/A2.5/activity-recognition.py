# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:34:11 2016

Assignment A0 : Data Collection

@author: cs390mb

This Python script receives incoming unlabelled accelerometer data through 
the server and uses your trained classifier to predict its class label. 
The label is then sent back to the Android application via the server.

"""


import sys

import numpy as np
import pickle
from features import extract_features # make sure features.py is in the same directory

import os
from datetime import datetime
from util import slidingWindow, reorient, reset_vars
import matplotlib.pyplot as plt


with open('classifier.pickle', 'rb') as f:
    classifier = pickle.load(f)

if classifier == None:
    print("Classifier is null; make sure you have trained it!")
    sys.exit()


def load(fileName):
    with open(fileName) as f:
        return pickle.load(f)

def predict():
    """
    Given a window of accelerometer data, predict the activity label. 
    Then use the onActivityDetected(activity) function to notify the 
    Android must use the same feature extraction that you used to 
    train the model.
    """
    # have to fix the window size
    prediction_array=np.array([])
    p_time=np.array([])
    clf = load("classifier.pickle")
    # maybe we are not even filling buffer but just running a for loop

    data_file_ss_11 = os.path.join('data', 'accel_data-12-08-BP-ss.csv')
    data_ss_11 = np.loadtxt(data_file_ss_11, delimiter=',', dtype = object, converters = {0: np.float, 1: np.float, 2: np.float, 3: lambda t: datetime.strptime(t.decode("utf-8"), "%d/%m/%Y %H:%M")})
    data_ss_11 = np.insert(data_ss_11, 3, 0, axis = 1)
    hdata_file_ss_11 = os.path.join('data', 'BPM_2017-12-08-BP-ss.csv')
    hdata_ss_11 = np.loadtxt(hdata_file_ss_11, delimiter=',', dtype = object, converters = {0: lambda t: datetime.strptime(t.decode("utf-8"), "%d/%m/%Y %H:%M"), 1: np.float})


    data = data_ss_11
    hdata = hdata_ss_11


    window_size=20
    step_size=20
    #because hr data in backwards
    count = len(hdata)-1
    for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
        temp = np.zeros((1,3))
        #while time at row count is under time at accel, increase count (move to next row)
        #only have one window. Each row in window has own observation that needs hr
        for row in range(len(window_with_timestamp_and_label)):

            # print (hdata[count])
            # print(" ")
            # print (window_with_timestamp_and_label[row])


            while hdata[count][0] < window_with_timestamp_and_label[row][4] and count > 0:
                count=count-1
                print("changed count ", count)

            if row==0:
                p_time=np.append(p_time,window_with_timestamp_and_label[row][4])
            #remove timestamps from accel data
            temp = np.vstack((temp,window_with_timestamp_and_label[row][:-2]))
            #add hr data to accel
            hr_label = np.append(hdata[count][1],9)
            window_with_timestamp_and_label[row] = np.append(temp[row+1], hr_label)
            #add in label (hr_data is on form hr, t, label)
            #remove time and label for feature extraction
        window = window_with_timestamp_and_label[:,:-1]
        # extract features over window:
        # print("Buffer filled. Run your classifier.")

        prediction=clf.predict(np.reshape(extract_features(window),(1,-1)))[0]
        prediction_array=   np.append(prediction_array,prediction)
        # print prediction

    # for i in range(0,len(prediction_array)):
    #     p_time=np.append(p_time,i)

    plt.plot(p_time,prediction_array)
    plt.xlabel('Time')
    plt.ylabel('Predicted Label')
    plt.show()
    return


if __name__=='__main__':
    predict()