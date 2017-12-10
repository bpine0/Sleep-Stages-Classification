# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:08:49 2016

@author: cs390mb

This file is used for extracting features over windows of tri-axial accelerometer 
and heart tate data.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np
from scipy import integrate

def compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    all_means = np.mean(window, axis=0)
    mag = (all_means[0]**2 + all_means[1]**2 + all_means[2]**2)**0.5
    # return np.mean(window, axis=0)
    #return np.append(all_means, mag)
    return mag

def variance(window):
    all_var = np.var(window, axis = 0)
    mag = (all_var[0]**2+all_var[1]**2+all_var[2]**2)**0.5
    #return(np.append(all_var,mag))
    return mag

def sum_local_max(window):
    
    #extracts just x/y/z values from window
    xs = np.array([point[0] for point in window])
    ys = np.array([point[1] for point in window])
    zs = np.array([point[2] for point in window])

    #creates bool array comparing values to preceding and following values (excluding endpoints)
    x = (xs[1:-1] > xs[:-2]) & (xs[1:-1] > xs[2:])
    y = (ys[1:-1] > ys[:-2]) & (ys[1:-1] > ys[2:])
    z = (zs[1:-1] > zs[:-2]) & (zs[1:-1] > zs[2:])

    #cut off first and last element
    xs = xs[1:-1]
    ys = ys[1:-1]
    zs = zs[1:-1]

    xsum = 0
    ysum = 0
    zsum = 0

    #add all local maximums
    for i in range(0,len(xs)):
        if x[i] == True:
            xsum += xs[i]

    for i in range(0,len(ys)):
        if y[i] == True:
            ysum += ys[i]

    for i in range(0,len(zs)):
        if z[i] == True:
            zsum += zs[i]

    #return np.array([xsum, ysum, zsum])
    mag = (xsum**2+ysum**2+zsum**2)**0.5
    return mag

def sum_local_min(window):

    #extracts just x/y/z values from window
    xs = np.array([point[0] for point in window])
    ys = np.array([point[1] for point in window])
    zs = np.array([point[2] for point in window])

    #creates bool array comparing values to preceding and following values
    x = (xs[1:-1] < xs[:-2]) & (xs[1:-1] < xs[2:])
    y = (ys[1:-1] < ys[:-2]) & (ys[1:-1] < ys[2:])
    z = (zs[1:-1] < zs[:-2]) & (zs[1:-1] < zs[2:])

    xs = xs[1:-1]
    ys = ys[1:-1]
    zs = zs[1:-1]

    xsum = 0
    ysum = 0
    zsum = 0

    for i in range(0,len(xs)):
        if x[i] == True:
            xsum += xs[i]

    for i in range(0,len(ys)):
        if y[i] == True:
            ysum += ys[i]

    for i in range(0,len(zs)):
        if z[i] == True:
            zsum += zs[i]

    #return np.array([xsum, ysum, zsum])
    mag = (xsum**2+ysum**2+zsum**2)**0.5
    return mag

def range_estimate(window):
    #does not use max and min in case there are high/low points from random noise
    upper = np.percentile(window, 95, axis = 0)
    lower = np.percentile(window, 5, axis = 0)
    range_est = upper-lower
    mag = (range_est[0]**2+range_est[1]**2+range_est[2]**2)**0.5
    #return np.append(range_est, mag)
    return mag

def fft_max_ct(window):
    #transform each set of data points
    freq=np.fft.rfft(window,axis = 0) #why 1 and not 0?
    freq=freq.astype(float)
    #does walking have the same most common frequency as stationary? (get maximum per axis)
    freq_all = np.amax(freq, axis = 0)
    mag = (freq_all[0]**2+freq_all[1]**2+freq_all[2]**2)**0.5
    #return np.append(freq_all, mag)
    return mag

#using mean only because will just be 1 or 2 values over the whole window
def mean_hr(window):
    return np.mean(window, axis=0)[3]


def extract_features(window):
    """
        Make sure that X is an N x d matrix, where N is the number 
    of data points and d is the number of features.
    
    """
    
    x = np.array([compute_mean_features(window), variance(window), sum_local_min(window), sum_local_max(window), range_estimate(window), fft_max_ct(window), mean_hr(window)])
    return x