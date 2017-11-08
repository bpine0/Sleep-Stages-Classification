# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:00:51 2016

@author: snoran

Includes various utility functions.
"""

import numpy as np
 
def slidingWindow(sequence,winSize,step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable.
    Thanks to https://scipher.wordpress.com/2010/12/02/simple-sliding-window-iterator-in-python/"""
 
    # Verify the inputs
    try: 
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")
 
    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence)-winSize)/step)+1
 
    # Do the work
    for i in range(0,numOfChunks*step,step):
        yield i, sequence[i:i+winSize]


GRAVITY = 9.81
READ_LIMIT = 400;
acc_readings = np.zeros((READ_LIMIT, 3))

acc_state = False;
read_counter = 0;
aggX = 0
aggY = 0
aggZ = 0

def reset_vars():
    """
    Resets the variables used in reorientation. Since they are global 
    variables, we need to make sure that they are reset. In the future, 
    this should really be done using some sort of Python object.
    """
    
    global acc_state
    global read_counter
    global aggX
    global aggY
    global aggZ
    
    acc_state = False;
    read_counter = 0;
    aggX = 0
    aggY = 0
    aggZ = 0

def reorient(acc_x, acc_y, acc_z):
    """
    Reorients the accelerometer data. It comes from some legacy 
    Java code, so it's very messy. You don't need to worry about 
    how it works.
    """
    x = acc_x
    y = acc_z
    z = -acc_y
    
    global acc_state
    global read_counter
    global aggX
    global aggY
    global aggZ
    
    if read_counter >= READ_LIMIT:
        read_counter = 0
    
    accState = True;
    
    aggX += x - acc_readings[read_counter][0];
    aggY += y - acc_readings[read_counter][1];
    aggZ += z - acc_readings[read_counter][2];

    acc_readings[read_counter][0] = x;
    acc_readings[read_counter][1] = y;
    acc_readings[read_counter][2] = z;

    if(accState):
        acc_z_o = aggZ/(READ_LIMIT*GRAVITY);
        acc_y_o = aggY/(READ_LIMIT*GRAVITY);
        acc_x_o = aggX/(READ_LIMIT*GRAVITY);
        
        if acc_z_o > 1.0:
            acc_z_o = 1.0
        if acc_z_o < -1.0:
            acc_z_o = -1.0
        x = x/GRAVITY;
        y = y/GRAVITY;
        z = z/GRAVITY;
        
        theta_tilt = np.arccos(acc_z_o);
        phi_pre = np.arctan2(acc_y_o, acc_x_o);
        tan_psi = (-acc_x_o*np.sin(phi_pre) + acc_y_o*np.cos(phi_pre))/((acc_x_o*np.cos(phi_pre)+acc_y_o*np.sin(phi_pre))*np.cos(theta_tilt)-acc_z_o*np.sin(theta_tilt));
        psi_post = np.arctan(tan_psi);
        acc_x_pre = x*np.cos(phi_pre)+ y*np.sin(phi_pre);
        acc_y_pre = -x*np.sin(phi_pre)+ y*np.cos(phi_pre);
        acc_x_pre_tilt = acc_x_pre*np.cos(theta_tilt)-z*np.sin(theta_tilt);
        acc_y_pre_tilt = acc_y_pre;
        orient_acc_x = (acc_x_pre_tilt*np.cos(psi_post)+acc_y_pre_tilt*np.sin(psi_post))*GRAVITY;
        orient_acc_y =(-acc_x_pre_tilt*np.sin(psi_post)+acc_y_pre_tilt*np.cos(psi_post))*GRAVITY;
        orient_acc_z = z*GRAVITY/(np.cos(theta_tilt));
        
        if orient_acc_z > 3 * GRAVITY:
            orient_acc_z = 3 * GRAVITY;
        if orient_acc_z < -3 * GRAVITY:
            orient_acc_z = -3 * GRAVITY;
            
        orient_acc_z = np.sqrt((x*x+y*y+z*z)*GRAVITY*GRAVITY - (orient_acc_x*orient_acc_x + orient_acc_y*orient_acc_y));
        
        result = [orient_acc_x, orient_acc_y, orient_acc_z]
    read_counter += 1;
    return result;