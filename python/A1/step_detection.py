# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:34:11 2016

@author: cs390mb

Step Detection

This Python script receives incoming accelerometer data through the 
server, detects step events and sends them back to the server for 
visualization/notifications.

"""

import numpy as np
from filters import ButterworthFilter, ExponentialFilter

from save_display_data import CSVSaver

# TODO (optional): you can create additional CSVSaver objects here with different filenames to save multiple files
# (for example, if you want to compare different types of filtering on the same data)
csv_saver = CSVSaver("accel_data.csv")

def detect_steps(data, on_step_detected, *args):
    """
    Accelerometer-based step detection algorithm.
    
    Implement your step detection algorithm. This may be functionally 
    equivalent to your Java step detection algorithm if you like. 
    Remember to use the global keyword if you would like to access global 
    variables such as counters or buffers. When a step has been detected, 
    call the onStepDetected method, passing in the timestamp:
    
        onStepDetected("STEP_DETECTED", timestamp)       
    
    """
    
    timestamp = data['t']
    x = data['x']
    y = data['y']
    z = data['z']
    



    # TODO: Implement step detection algorithm




    # This saves the data to accel_data.csv
    # TODO: You can plot this later using save_display_data.plot_csv_data("accel_data.csv")
    # TODO: You can (and should) change this to save the data after it's been filtered once you are doing filtering
    # (which will help you to determine window size and what filtering parameters to use)
    csv_saver.save_data_item([x,y,z,timestamp])
    
    # TODO: call on_step_detected only when you detect a step:
    on_step_detected("STEP_DETECTED", {"timestamp" : timestamp})  
    
    return