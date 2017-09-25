# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 17:05:12 2016

@author: cs390mb

My Activities Main Python Module

This is the main entry point for the Python analytics.

You should modify the user_id field passed into the Client instance, 
so that your connection to the server can be authenticated. Also, 
uncomment the lines relevant to the current assignment. The 
map_data_to_function() function simply maps a sensor type, one of 
"SENSOR_ACCEL", "SENSOR_AUDIO" or "SENSOR_CLUSTERING_REQUEST", 
to the appropriate function for analytics.

"""

from client import Client

# TODO: uncomment each line as needed:
from A1 import compute_average_acceleration
from A1 import step_detection

# TODO: fill this empty string with your batch id
# e.g., user_id = "waz92bs17acqn3cx"
user_id = ""
c = Client(user_id)

# TODO: comment this line out when starting part 3
c.map_data_to_function("SENSOR_ACCEL", compute_average_acceleration.compute_average)

# TODO: uncomment this line for part 3
# c.map_data_to_function("SENSOR_ACCEL", step_detection.detect_steps)

# connect to the server to begin:
c.connect()