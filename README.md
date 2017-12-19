

# Acquiring Labelled Data

One night of sleeping data was collected from five different people. Acceleration was collected using an android phone accelerometer, and heart rate data was collected using the Huawei Watch 2. CSVs of the data were stored onto the phone and exported for analysis. True sleep stage values were determined by an already-fabricated sleep app.


# Training a Classifier

(In python/A2.5/activity-classification-train.py)  
  
All CSVs were loaded and stacked together. After merging the heartrate and acceleration datasets together by timestamp, a random forest classifier worked best to classify the data. Results are as follows:
  
average accuracy: 0.980  
average precision awake: 0.983  
average recall awake: 0.981  
average precision light sleep: 0.967  
average recall light sleep: 0.988  
average precision deep sleep: 0.994  
average recall deep sleep: 0.902  
