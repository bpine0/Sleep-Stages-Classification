# CS328 A2.5-Activity Recognition (The Sequel!)

Completing the Loop

Now that you’re savvy on supervised learning, you’re ready to train and deploy your activity classifier. Use the collect-labelled-activity-data.py (in the A2.5 repo) for collecting activity-labelled raw accelerometer data via the data collection server. It simply stores incoming data into a list and writes the data to disk in the same format as the sample activity data we provided in the part 2. After you modify your Android application to collect labelled data, you will be able to run your activity-classification-train.py script from part 2, which automatically saves the classifier. For testing, use the activity-recognition.py script.

A note on extensions
Please be aware that, due to schedule constraints, we will not be offering extensions on this assignment unless there are serious technical difficulties with the provided code (in which case we expect to be notified early so that we can help you sort it out). With that in mind, we will also not be primarily grading this assignment on the performance of the classifier you train.

Building a robust classification model for activities is difficult, we just want to see that you can implement it and try it on live data. The goal is for you to be able to demonstrate some notable performance improvement by turning certain knobs (feature set or classifier parameters).

Acquiring Labelled Data

Before we can train a classifier for the activity recognition task, we need to acquire training data with labels corresponding to the activity.  We will be using the A2.5 My Activities Android app both to collect labeled data, and to display detected activity labels in real time once the classifier has been trained.

You will be classifying at least 4 activities, two of which are walking and sitting. Select at least two additional activities. Here are some suggestions:

Jogging
Running
Biking
Jumping

You can choose anything you like, but please make sure it’s safe and doable while you have your phone on you. 

Adding labels to the app for data collection is easy -- look in the res > strings.xml file for the array of activity labels.  Add any activities that are of interest to you, being sure to follow the same format as in the example label.  (Feel free to remove the example if running is not an activity you want to learn!)  The label strings must begin with a unique, non-negative integer.  This integer will be stripped and sent to the backend as the label for use in the model-learner.  We recommend creating a simple mapping from activities to labels that everyone on the team has access to. Just make sure it is consistent everywhere.

For each activity, collect at least 5 minutes of data. Use the switch in the Android app to connect to the backend and begin transmitting data.  Use the spinner in the top left of the window to select an activity from the menu you populated above.  Make sure that you turn on data collection before using the spinner to label the data!  In the output of collect-labelled-activity-data.py you’ll begin to see incoming labeled data.  (Unlabeled data will be ignored).  Collect as much data as you can (more is better), then turn off data collection in the app.  Repeat this process for each activity.  When you have collected data for all the activities you’re interested in classifying, you will have a dataset similar to the one provided in A2.0.

In your write-up, clearly describe the data collection process. If you ever require research approval in the future, this will be a very important skill to have. Talk specifically about how much data you collected and at what orientation and position the phone was held. How many people performed the activities? Who were they? Can you identify any potential sources of inconsistencies in the data? Are there any external factors we should be aware of? Etc.

Training a Classifier

Training a classifier should be very simple now, since you have already done it in A2.0.  The same starter code you used in A2.0 has been provided in the A2.5 repo for your convenience. You may have to make a few minor modifications to make sure the code still works now that you have more classes than before. The script is already set up to save the trained classifier to disk using pickle. So choose your best classifier and make sure to report the cross-validated accuracy, precision and recall metrics on your dataset. How do the results compare to the previous results on the sample data?

Test & Deploy

When you run the activity-recognition.py script, it will load the classifier from disk. Upon receiving data from over the server, the script already buffers one second of data and sends it to the detectActivity() function. There, you need to call the predict() function of your classifier, then send the activity back to the phone using the onActivityDetected() function. The AccelerometerService is already configured to receive activity messages and send the result back to the UI.

Write-up

Describe the data collection process. Make sure it is thorough and unambiguous; imagine you are submitting a paper to a conference or demoing a commercial system: You want to cover your bases. Here are some questions you may want to answer:
Who were the research subjects?
What activities were being performed?
How long were the activities performed?
How did they hold their phone?
How did they perform the motion?
Did you account for variation in the phone position/orientation and the activity style? We don’t expect that your application works with the phone in any position/orientation, but we do expect you to be clear about when your application is expected to work.
Which classification algorithm and parameters did you select in your final system? Why?
How did the classifier perform initially? Try at making at least two changes to the model (use different features or different parameters to the classifier) and report how the results change.
Describe your results. Report the accuracy, precision and recall metrics for the classifier and features you decided to use. How do the results compare to the results on the sample data? Briefly speak to how well your algorithm works in practice, drawing on the empirical results. Do they match up?

Deadline

As always, mark your submission commit with “final” and push by Wednesday, November 15th at 11:55 pm. Also include a writeup file in the repository including your group number and teammate names and any additional comments, recommendations or concerns.

Remember, there will be a demo for A2.5 using live data from the Android app.

Happy coding!
