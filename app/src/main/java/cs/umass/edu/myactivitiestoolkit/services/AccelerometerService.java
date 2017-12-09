package cs.umass.edu.myactivitiestoolkit.services;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.support.v4.content.LocalBroadcastManager;
import android.util.Log;
import android.widget.Spinner;
import android.os.Environment;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

import cs.umass.edu.myactivitiestoolkit.R;
import cs.umass.edu.myactivitiestoolkit.constants.Constants;
import cs.umass.edu.myactivitiestoolkit.steps.StepDetector;
import edu.umass.cs.MHLClient.client.MessageReceiver;
import edu.umass.cs.MHLClient.sensors.AccelerometerReading;


/**
 * This service is responsible for collecting the accelerometer data on
 * the phone. It is an ongoing foreground service that will run even when the
 * application is not running. Note, however, that a process of the application
 * will still be running! The sensor service will receive sensor events in the
 * {@link #onSensorChanged(SensorEvent)} method defined in the {@link SensorEventListener}
 * interface.
 *
 */

public class AccelerometerService extends SensorService implements SensorEventListener {


    private static int myCounter = 0;

    /** Used during debugging to identify logs by class */
    private static final String TAG = AccelerometerService.class.getName();

    /** Sensor Manager object for registering and unregistering system sensors */
    private SensorManager mSensorManager;

    /** Manages the physical accelerometer sensor on the phone. */
    private Sensor mAccelerometerSensor;

    /** Android built-in step detection sensor **/
    private Sensor mStepSensor;

    /** Defines your step detection algorithm. **/
    private final StepDetector mStepDetector;

    /** The step count as predicted by the Android built-in step detection algorithm. */
    private int mAndroidStepCount = 0;

    /**
     * The step count as predicted by your server-side step detection algorithm.
     */
    private int serverStepCount = 0;

    /** The spinner containing the activity label. */
    Spinner spinner;

    /** The activity label for data collection. */
    String label = "";


    public AccelerometerService(){
        mStepDetector = new StepDetector();
    }

    @Override
    protected void onServiceStarted() {
        broadcastMessage(Constants.MESSAGE.ACCELEROMETER_SERVICE_STARTED);

//        LayoutInflater inflater = (LayoutInflater) getSystemService(LAYOUT_INFLATER_SERVICE);
//        View layout = inflater.inflate(R.layout.fragment_exercise, null);
//        spinner = (Spinner)layout.findViewById(R.id.spinner_activity);
//        spinner.setOnItemSelectedListener(this);

//        Log.i(TAG, spinner.getAdapter().)
        Log.i(TAG, "registered spinner listener");

        BroadcastReceiver receiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                if (intent.getAction().equals("LABEL")) {
                    label = intent.getStringExtra("LABEL");
                }
            }
        };

        LocalBroadcastManager localBroadcastManager = LocalBroadcastManager.getInstance(this);
        localBroadcastManager.registerReceiver(receiver, new IntentFilter("LABEL"));

    }

    @Override
    protected void onServiceStopped() {

        broadcastMessage(Constants.MESSAGE.ACCELEROMETER_SERVICE_STOPPED);

//        LayoutInflater inflater = (LayoutInflater) getSystemService(LAYOUT_INFLATER_SERVICE);
//        View layout = inflater.inflate(R.layout.fragment_exercise, null);
//        spinner = (Spinner)layout.findViewById(R.id.spinner_activity);
//        spinner.listener

    }

    @Override
    public void onConnected() {
        super.onConnected();
        mClient.registerMessageReceiver(new MessageReceiver(Constants.MHLClientFilter.STEP_DETECTED) {
            @Override
            protected void onMessageReceived(JSONObject json) {
                Log.d(TAG, "Received step update from server.");
                try {
                    JSONObject data = json.getJSONObject("data");
                    long timestamp = data.getLong("timestamp");
                    Log.d(TAG, "Step occurred at " + timestamp + ".");
                    serverStepCount++;
                    broadcastServerStepCount(serverStepCount);
                    broadcastStepDetected(timestamp);
                } catch (JSONException e) {
                    e.printStackTrace();
                }
            }
        });
        mClient.registerMessageReceiver(new MessageReceiver(Constants.MHLClientFilter.ACTIVITY_DETECTED) {
            @Override
            protected void onMessageReceived(JSONObject json) {
                String activity;
                try {
                    JSONObject data = json.getJSONObject("data");
                    activity = data.getString("activity");
                    broadcastActivity(activity);
                } catch (JSONException e) {
                    e.printStackTrace();
                }
            }
        });
    }

    /**
     * Register accelerometer sensor listener
     */
    @Override
    protected void registerSensors(){

        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        mAccelerometerSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mSensorManager.registerListener(this, mAccelerometerSensor, SensorManager.SENSOR_DELAY_NORMAL);

        //TODO : (Assignment 0) Register the accelerometer sensor from the sensor manager.
    }

    /**
     * Unregister the sensor listener, essential for the battery life!
     */
    @Override
    protected void unregisterSensors() {
        //TODO : Unregister your sensors. Make sure mSensorManager is not null before calling its unregisterListener method.
        if (mSensorManager == null)
            mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);

        if (mAccelerometerSensor == null)
            mAccelerometerSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);

        mSensorManager.unregisterListener(this, mAccelerometerSensor);
    }

    @Override
    protected int getNotificationID() {
        return Constants.NOTIFICATION_ID.ACCELEROMETER_SERVICE;
    }

    @Override
    protected String getNotificationContentText() {
        return getString(R.string.activity_service_notification);
    }

    @Override
    protected int getNotificationIconResourceID() {
        return R.drawable.ic_running_white_24dp;
    }

    /**
     * This method is called when we receive a sensor reading.
     */

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {


            Date currentTime = Calendar.getInstance().getTime();
            SimpleDateFormat df = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");
            String timestamp = df.format(currentTime);

            float x = event.values[0];
            float y = event.values[1];
            float z = event.values[2];

            String entry = Float.toString(x) + "," + Float.toString(y) + "," + Float.toString(z) + "," + timestamp + "\n";

            myCounter++;
            if(myCounter % 5 == 0) {
                myCounter = 0;
                try {

                    File dir = new File(Environment.getExternalStoragePublicDirectory(
                            Environment.DIRECTORY_DCIM), "/data");
                    dir.mkdir();

                    FileOutputStream f = new FileOutputStream(dir + "/data.csv", true);

                    try {
                        f.write(entry.getBytes());
                        f.flush();
                        f.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
            }

            // convert the timestamp to milliseconds (note this is not in Unix time)
            long timestamp_in_milliseconds = (long) ((double) event.timestamp / Constants.TIMESTAMPS.NANOSECONDS_PER_MILLISECOND);

            int labelInt = -1;
            if (!(label.equals("") || label.equals("Label"))) {
                labelInt = Integer.parseInt("" + label.charAt(0));
            }
            mClient.sendSensorReading(new AccelerometerReading(getString(R.string.mobile_health_client_user_id), "MOBILE", "", timestamp_in_milliseconds, labelInt, event.values));

//            //TODO: Send the accelerometer reading to the server
//            mClient.sendSensorReading(new AccelerometerReading(getString(R.string.mobile_health_client_user_id), "MOBILE", "", timestamp_in_milliseconds, event.values));

            //TODO: broadcast the accelerometer reading to the UI
            broadcastAccelerometerReading(timestamp_in_milliseconds, event.values);

            //TODO: (Assignment 1) Call the detectSteps method in the StepDetector class
//            broadcastStepDetected(timestamp_in_milliseconds, event.values);



        }else if (event.sensor.getType() == Sensor.TYPE_STEP_DETECTOR) {

            // we received a step event detected by the built-in Android step detector (assignment 1)
            broadcastAndroidStepCount(mAndroidStepCount++);

        } else {

            // cannot identify sensor type
            Log.w(TAG, Constants.ERROR_MESSAGES.WARNING_SENSOR_NOT_SUPPORTED);

        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        Log.i(TAG, "Accuracy changed: " + accuracy);
    }

    /**
     * Broadcasts the accelerometer reading to other application components, e.g. the main UI.
     * @param accelerometerReadings the x, y, and z accelerometer readings
     */
    public void broadcastAccelerometerReading(final long timestamp, final float[] accelerometerReadings) {
        Intent intent = new Intent();
        intent.putExtra(Constants.KEY.TIMESTAMP, timestamp);
        intent.putExtra(Constants.KEY.ACCELEROMETER_DATA, accelerometerReadings);
        intent.setAction(Constants.ACTION.BROADCAST_ACCELEROMETER_DATA);
        LocalBroadcastManager manager = LocalBroadcastManager.getInstance(this);
        manager.sendBroadcast(intent);
    }

    // ***************** Methods for broadcasting step counts (assignment 1) *****************

    /**
     * Broadcasts the step count computed by the Android built-in step detection algorithm
     * to other application components, e.g. the main UI.
     */
    public void broadcastAndroidStepCount(int stepCount) {
        Intent intent = new Intent();
        intent.putExtra(Constants.KEY.STEP_COUNT, stepCount);
        intent.setAction(Constants.ACTION.BROADCAST_ANDROID_STEP_COUNT);
        LocalBroadcastManager manager = LocalBroadcastManager.getInstance(this);
        manager.sendBroadcast(intent);
    }

    /**
     * Broadcasts the step count computed by your step detection algorithm
     * to other application components, e.g. the main UI.
     */
    public void broadcastLocalStepCount(int stepCount) {
        Intent intent = new Intent();
        intent.putExtra(Constants.KEY.STEP_COUNT, stepCount);
        intent.setAction(Constants.ACTION.BROADCAST_LOCAL_STEP_COUNT);
        LocalBroadcastManager manager = LocalBroadcastManager.getInstance(this);
        manager.sendBroadcast(intent);
    }


    /**
     * Broadcasts the step count computed by your server-side step detection algorithm
     * to other application components, e.g. the main UI.
     */
    public void broadcastServerStepCount(int stepCount) {
        Intent intent = new Intent();
        intent.putExtra(Constants.KEY.STEP_COUNT, stepCount);
        intent.setAction(Constants.ACTION.BROADCAST_SERVER_STEP_COUNT);
        LocalBroadcastManager manager = LocalBroadcastManager.getInstance(this);
        manager.sendBroadcast(intent);
    }


    /**
     * Broadcasts a step event to other application components, e.g. the main UI.
     * Use this if you would like to visualize the detected step on the accelerometer signal.
     */
    public void broadcastStepDetected(long timestamp) {
        Intent intent = new Intent();
        intent.putExtra(Constants.KEY.ACCELEROMETER_PEAK_TIMESTAMP, timestamp);
//        intent.putExtra(Constants.KEY.ACCELEROMETER_PEAK_VALUE, values);
        intent.setAction(Constants.ACTION.BROADCAST_ACCELEROMETER_PEAK);
        LocalBroadcastManager manager = LocalBroadcastManager.getInstance(this);
        manager.sendBroadcast(intent);
    }

    /**
     * Broadcasts the step count computed by your server-side step detection algorithm
     * to other application components, e.g. the main UI.
     */
    public void broadcastActivity(String activity) {
        Intent intent = new Intent();
        intent.putExtra(Constants.KEY.ACTIVITY, activity);
        intent.setAction(Constants.ACTION.BROADCAST_ACTIVITY);
        LocalBroadcastManager manager = LocalBroadcastManager.getInstance(this);
        manager.sendBroadcast(intent);
    }
}