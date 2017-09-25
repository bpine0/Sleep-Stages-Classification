package cs.umass.edu.myactivitiestoolkit.constants;

/**
 * The Constants class stores various constants that are used across various classes, including
 * identifiers for intent actions used when the main UI communicates with the sensor services.
 *
 * @author CS390MB
 *
 */
public class Constants {

    /** Intent actions used to communicate between the main UI and the sensor service
     * @see android.content.Intent */
    public interface ACTION {
        String START_SERVICE = "edu.umass.cs.my-activities-toolkit.action.start-service";
        String STOP_SERVICE = "edu.umass.cs.my-activities-toolkit.action.stop-service";
        String NAVIGATE_TO_APP = "edu.umass.cs.my-activities-toolkit.action.navigate-to-app";

        String BROADCAST_MESSAGE = "edu.umass.cs.my-activities-toolkit.action.broadcast-message";
        String BROADCAST_STATUS = "edu.umass.cs.my-activities-toolkit.action.broadcast-status";

        String BROADCAST_ACCELEROMETER_DATA = "edu.umass.cs.my-activities-toolkit.action.broadcast-accelerometer-data";
        String BROADCAST_ANDROID_STEP_COUNT = "edu.umass.cs.my-activities-toolkit.action.broadcast-android-step-count";
        String BROADCAST_LOCAL_STEP_COUNT = "edu.umass.cs.my-activities-toolkit.action.broadcast-local-step-count";
        String BROADCAST_SERVER_STEP_COUNT = "edu.umass.cs.my-activities-toolkit.action.broadcast-server-step-count";
        String BROADCAST_ACCELEROMETER_PEAK = "edu.umass.cs.my-activities-toolkit.action.broadcast-accelerometer-peak";

        String BROADCAST_ACTIVITY = "edu.umass.cs.my-activities-toolkit.action.broadcast-activity";

        String BROADCAST_PPG = "edu.umass.cs.my-activities-toolkit.action.broadcast-ppg";
        String BROADCAST_PPG_PEAK = "edu.umass.cs.my-activities-toolkit.action.broadcast-ppg-peak";
        String BROADCAST_HEART_RATE = "edu.umass.cs.my-activities-toolkit.action.broadcast-heart-rate";

        String BROADCAST_SPECTROGRAM = "edu.umass.cs.my-activities-toolkit.action.broadcast-spectrogram";
    }

    /**
     * Unique IDs associated with each service notification.
     */
    public interface NOTIFICATION_ID {
        int ACCELEROMETER_SERVICE = 101;
        int PPG_SERVICE = 102;
        int LOCATION_SERVICE = 103;
        int AUDIO_SERVICE = 104;
    }

    /** Keys to identify key-value data sent to/from the sensor service */
    public interface KEY {
        String NOTIFICATION_ID = "edu.umass.cs.my-activities-toolkit.key.sensor-service-type";
        String MESSAGE = "edu.umass.cs.my-activities-toolkit.key.message";
        String STATUS = "edu.umass.cs.my-activities-toolkit.key.status";

        String TIMESTAMP = "edu.umass.cs.my-activities-toolkit.key.ppg-timestamp";

        String ACCELEROMETER_DATA = "edu.umass.cs.my-activities-toolkit.key.accelerometer-data";
        String ACCELEROMETER_PEAK_TIMESTAMP = "edu.umass.cs.my-activities-toolkit.key.accelerometer-peak-timestamp";
        String ACCELEROMETER_PEAK_VALUE = "edu.umass.cs.my-activities-toolkit.key.accelerometer-peak-value";
        String STEP_COUNT = "edu.umass.cs.my-activities-toolkit.key.step-count";

        String ACTIVITY = "edu.umass.cs.my-activities-toolkit.key.activity";

        String PPG_DATA = "edu.umass.cs.my-activities-toolkit.key.ppg-value";
        String PPG_PEAK_TIMESTAMP = "edu.umass.cs.my-activities-toolkit.key.ppg-peak-timestamp";
        String PPG_PEAK_VALUE = "edu.umass.cs.my-activities-toolkit.key.ppg-peak-value";
        String HEART_RATE = "edu.umass.cs.my-activities-toolkit.key.heart-rate";

        String SPECTROGRAM = "edu.umass.cs.my-activities-toolkit.key.spectrogram";
    }

    /**
     * Messages sent to the main UI to update the status. These must be unique values.
     */
    public interface MESSAGE {
        int ACCELEROMETER_SERVICE_STARTED = 0;
        int ACCELEROMETER_SERVICE_STOPPED = 1;
        int PPG_SERVICE_STARTED = 2;
        int PPG_SERVICE_STOPPED = 3;
        int AUDIO_SERVICE_STARTED = 4;
        int AUDIO_SERVICE_STOPPED = 5;
        int LOCATION_SERVICE_STARTED = 6;
        int LOCATION_SERVICE_STOPPED = 7;
        int BAND_SERVICE_STARTED = 8;
        int BAND_SERVICE_STOPPED = 9;
    }

    /** Error/warning messages displayed to the user TODO: put into string resources */
    public interface ERROR_MESSAGES {
        String ERROR_NO_ACCELEROMETER = "ERROR: No accelerometer available...";
        String ERROR_NO_SENSOR_MANAGER = "ERROR: Could not retrieve sensor manager...";
        String WARNING_SENSOR_NOT_SUPPORTED = "WARNING: Sensor not supported!";
    }

    /** Timestamp-relevant constants */
    public interface TIMESTAMPS {
        long NANOSECONDS_PER_MILLISECOND = 1000000;
    }

    /**
     * Identifies common data filters used when receiving data from the server.
     * <br><br>
     * As an example, to listen for steps and activities detected, register a
     * message receiver object as follows:
     *
     * <pre>
     * {@code setMessageReceiver(new MessageReceiver(MessageReceiver.Filter.STEP_DETECTED, MessageReceiver.Filter.ACTIVITY_DETECTED) {
     *         @literal @Override
     *          void onMessageReceived(JSONObject json) {
     *              //parse json, handle message
     *          }
     *   });
     * }
     * </pre>
     */
    public interface MHLClientFilter {
        String STEP_DETECTED = "STEP_DETECTED";
        String ACTIVITY_DETECTED = "ACTIVITY_DETECTED";
        String SPEAKER_DETECTED = "SPEAKER_DETECTED";
        String CLUSTER = "CLUSTER";
    }
}
