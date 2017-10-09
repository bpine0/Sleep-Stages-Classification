package cs.umass.edu.myactivitiestoolkit.view.fragments;

import android.app.Fragment;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.support.annotation.NonNull;
import android.support.v4.content.LocalBroadcastManager;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CompoundButton;
import android.widget.Switch;
import android.widget.TextView;

import com.androidplot.util.PixelUtils;
import com.androidplot.xy.BoundaryMode;
import com.androidplot.xy.LineAndPointFormatter;
import com.androidplot.xy.SimpleXYSeries;
import com.androidplot.xy.StepMode;
import com.androidplot.xy.XYGraphWidget;
import com.androidplot.xy.XYPlot;

import java.text.FieldPosition;
import java.text.Format;
import java.text.ParsePosition;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Locale;
import java.util.Queue;

import cs.umass.edu.myactivitiestoolkit.R;
import cs.umass.edu.myactivitiestoolkit.constants.Constants;
import cs.umass.edu.myactivitiestoolkit.services.AccelerometerService;
import cs.umass.edu.myactivitiestoolkit.services.ServiceManager;
import cs.umass.edu.myactivitiestoolkit.services.msband.BandService;

/**
 * Fragment which visualizes the 3-axis accelerometer signal, displays the step count estimates and
 * current activity to the user and allows the user to interact with the {@link AccelerometerService}.
 * <br><br>
 *
 * <b>ASSIGNMENT 0 (Data Collection & Visualization)</b> :
 *      In this assignment, you will display and visualize the accelerometer readings
 *      and send the data to the server. The framework is there for you; you only need
 *      to make the calls in the {@link AccelerometerService} to communicate the data.
 * <br><br>
 *
 * <b>ASSIGNMENT 1 (Step Detection)</b> :
 *      In this assignment, you will detect steps using the accelerometer sensor. You
 *      will design both a local step detection algorithm and a server-side (Python)
 *      step detection algorithm. Your algorithm should look for peaks and account for
 *      the fact that humans generally take steps every 0.5 - 2.0 seconds. Your local
 *      and server-side algorithms may be functionally identical, or you may choose
 *      to take advantage of other Python tools/libraries to improve performance.
 *  <br><br>
 *
 *  <b>ASSIGNMENT 2 (Activity Detection)</b> :
 *      In this assignment, you will classify the user's activity based on the
 *      accelerometer data. The only modification you should make to the mobile
 *      app is to register a listener which will parse the activity from the acquired
 *      {@link org.json.JSONObject} and update the UI. The real work, that is
 *      your activity detection algorithm, will be running in the Python script
 *      and acquiring data via the server.
 *
 * @author CS390MB
 *
 * @see AccelerometerService
 * @see XYPlot
 * @see Fragment
 */
public class ExerciseFragment extends Fragment {

    /** Used during debugging to identify logs by class. */
    @SuppressWarnings("unused")
    private static final String TAG = ExerciseFragment.class.getName();

    /** The switch which toggles the {@link AccelerometerService}. **/
    private Switch switchAccelerometer;

    /** Displays the accelerometer x, y and z-readings. **/
    private TextView txtAccelerometerReading;

    /** Displays the step count computed by the built-in Android step detector. **/
    private TextView txtAndroidStepCount;

    /** Displays the step count computed by your local step detection algorithm. **/
    private TextView txtLocalStepCount;

    /** Displays the step count computed by your server-side step detection algorithm. **/
    private TextView txtServerStepCount;

    /** Displays the activity identified by your server-side activity classification algorithm. **/
    private TextView txtActivity;

    /** The plot which displays the PPG data in real-time. **/
    private XYPlot mPlot;

    /** The series formatter that defines how the x-axis signal should be displayed. **/
    private LineAndPointFormatter mXSeriesFormatter;

    /** The series formatter that defines how the y-axis signal should be displayed. **/
    private LineAndPointFormatter mYSeriesFormatter;

    /** The series formatter that defines how the z-axis signal should be displayed. **/
    private LineAndPointFormatter mZSeriesFormatter;

    /** The series formatter that defines how the peaks should be displayed. **/
    private LineAndPointFormatter mPeakSeriesFormatter;

    private SimpleXYSeries xSeries, ySeries, zSeries, peaks;

    /** The number of data points to display in the graph. **/
    private static final int GRAPH_CAPACITY = 100;

    /** The number of points displayed on the plot. This should only ever be less than
     * {@link #GRAPH_CAPACITY} before the plot is fully populated. **/
    private int mNumberOfPoints = 0;

    /**
     * The queue of timestamps.
     */
    private final Queue<Number> mTimestamps = new LinkedList<>();

    /**
     * The queue of accelerometer values along the x-axis.
     */
    private final Queue<Number> mXValues = new LinkedList<>();

    /**
     * The queue of accelerometer values along the y-axis.
     */
    private final Queue<Number> mYValues = new LinkedList<>();

    /**
     * The queue of accelerometer values along the z-axis.
     */
    private final Queue<Number> mZValues = new LinkedList<>();

    /**
     * The queue of peak timestamps.
     */
    private final Queue<Number> mPeakTimestamps = new LinkedList<>();

    /**
     * The queue of peak values.
     */
    private final Queue<Number> mPeakValues = new LinkedList<>();

    /** Reference to the service manager which communicates to the {@link AccelerometerService}. **/
    private ServiceManager mServiceManager;

    /**
     * The receiver listens for messages from the {@link AccelerometerService}, e.g. was the
     * service started/stopped, and updates the status views accordingly. It also
     * listens for sensor data and displays the sensor readings to the user.
     *
     * In Assignment 1 (Step detection), you should listen for intent actions indicating that
     * a step was taken, as predicted by your server side algorithm. We already do this for the
     * built-in and local step detection algorithms. You may display the step count in
     * {@link #txtServerStepCount}.
     */
    private final BroadcastReceiver receiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            // TODO: (Assignment 1) Display the step count as predicted by your server-side algorithm, when a step event occurs
            if (intent.getAction() != null) {
                if (intent.getAction().equals(Constants.ACTION.BROADCAST_MESSAGE)) {
                    int message = intent.getIntExtra(Constants.KEY.MESSAGE, -1);
                    if (message == Constants.MESSAGE.ACCELEROMETER_SERVICE_STOPPED) {
                        switchAccelerometer.setChecked(false);
                    } else if (message == Constants.MESSAGE.BAND_SERVICE_STOPPED){
                        switchAccelerometer.setChecked(false);
                    }
                } else if (intent.getAction().equals(Constants.ACTION.BROADCAST_ACCELEROMETER_DATA)) {
                    long timestamp = intent.getLongExtra(Constants.KEY.TIMESTAMP, -1);
                    float[] accelerometerValues = intent.getFloatArrayExtra(Constants.KEY.ACCELEROMETER_DATA);
                    displayAccelerometerReading(accelerometerValues[0], accelerometerValues[1], accelerometerValues[2]);

                    mTimestamps.add(timestamp);
                    mXValues.add(accelerometerValues[0]);
                    mYValues.add(accelerometerValues[1]);
                    mZValues.add(accelerometerValues[2]);
                    xSeries.addFirst(timestamp, accelerometerValues[0]);
                    ySeries.addFirst(timestamp, accelerometerValues[1]);
                    zSeries.addFirst(timestamp, accelerometerValues[2]);
                    if (mNumberOfPoints >= GRAPH_CAPACITY) {
                        mTimestamps.poll();
                        mXValues.poll();
                        mYValues.poll();
                        mZValues.poll();
                        xSeries.removeLast();
                        ySeries.removeLast();
                        zSeries.removeLast();
                        while (mPeakTimestamps.size() > 0 && (mPeakTimestamps.peek().longValue() < mTimestamps.peek().longValue())){
                            mPeakTimestamps.poll();
                            mPeakValues.poll();
                            peaks.removeLast();
                        }
                    }
                    else
                        mNumberOfPoints++;

//                    updatePlot();
                    mPlot.redraw();

                } else if (intent.getAction().equals(Constants.ACTION.BROADCAST_ANDROID_STEP_COUNT)) {
                    int stepCount = intent.getIntExtra(Constants.KEY.STEP_COUNT, 0);
                    displayAndroidStepCount(stepCount);
                } else if (intent.getAction().equals(Constants.ACTION.BROADCAST_LOCAL_STEP_COUNT)) {
                    int stepCount = intent.getIntExtra(Constants.KEY.STEP_COUNT, 0);
                    displayLocalStepCount(stepCount);
                } else if (intent.getAction().equals(Constants.ACTION.BROADCAST_SERVER_STEP_COUNT)) {
                    int stepCount = intent.getIntExtra(Constants.KEY.STEP_COUNT, 0);
                    displayServerStepCount(stepCount);

                } else if (intent.getAction().equals(Constants.ACTION.BROADCAST_ACCELEROMETER_PEAK)){
                    long timestamp = intent.getLongExtra(Constants.KEY.ACCELEROMETER_PEAK_TIMESTAMP, -1);
//                    float[] values = intent.getFloatArrayExtra(Constants.KEY.ACCELEROMETER_PEAK_VALUE);
                    if (timestamp > 0) {
                        mPeakTimestamps.add(timestamp);
                        mPeakValues.add(0); //place on z-axis signal
                        peaks.addFirst(timestamp, 0);
                    }
                }
            }
        }
    };

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        this.mServiceManager = ServiceManager.getInstance(getActivity());
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        final View view = inflater.inflate(R.layout.fragment_exercise, container, false);

        //obtain a reference to the accelerometer reading text field
        txtAccelerometerReading = (TextView) view.findViewById(R.id.txtAccelerometerReading);

        //obtain references to the step count text fields
        txtAndroidStepCount = (TextView) view.findViewById(R.id.txtAndroidStepCount);
        txtLocalStepCount = (TextView) view.findViewById(R.id.txtLocalStepCount);
        txtServerStepCount = (TextView) view.findViewById(R.id.txtServerStepCount);

        //obtain reference to the activity text field
        txtActivity = (TextView) view.findViewById(R.id.txtActivity);

        //obtain references to the on/off switches and handle the toggling appropriately
        switchAccelerometer = (Switch) view.findViewById(R.id.switchAccelerometer);
        switchAccelerometer.setChecked(mServiceManager.isServiceRunning(AccelerometerService.class));
        switchAccelerometer.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean enabled) {
                if (enabled){
                    clearPlotData();

                    SharedPreferences preferences = PreferenceManager.getDefaultSharedPreferences(getActivity());
                    boolean runOverMSBand = preferences.getBoolean(getString(R.string.pref_msband_key),
                            getResources().getBoolean(R.bool.pref_msband_default));
                    if (runOverMSBand){
                        mServiceManager.startSensorService(BandService.class);
                    }else{
                        mServiceManager.startSensorService(AccelerometerService.class);
                    }
                }else{
                    mServiceManager.stopSensorService(AccelerometerService.class);
                    mServiceManager.stopSensorService(BandService.class);
                }
            }
        });

        // initialize plot and set plot parameters
        mPlot = (XYPlot) view.findViewById(R.id.accelerometerPlot);
        mPlot.setRangeBoundaries(-30, 30, BoundaryMode.FIXED);
        mPlot.setRangeStep(StepMode.SUBDIVIDE, 5);
        mPlot.getGraph().getDomainOriginLinePaint().setColor(Color.TRANSPARENT);
        mPlot.getGraph().getDomainGridLinePaint().setColor(Color.TRANSPARENT);
        mPlot.getGraph().getRangeGridLinePaint().setColor(Color.TRANSPARENT);
        mPlot.setDomainStep(StepMode.SUBDIVIDE, 1);

        // To remove the x labels, just set each label to an empty string:
        mPlot.getGraph().getLineLabelStyle(XYGraphWidget.Edge.BOTTOM).setFormat(new Format() {
            @Override
            public StringBuffer format(Object obj, @NonNull StringBuffer toAppendTo, @NonNull FieldPosition pos) {
                return toAppendTo.append("");
            }
            @Override
            public Object parseObject(String source, @NonNull ParsePosition pos) {
                return null;
            }
        });
        mPlot.setPlotPaddingBottom(-150); //TODO: This isn't device-dependent, might need to be changed if the plot/legend is cut off
        mPlot.getLegend().setPaddingBottom(280);

        // set formatting parameters for each signal (accelerometer and accelerometer peaks)
        mXSeriesFormatter = new LineAndPointFormatter(Color.RED, null, null, null);
        mYSeriesFormatter = new LineAndPointFormatter(Color.GREEN, null, null, null);
        mZSeriesFormatter = new LineAndPointFormatter(Color.BLUE, null, null, null);

        mPeakSeriesFormatter = new LineAndPointFormatter(null, Color.BLUE, null, null);
        mPeakSeriesFormatter.getVertexPaint().setStrokeWidth(PixelUtils.dpToPix(10)); //enlarge the peak points

        xSeries = new SimpleXYSeries(new ArrayList<>(mTimestamps), new ArrayList<>(mXValues), "X");
        ySeries = new SimpleXYSeries(new ArrayList<>(mTimestamps), new ArrayList<>(mYValues), "Y");
        zSeries = new SimpleXYSeries(new ArrayList<>(mTimestamps), new ArrayList<>(mZValues), "Z");

        peaks = new SimpleXYSeries(new ArrayList<>(mPeakTimestamps), new ArrayList<>(mPeakValues), "PEAKS");

        mPlot.addSeries(xSeries, mXSeriesFormatter);
        mPlot.addSeries(ySeries, mYSeriesFormatter);
        mPlot.addSeries(zSeries, mZSeriesFormatter);
        mPlot.addSeries(peaks, mPeakSeriesFormatter);
        mPlot.redraw();

        return view;
    }

    /**
     * When the fragment starts, register a {@link #receiver} to receive messages from the
     * {@link AccelerometerService}. The intent filter defines messages we are interested in receiving.
     * <br><br>
     *
     * We would like to receive sensor data, so we specify {@link Constants.ACTION#BROADCAST_ACCELEROMETER_DATA}.
     * We would also like to receive step count updates, so include {@link Constants.ACTION#BROADCAST_ANDROID_STEP_COUNT}
     * and {@link Constants.ACTION#BROADCAST_LOCAL_STEP_COUNT}.
     * <br><br>
     *
     * To optionally display the peak values you compute, include
     * {@link Constants.ACTION#BROADCAST_ACCELEROMETER_PEAK}.
     * <br><br>
     *
     * Lastly to update the state of the accelerometer switch properly, we listen for additional
     * messages, using {@link Constants.ACTION#BROADCAST_MESSAGE}.
     *
     * @see Constants.ACTION
     * @see IntentFilter
     * @see LocalBroadcastManager
     * @see #receiver
     */
    @Override
    public void onStart() {
        super.onStart();

        LocalBroadcastManager broadcastManager = LocalBroadcastManager.getInstance(getActivity());
        IntentFilter filter = new IntentFilter();
        filter.addAction(Constants.ACTION.BROADCAST_MESSAGE);
        filter.addAction(Constants.ACTION.BROADCAST_ACCELEROMETER_DATA);
        filter.addAction(Constants.ACTION.BROADCAST_ACCELEROMETER_PEAK);
        filter.addAction(Constants.ACTION.BROADCAST_ANDROID_STEP_COUNT);
        filter.addAction(Constants.ACTION.BROADCAST_LOCAL_STEP_COUNT);
        filter.addAction(Constants.ACTION.BROADCAST_SERVER_STEP_COUNT);
        broadcastManager.registerReceiver(receiver, filter);
    }

    /**
     * When the fragment stops, e.g. the user closes the application or opens a new activity,
     * then we should unregister the {@link #receiver}.
     */
    @Override
    public void onStop() {
        LocalBroadcastManager broadcastManager = LocalBroadcastManager.getInstance(getActivity());
        try {
            broadcastManager.unregisterReceiver(receiver);
        }catch (IllegalArgumentException e){
            e.printStackTrace();
        }
        super.onStop();
    }

    /**
     * Displays the accelerometer reading on the UI.
     * @param x acceleration along the x-axis
     * @param y acceleration along the y-axis
     * @param z acceleration along the z-axis
     */
    private void displayAccelerometerReading(final float x, final float y, final float z){
        getActivity().runOnUiThread(new Runnable() {
            @Override
            public void run() {
                txtAccelerometerReading.setText(String.format(Locale.getDefault(), getActivity().getString(R.string.accelerometer_reading_format_string), x, y, z));
            }
        });
    }

    /**
     * Displays the step count as computed by your local step detection algorithm.
     * @param stepCount the number of steps taken since the service started
     */
    private void displayLocalStepCount(final int stepCount){
        getActivity().runOnUiThread(new Runnable() {
            @Override
            public void run() {
                txtLocalStepCount.setText(String.format(Locale.getDefault(), getString(R.string.local_step_count), stepCount));
            }
        });
    }

    /**
     * Displays the step count as computed by the Android built-in step detection algorithm.
     * @param stepCount the number of steps taken since the service started
     */
    private void displayAndroidStepCount(final int stepCount){
        getActivity().runOnUiThread(new Runnable() {
            @Override
            public void run() {
                txtAndroidStepCount.setText(String.format(Locale.getDefault(), getString(R.string.android_step_count), stepCount));
            }
        });
    }

    /**
     * Displays the step count as computed by your server-side step detection algorithm.
     * @param stepCount the number of steps taken since the service started
     */
    private void displayServerStepCount(final int stepCount){
        getActivity().runOnUiThread(new Runnable() {
            @Override
            public void run() {
                txtServerStepCount.setText(String.format(Locale.getDefault(), getString(R.string.server_step_count), stepCount));
            }
        });
    }



    /**
     * Clears the x, y, z and peak plot data series.
     */
    private void clearPlotData(){
        mPeakTimestamps.clear();
        mPeakValues.clear();
        mTimestamps.clear();
        mXValues.clear();
        mYValues.clear();
        mZValues.clear();
        mNumberOfPoints = 0;

        int dataSize = xSeries.size();
        for (int i = 0; i < dataSize; i++)
        {
            xSeries.removeLast();
            ySeries.removeLast();
            zSeries.removeLast();
        }

        int peakSize = peaks.size();
        for (int i = 0; i < peakSize; i++)
            peaks.removeLast();

        mPlot.redraw();
    }

    /**
     * Updates and redraws the accelerometer plot, along with the peaks detected.
     */
    private void updatePlot(){

        //redraw the plot:
//        mPlot.clear();
//        mPlot.addSeries(xSeries, mXSeriesFormatter);
//        mPlot.addSeries(ySeries, mYSeriesFormatter);
//        mPlot.addSeries(zSeries, mZSeriesFormatter);
//        mPlot.addSeries(peaks, mPeakSeriesFormatter);
//        mPlot.redraw();
    }
}