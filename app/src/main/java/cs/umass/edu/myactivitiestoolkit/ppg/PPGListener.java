package cs.umass.edu.myactivitiestoolkit.ppg;

/**
 * Handler for PPG sensor events.
 * @author CS390MB
 */
public interface PPGListener {
    void onSensorChanged(PPGEvent event);
}

