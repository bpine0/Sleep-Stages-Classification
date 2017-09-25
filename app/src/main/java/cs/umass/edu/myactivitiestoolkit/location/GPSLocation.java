package cs.umass.edu.myactivitiestoolkit.location;

import cs.umass.edu.myactivitiestoolkit.clustering.Clusterable;

/**
 * This class represents a GPS location event.
 * 
 * @author Abhinav Parate
 */
public class GPSLocation implements Clusterable<GPSLocation> {

    public static final int RADIUS_OF_EARTH_IN_METERS = 6371000;

	/** Report id */
	public int id = -1;
	
	/** timestamp of the event */
	public long timestamp;
	
	/** latitude */
	public double latitude;
	
	/** longitude */
	public double longitude;
	
	/** accuracy */
	public float accuracy;
		
	/**
	 * Constructor without id
	 * @param timestamp the timestamp at which the GPS location was recorded.
	 * @param lat the latitude.
	 * @param lng the longitude.
	 * @param accuracy the accuracy of the reading.
	 */
	public GPSLocation(long timestamp, double lat, double lng, float accuracy) {
		this.timestamp = timestamp;
		this.latitude = lat;
		this.longitude = lng;
		this.accuracy = accuracy;
	}
	
	/**
	 * Constructor with report id
	 * @param id a unique identifier for the GPS location.
	 * @param timestamp the timestamp at which the GPS location was recorded.
	 * @param lat the latitude.
	 * @param lng the longitude.
	 * @param accuracy the accuracy of the reading.
	 */
	public GPSLocation(int id, long timestamp, double lat, double lng, float accuracy) {
		this.id =id;
		this.timestamp = timestamp;
		this.latitude = lat;
		this.longitude = lng;
		this.accuracy = accuracy;
	}
	
	/**
	 * Empty Constructor
	 */
	public GPSLocation() {}
	
	public boolean equals(GPSLocation r){
		return this.id == r.id;
	}

	/**
	 * Returns the latitude of the GPS reading.
	 * @return location's latitude.
     */
	public double getLatitude() {
		return latitude;
	}

	/**
	 * Returns the longitude of the GPS reading.
	 * @return location's longitude.
	 */
	public double getLongitude() {
		return longitude;
	}

	/**
	 * Returns the accuracy of the GPS reading.
	 * @return accuracy of the reading.
	 */
	public float getAccuracy() {
		return accuracy;
	}

	/**
	 * Returns the unique identifier of the GPS reading.
	 * @return location's ID.
	 */
	public int getId() {
		return id;
	}

	/**
	 * Returns the timestamp at which the GPS reading was recorded.
	 * @return reading's timestamp.
	 */
	public long getTimestamp() {
		return timestamp;
	}

	/**
	 * The distance between two GPS coordinates can be computed using the
	 * <a href = https://en.wikipedia.org/wiki/Haversine_formula>Haversine</a> formula.
	 * @param other the other GPS coordinate we're measuring distance to
	 * @return the distance between the two coordinates
	 */
	@Override
	public double distance(GPSLocation other) {
		double deltaLatitude = other.latitude - this.latitude;
		double deltaLongitude = other.longitude - this.longitude;

		//convert change in latitude and longitude to radians
		double dLat = Math.toRadians(deltaLatitude);
		double dLng = Math.toRadians(deltaLongitude);

		//compute the haversine of the latitude and longitude deltas
		double haversinLat = (1 - Math.cos(dLat))/2; // = Math.sin(dLat/2)*Math.sin(dLat/2);
		double haversinLng = (1 - Math.cos(dLng))/2;

		//compute the haversine of the central angle between the two points
		double haversinCentralAngle = haversinLat + Math.cos(Math.toRadians(this.latitude)) * Math.cos(Math.toRadians(other.longitude)) *  haversinLng;

		//inverse the haversine function using the arctan to get the distance as a measure of the angle difference
		double d = 2 * Math.atan2(Math.sqrt(haversinCentralAngle), Math.sqrt(1-haversinCentralAngle));

		//We want to compute the arc length s=r*d, so simply multiply by the radius
		return RADIUS_OF_EARTH_IN_METERS * d;
	}

	/*double a = Math.sin(dLat/2) * Math.sin(dLat/2) +
		Math.cos(Math.toRadians(this.latitude)) * Math.cos(Math.toRadians(other.longitude)) *
				Math.sin(dLng/2) * Math.sin(dLng/2);*/
}