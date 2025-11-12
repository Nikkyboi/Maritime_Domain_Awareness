import numpy as np


def validate_baselineModel(y_test, y_pred, threshold_meters=50.0):
    """
    Validate the baseline model predictions against ground truth positions.
    
    For trajectory prediction, we measure if predictions are within a threshold distance.
    
    Parameters:
    -----------
    y_test : np.ndarray
        Ground truth positions, shape (n, 2) with [lat, lon]
    y_pred : np.ndarray
        Predicted positions, shape (n, 2) with [lat, lon]
    threshold_meters : float
        Distance threshold in meters to consider a prediction "correct"
        Default: 50 meters
        
    Returns:
    --------
    accuracy : float
        Fraction of predictions within the threshold (0.0 to 1.0)
    """
    if len(y_test) == 0 or len(y_pred) == 0:
        return 0.0
    
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    # Convert degree differences to approximate meters
    # At mid-latitudes: 1 degree latitude ≈ 111 km, 1 degree longitude varies
    # For simplicity, use Euclidean distance in degrees, then convert
    lat_avg = np.mean(y_test[:, 0])
    meters_per_degree_lat = 111320  # meters
    meters_per_degree_lon = 111320 * np.cos(np.radians(lat_avg))
    
    # Calculate distances in meters
    lat_diff = (y_pred[:, 0] - y_test[:, 0]) * meters_per_degree_lat
    lon_diff = (y_pred[:, 1] - y_test[:, 1]) * meters_per_degree_lon
    
    distances = np.sqrt(lat_diff**2 + lon_diff**2)
    
    # Count predictions within threshold
    correct = np.sum(distances <= threshold_meters)
    total = len(y_test)
    accuracy = correct / total
    
    return accuracy


def calculate_distance_errors(y_test, y_pred):
    """
    Calculate distance-based errors in meters.
    
    Parameters:
    -----------
    y_test : np.ndarray
        Ground truth positions, shape (n, 2) with [lat, lon]
    y_pred : np.ndarray
        Predicted positions, shape (n, 2) with [lat, lon]
        
    Returns:
    --------
    dict with keys:
        - 'mean_error_m': Mean error in meters
        - 'median_error_m': Median error in meters
        - 'max_error_m': Maximum error in meters
        - 'std_error_m': Standard deviation of errors
    """
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    # Convert to meters
    lat_avg = np.mean(y_test[:, 0])
    meters_per_degree_lat = 111320
    meters_per_degree_lon = 111320 * np.cos(np.radians(lat_avg))
    
    lat_diff = (y_pred[:, 0] - y_test[:, 0]) * meters_per_degree_lat
    lon_diff = (y_pred[:, 1] - y_test[:, 1]) * meters_per_degree_lon
    
    distances = np.sqrt(lat_diff**2 + lon_diff**2)
    
    return {
        'mean_error_m': np.mean(distances),
        'median_error_m': np.median(distances),
        'max_error_m': np.max(distances),
        'std_error_m': np.std(distances)
    }
    