import numpy as np
from .models.KalmanFilter import KalmanFilter

def kalman_trajectory_prediction(
    X_seq_raw,
    seq_len: int = 50,
    future_steps: int = 50,
    dt: float = 60.0,  # 1 minute in seconds
    haversine = 0
):
    """
    Perform Kalman filter prediction for trajectory forecasting.
    Similar interface to trajectory_prediction but uses Kalman filter.
    
    Args:
        X_seq_raw: torch tensor [N, 4] with [Lat, Lon, SOG, COG] in raw units
        seq_len: context window (not heavily used by Kalman, but kept for consistency)
        future_steps: how many steps to predict ahead
        dt: time step in seconds (default 60s = 1 minute)
    
    Returns:
        dict with predicted lat/lon arrays
    """
    X_np_raw = X_seq_raw.cpu().numpy()
    N = X_np_raw.shape[0]

    if N < seq_len + future_steps:
        raise ValueError(f"Not enough points in sequence (N={N})")

    # Earth radius for lat/lon to meters conversion (approximate)
    R_earth = 6371000  # meters
    lat_to_m = R_earth * np.pi / 180.0

    # Indices
    first_future_idx = N - future_steps
    base_idx = first_future_idx - 1

    # Convert lat/lon to meters (approximate local projection)
    # Use mean latitude for longitude conversion
    mean_lat = np.mean(X_np_raw[:first_future_idx, 0])
    lon_to_m = lat_to_m * np.cos(np.radians(mean_lat))

    # Initialize Kalman filter with first observation
    lat0, lon0 = X_np_raw[0, 0], X_np_raw[0, 1]
    x0_m = lon0 * lon_to_m  # east in meters
    y0_m = lat0 * lat_to_m  # north in meters

    df_init = {'x': x0_m, 'y': y0_m, 'v_e': 0.0, 'v_n': 0.0}
    kf = KalmanFilter(
        df=df_init,
        dt=dt,
        process_variance=1e-3,  # tune this for smoothness
        measurement_variance=10.0,  # tune based on GPS accuracy
        init_error=100.0
    )
    kf.x_est = np.array([[x0_m], [y0_m], [0.0], [0.0]])

    # Run filter through all historical data up to prediction point
    for t in range(first_future_idx):
        lat, lon = X_np_raw[t, 0], X_np_raw[t, 1]
        x_m = lon * lon_to_m
        y_m = lat * lat_to_m
        z = np.array([x_m, y_m])
        kf.step(z)

    # Now predict future without measurements (dead reckoning)
    pred_lat_future = []
    pred_lon_future = []

    for step in range(future_steps):
        # Predict next state
        kf.predict()
        # Use prediction as next estimate (no measurement update)
        kf.x_est = kf.x_pred.copy()
        kf.P = kf.P_pred.copy()

        # Convert back to lat/lon
        x_m, y_m = kf.x_est[0, 0], kf.x_est[1, 0]
        lon_pred = x_m / lon_to_m
        lat_pred = y_m / lat_to_m

        pred_lat_future.append(lat_pred)
        pred_lon_future.append(lon_pred)

    # Get base point and true future
    base_lat = X_np_raw[base_idx, 0]
    base_lon = X_np_raw[base_idx, 1]

    true_lat_future = X_np_raw[first_future_idx:first_future_idx + future_steps, 0]
    true_lon_future = X_np_raw[first_future_idx:first_future_idx + future_steps, 1]

    # Prepend base point
    pred_lat = np.concatenate([[base_lat], pred_lat_future])
    pred_lon = np.concatenate([[base_lon], pred_lon_future])
    true_lat = np.concatenate([[base_lat], true_lat_future])
    true_lon = np.concatenate([[base_lon], true_lon_future])

    # Compute error
    error_m = haversine(true_lat[-1], true_lon[-1], pred_lat[-1], pred_lon[-1])

    return {
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "true_lat": true_lat,
        "true_lon": true_lon,
        "error_m": error_m,
    }