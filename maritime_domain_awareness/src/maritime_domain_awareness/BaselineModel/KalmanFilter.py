import numpy as np
from pyproj import Transformer

def KALMAN_Filter(df):
    """
    Apply a simple 2D Kalman filter to ship GPS data (Latitude, Longitude).
    Assumes constant velocity motion model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'Latitude', 'Longitude', and optionally 'Timestamp' columns
        
    Returns:
    --------
    estimates : list of tuples
        Filtered (lat, lon) positions
    x_estimate : np.ndarray
        Final state estimate [lat, lon, vel_lat, vel_lon]
    A, B, u : np.ndarray
        State transition matrices for future predictions
    """
    
    # Calculate actual time step from data
    if 'Timestamp' in df.columns:
        time_diffs = df['Timestamp'].diff().dt.total_seconds()
        dt = time_diffs.median()  # Use median time step
        if dt is None or dt <= 0:
            dt = 1.0  # Fallback
    else:
        dt = 1.0  # Default if no timestamp

    # --- Hyperparameters ---
    process_variance = 1e-5  # process noise variance (Q)
    measurement_variance = 0.1  # measurement noise variance (R)
    initial_estimate_error = 1.0  # initial covariance

    # Reconstruct lat/lon if they were removed earlier (some preprocessing drops them)
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        # Expect X,Y,Z unit vectors are present
        if {'X', 'Y', 'Z'}.issubset(set(df.columns)):
            lat_rad = np.arcsin(np.clip(df['Z'].to_numpy(dtype=float), -1.0, 1.0))
            lon_rad = np.arctan2(df['Y'].to_numpy(dtype=float), df['X'].to_numpy(dtype=float))
            df = df.copy()
            df['Latitude'] = np.degrees(lat_rad)
            df['Longitude'] = np.degrees(lon_rad)
        else:
            raise KeyError("DataFrame must contain Latitude/Longitude or X,Y,Z to run Kalman filter.")

    # initial position (lat/lon)
    initial_lat = float(df['Latitude'].iloc[0])
    initial_lon = float(df['Longitude'].iloc[0])

    # Initial velocity: use SOG (already in m/s in processed data) and COG to get east/north components
    # Take median of first few samples for robustness
    k = min(5, len(df))
    sog_m_s = float(np.median(df['SOG'].iloc[:k].to_numpy()))
    cog_deg = float(np.median(df['COG'].iloc[:k].to_numpy()))
    cog_rad = np.deg2rad(cog_deg)
    v_n = sog_m_s * np.cos(cog_rad)  # north (m/s)
    v_e = sog_m_s * np.sin(cog_rad)  # east  (m/s)

    # Transformer: lon,lat -> meters (use WebMercator / EPSG:3857 for simplicity)
    transformer_to_m = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    transformer_to_deg = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    # initial position in meters
    x0_m, y0_m = transformer_to_m.transform(initial_lon, initial_lat)
    # --- Matrices (meters) ---
    A_m = np.array([[1, 0, dt, 0],
                    [0, 1, 0, dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    B_m = np.array([[0.5 * dt**2, 0],
                    [0, 0.5 * dt**2],
                    [dt, 0],
                    [0, dt]])

    u_m = np.array([[0.0], [0.0]])

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

    w = process_variance * np.eye(4)
    v = measurement_variance * np.eye(2)
    P = np.eye(4) * initial_estimate_error

    # --- Initial state in meters ---
    x_est_m = np.array([[x0_m], [y0_m], [v_e], [v_n]])

    estimates_m = []

    # Precompute all measurements in meters to avoid per-row transformer calls
    lons = df['Longitude'].to_numpy(dtype=float)
    lats = df['Latitude'].to_numpy(dtype=float)
    xs, ys = transformer_to_m.transform(lons, lats)

    # --- Main filter loop (in meters) ---
    for xi, yi in zip(xs, ys):
        z_measure = np.array([[xi], [yi]])

        # Prediction
        x_pred = A_m @ x_est_m + B_m @ u_m
        P_pred = A_m @ P @ A_m.T + w

        # Kalman Gain
        S = H @ P_pred @ H.T + v
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Update
        y = z_measure - (H @ x_pred)
        x_est_m = x_pred + K @ y
        P = (np.eye(4) - K @ H) @ P_pred

        estimates_m.append((float(x_est_m[0, 0]), float(x_est_m[1, 0])))

    # Convert meter estimates back to lat/lon for compatibility with the rest of the code
    est_xs = np.array([p[0] for p in estimates_m])
    est_ys = np.array([p[1] for p in estimates_m])
    est_lons, est_lats = transformer_to_deg.transform(est_xs, est_ys)
    estimates_deg = list(zip(est_lats.tolist(), est_lons.tolist()))

    # Convert final state to degrees + degrees/sec velocities for returning
    final_x_m = float(x_est_m[0, 0])
    final_y_m = float(x_est_m[1, 0])
    final_vx_m = float(x_est_m[2, 0])  # east m/s
    final_vy_m = float(x_est_m[3, 0])  # north m/s

    final_lon, final_lat = transformer_to_deg.transform(final_x_m, final_y_m)

    # meters per degree at final latitude
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * np.cos(np.deg2rad(final_lat))

    vel_lat_deg_s = final_vy_m / meters_per_deg_lat
    vel_lon_deg_s = final_vx_m / meters_per_deg_lon if meters_per_deg_lon != 0 else 0.0

    # Build degree-space state and matrices for prediction compatibility
    x_est_deg = np.array([[final_lat], [final_lon], [vel_lat_deg_s], [vel_lon_deg_s]])

    # A/B matrices in degree units (positions in degrees, velocities in deg/sec)
    A_deg = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    B_deg = np.array([[0.5 * dt**2, 0],
                      [0, 0.5 * dt**2],
                      [dt, 0],
                      [0, dt]])

    u_deg = np.array([[0.0], [0.0]])

    print(f"Kalman Filter applied to {len(df)} data points (metric internal).")
    return estimates_deg, x_est_deg, A_deg, B_deg, u_deg

