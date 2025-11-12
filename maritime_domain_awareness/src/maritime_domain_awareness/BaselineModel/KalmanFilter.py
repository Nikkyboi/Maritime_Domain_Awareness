import numpy as np

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
    initial_velocity = 0.0  # start at rest
    initial_position = (df['Latitude'].iloc[0], df['Longitude'].iloc[0])

    # --- Matrices ---
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    B = np.array([[0.5 * dt**2, 0],
                  [0, 0.5 * dt**2],
                  [dt, 0],
                  [0, dt]])

    u = np.array([[0],
                  [0]])

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

    w = process_variance * np.eye(4)
    v = measurement_variance * np.eye(2)
    P = np.eye(4) * initial_estimate_error

    # --- Initial state ---
    x_estimate = np.array([[initial_position[0]],
                           [initial_position[1]],
                           [initial_velocity],
                           [initial_velocity]])

    estimates = []

    # --- Main filter loop ---
    for _, row in df.iterrows():
        # Measurement
        z_measure = np.array([[row['Latitude']],
                              [row['Longitude']]])

        # Prediction
        x_pred = A @ x_estimate + B @ u
        P_pred = A @ P @ A.T + w

        # Kalman Gain
        S = H @ P_pred @ H.T + v
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Update
        y = z_measure - (H @ x_pred)
        x_estimate = x_pred + K @ y
        P = (np.eye(4) - K @ H) @ P_pred

        estimates.append((x_estimate[0, 0], x_estimate[1, 0]))

    print(f"Kalman Filter applied to {len(df)} data points.")
    return estimates, x_estimate, A, B, u

