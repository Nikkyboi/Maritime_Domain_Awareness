import numpy as np
from typing import Optional


class KalmanFilter:
    """
    A simple 4-state constant-acceleration Kalman filter for 2D tracking.

    State vector: [x, y, v_e, v_n]^T (meters, meters, m/s east, m/s north)
    Measurements: [x, y]^T in meters (projected coordinates)
    """

    def __init__(
        self,
        df,
        dt: float = 1.0,
        process_variance: float = 1e-5,
        measurement_variance: float = 0.1,
        init_error: float = 1.0,
    ):
        # store basic params
        self.dt = float(dt)
        self.process_variance = float(process_variance)
        self.measurement_variance = float(measurement_variance)
        self.init_error = float(init_error)

        self.df = df.copy()

        # initialize Kalman matrices
        self._init_matrices()

    def _init_matrices(self):
        """
        Initialize the Kalman filter matrices.
        4D state: [x, y, v_e, v_n]^T
        2D measurement: [x, y]^T
        """
        dt = self.dt
        # State transition
        self.A = np.array(
            [[1, 0, dt, 0], 
            [0, 1, 0, dt], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]], 
            dtype=float
        )

        # Control-input model (acceleration -> position/velocity)
        self.B = np.array(
            [[0.5 * dt ** 2, 0.0], 
            [0.0, 0.5 * dt ** 2], 
            [dt, 0.0], [0.0, dt]], 
            dtype=float
        )

        # measurement matrix: we measure position only
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)

        # Covariances
        self.Q = self.process_variance * np.eye(4)
        self.R = self.measurement_variance * np.eye(2)

        # initial estimate covariance
        self.P = self.init_error * np.eye(4)

        # initial control (no acceleration)
        self.u = np.zeros((2, 1), dtype=float)

        # placeholders set by predict/update
        self.x_est = None
        self.x_pred = None
        self.P_pred = None
        self.K = None


    def predict(self):
        """Run the prediction step."""

        self.x_pred = self.A @ self.x_est + self.B @ self.u
        self.P_pred = self.A @ self.P @ self.A.T + self.Q
        return self.x_pred, self.P_pred

    def _compute_gain(self):
        S = self.H @ self.P_pred @ self.H.T + self.R
        K = self.P_pred @ self.H.T @ np.linalg.inv(S)
        self.K = K
        return K

    def update(self, z):
        """Update the filter with measurement z (length-2 iterable or 2x1 array in meters)."""
        z = np.asarray(z, dtype=float).reshape(2, 1)

        # residual
        y = z - (self.H @ self.x_pred)

        # gain
        K = self._compute_gain()

        # update estimate
        self.x_est = self.x_pred + K @ y
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P_pred

        return self.x_est, self.P

    def step(self, z):
        """Convenience: predict then update with measurement z."""
        self.predict()
        return self.update(z)



