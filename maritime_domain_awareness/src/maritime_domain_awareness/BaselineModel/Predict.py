import numpy as np

def predict_future_position(x_estimate, A, B, u, steps=10):
    """
    Predict future positions using the final Kalman state.
    
    Parameters:
    -----------
    x_estimate : np.ndarray
        Current state estimate [latitude, longitude, velocity_lat, velocity_lon]
        Shape: (4, 1)
    A : np.ndarray
        State transition matrix (4x4)
    B : np.ndarray
        Control input matrix (4x2)
    u : np.ndarray
        Control input vector (2x1)
    steps : int
        Number of future time steps to predict
        
    Returns:
    --------
    predictions : list of tuples
        List of (latitude, longitude) predictions for future time steps
    """
    predictions = []
    x_pred = x_estimate.copy()
    
    for _ in range(steps):
        # Predict next state using motion model
        x_pred = A @ x_pred + B @ u
        # Store predicted position (lat, lon)
        predictions.append((x_pred[0, 0], x_pred[1, 0]))
    
    return predictions


def calculate_prediction_metrics(predictions, ground_truth):
    """
    Calculate prediction error metrics.
    
    Parameters:
    -----------
    predictions : list or np.ndarray
        Predicted positions [(lat1, lon1), (lat2, lon2), ...]
    ground_truth : list or np.ndarray
        True positions [(lat1, lon1), (lat2, lon2), ...]
        
    Returns:
    --------
    metrics : dict
        Dictionary containing:
        - 'mae': Mean Absolute Error
        - 'rmse': Root Mean Squared Error
        - 'max_error': Maximum error
    """
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Calculate errors (Euclidean distance)
    errors = np.linalg.norm(predictions - ground_truth, axis=1)
    
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(errors)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'mean_lat_error': np.mean(np.abs(predictions[:, 0] - ground_truth[:, 0])),
        'mean_lon_error': np.mean(np.abs(predictions[:, 1] - ground_truth[:, 1]))
    }
    
    return metrics