import numpy as np
from pathlib import Path
from ProcessData import (
    load_parquet_files,
    load_and_prepare_data,
    split_train_test,
    calculate_prediction_steps,
    validate_data_size,
    extract_ground_truth
)
from KalmanFilter import KALMAN_Filter
from Predict import predict_future_position, calculate_prediction_metrics
from validate import validate_baselineModel, calculate_distance_errors


def evaluate_single_file(file_path, prediction_minutes=30, train_split=0.8):
    """
    Evaluate predictions for a single file.
    
    Parameters:
    -----------
    file_path : str
        Path to parquet file
    prediction_minutes : int
        Prediction horizon in minutes
    train_split : float
        Train/test split ratio
        
    Returns:
    --------
    predictions : list
        Predicted positions
    ground_truth : np.ndarray
        Actual future positions
    time_span : float
        Actual prediction time span in minutes
    prediction_steps : int
        Number of prediction steps
    """
    # Load and prepare data
    df = load_and_prepare_data(file_path)
    
    # Calculate prediction steps
    prediction_steps, avg_interval = calculate_prediction_steps(df, prediction_minutes)
    
    # Split data
    df_train, df_test = split_train_test(df, train_split)
    
    # Validate data size
    is_valid, message = validate_data_size(df_train, df_test, prediction_steps)
    
    if not is_valid:
        return None, None, None, None, message
    
    # Apply Kalman filter on training data
    estimates, x_estimate, A, B, u = KALMAN_Filter(df_train)
    
    # Predict future positions
    predictions = predict_future_position(x_estimate, A, B, u, steps=prediction_steps)
    
    # Extract ground truth from test data
    ground_truth = extract_ground_truth(df_test, prediction_steps)
    
    # Calculate actual time span
    if 'Timestamp' in df_test.columns:
        time_span = (df_test['Timestamp'].iloc[prediction_steps-1] - df_test['Timestamp'].iloc[0]).total_seconds() / 60
    else:
        time_span = prediction_minutes
    
    return predictions, ground_truth, time_span, prediction_steps, "Success"


def evaluate_predictions(data_path, prediction_minutes=30, train_split=0.8):
    """
    Evaluate Kalman filter predictions against ground truth for multiple files.
    
    Parameters:
    -----------
    data_path : str
        Path to parquet files (can use wildcards like *.parquet)
    prediction_minutes : int
        How many minutes ahead to predict
    train_split : float
        Fraction of data to use for filtering (rest is ground truth)
        
    Returns:
    --------
    results : dict
        Dictionary with predictions, ground_truth, and metrics
    """
    # Load files
    files = load_parquet_files(data_path)
    
    if not files:
        return None
    
    all_predictions = []
    all_ground_truth = []
    
    print(f"\nEvaluating predictions for {len(files)} file(s)...\n")
    
    # Process each file
    for file_path in files:
        predictions, ground_truth, time_span, pred_steps, status = evaluate_single_file(
            file_path, prediction_minutes, train_split
        )
        
        if predictions is None:
            print(f"  ⚠️  Skipping {Path(file_path).name}: {status}")
            continue
        
        all_predictions.extend(predictions)
        all_ground_truth.extend(ground_truth)
        
        print(f"  ✓ {Path(file_path).name}: {time_span:.1f} min ({pred_steps} steps)")
    
    # Check if we have any predictions
    if not all_predictions:
        print("\n⚠️  No predictions to evaluate!")
        return None
    
    # Convert to numpy arrays
    y_pred = np.array(all_predictions)
    y_test = np.array(all_ground_truth)
    
    # Calculate all metrics
    degree_metrics = calculate_prediction_metrics(y_pred, y_test)
    distance_metrics = calculate_distance_errors(y_test, y_pred)
    
    # Calculate accuracy at different thresholds
    accuracy_metrics = {
        'acc_50m': validate_baselineModel(y_test, y_pred, threshold_meters=50),
        'acc_100m': validate_baselineModel(y_test, y_pred, threshold_meters=100),
        'acc_500m': validate_baselineModel(y_test, y_pred, threshold_meters=500)
    }
    
    return {
        'predictions': y_pred,
        'ground_truth': y_test,
        'degree_metrics': degree_metrics,
        'distance_metrics': distance_metrics,
        'accuracy_metrics': accuracy_metrics,
        'num_files': len(files),
        'num_predictions': len(all_predictions)
    }


def print_evaluation_results(results):
    """
    Print evaluation results in a formatted way.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from evaluate_predictions()
    """
    if results is None:
        print("No results to display.")
        return
    
    degree = results['degree_metrics']
    distance = results['distance_metrics']
    accuracy = results['accuracy_metrics']
    
    print(f"\n{'='*60}")
    print(f"PREDICTION EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Files evaluated: {results['num_files']}")
    print(f"Total predictions: {results['num_predictions']}")
    
    print(f"\n--- Degree-based errors ---")
    print(f"  MAE: {degree['mae']:.6f}°")
    print(f"  RMSE: {degree['rmse']:.6f}°")
    print(f"  Max Error: {degree['max_error']:.6f}°")
    
    print(f"\n--- Distance-based errors (meters) ---")
    print(f"  Mean Error: {distance['mean_error_m']:.1f} m")
    print(f"  Median Error: {distance['median_error_m']:.1f} m")
    print(f"  Max Error: {distance['max_error_m']:.1f} m")
    print(f"  Std Deviation: {distance['std_error_m']:.1f} m")
    
    print(f"\n--- Accuracy (% within threshold) ---")
    print(f"  Within 50m: {accuracy['acc_50m']*100:.1f}%")
    print(f"  Within 100m: {accuracy['acc_100m']*100:.1f}%")
    print(f"  Within 500m: {accuracy['acc_500m']*100:.1f}%")
    print(f"{'='*60}\n")
