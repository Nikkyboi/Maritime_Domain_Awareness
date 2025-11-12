import pandas as pd
import glob
from pathlib import Path


def load_parquet_files(data_path):
    """
    Load parquet files from a given path pattern.
    
    Parameters:
    -----------
    data_path : str
        Path pattern (can include wildcards like *.parquet)
        
    Returns:
    --------
    files : list
        List of file paths that match the pattern
    """
    files = glob.glob(data_path)
    
    if not files:
        print(f"No files found matching: {data_path}")
        return []
    
    print(f"Found {len(files)} file(s)")
    return files


def load_and_prepare_data(file_path):
    """
    Load a parquet file and prepare it for processing.
    
    Parameters:
    -----------
    file_path : str
        Path to the parquet file
        
    Returns:
    --------
    df : pd.DataFrame
        Loaded and sorted DataFrame
    """
    df = pd.read_parquet(file_path)
    
    # Sort by timestamp if available
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp')
    
    return df


def split_train_test(df, train_split=0.8):
    """
    Split DataFrame into training and testing portions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with Timestamp, Latitude, Longitude columns
    train_split : float
        Fraction of data to use for training (0.0 to 1.0)
        
    Returns:
    --------
    df_train : pd.DataFrame
        Training portion
    df_test : pd.DataFrame
        Testing portion
    """
    split_idx = int(len(df) * train_split)
    
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    return df_train, df_test


def calculate_prediction_steps(df, prediction_minutes):
    """
    Calculate how many time steps correspond to a given prediction horizon.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Timestamp column
    prediction_minutes : int
        Desired prediction horizon in minutes
        
    Returns:
    --------
    prediction_steps : int
        Number of time steps for the prediction horizon
    avg_interval : float
        Average time interval between data points in seconds
    """
    if 'Timestamp' not in df.columns:
        # Default: assume 5 second intervals
        avg_interval = 5.0
    else:
        time_diffs = df['Timestamp'].diff().dt.total_seconds()
        avg_interval = time_diffs.median()
        
        if avg_interval is None or avg_interval <= 0:
            avg_interval = 5.0  # Fallback
    
    prediction_steps = int((prediction_minutes * 60) / avg_interval)
    
    return prediction_steps, avg_interval


def validate_data_size(df_train, df_test, prediction_steps, min_train_size=100):
    """
    Check if we have enough data for meaningful evaluation.
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Training data
    df_test : pd.DataFrame
        Testing data
    prediction_steps : int
        Number of prediction steps needed
    min_train_size : int
        Minimum training samples required
        
    Returns:
    --------
    is_valid : bool
        True if data is sufficient, False otherwise
    message : str
        Reason if data is insufficient
    """
    if len(df_train) < min_train_size:
        return False, f"Training data too small: {len(df_train)} < {min_train_size}"
    
    if len(df_test) < prediction_steps:
        return False, f"Test data too small: {len(df_test)} < {prediction_steps}"
    
    return True, "Data size is sufficient"


def extract_ground_truth(df_test, prediction_steps):
    """
    Extract ground truth positions from test data.
    
    Parameters:
    -----------
    df_test : pd.DataFrame
        Test DataFrame with Latitude and Longitude columns
    prediction_steps : int
        Number of prediction steps
        
    Returns:
    --------
    ground_truth : np.ndarray
        Array of shape (prediction_steps, 2) with [lat, lon]
    """
    return df_test[['Latitude', 'Longitude']].values[:prediction_steps]



