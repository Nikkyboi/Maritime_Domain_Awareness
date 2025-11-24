import pandas as pd
import glob
from pathlib import Path
import numpy as np


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
    # Make a local copy to avoid mutating caller's DataFrame
    df = df_test.copy()

    # Helper: find columns by lowercase name mapping
    cols_map = {c.lower(): c for c in df.columns}

    if 'latitude' in cols_map and 'longitude' in cols_map:
        lat_col = cols_map['latitude']
        lon_col = cols_map['longitude']
        return df[[lat_col, lon_col]].values[:prediction_steps]

    # Fallback: reconstruct from unit-sphere coordinates X, Y, Z
    if all(k in cols_map for k in ('x', 'y', 'z')):
        x = df[cols_map['x']].to_numpy(dtype=float)
        y = df[cols_map['y']].to_numpy(dtype=float)
        z = df[cols_map['z']].to_numpy(dtype=float)

        # Protect against out-of-range due to numerical noise
        z = np.clip(z, -1.0, 1.0)

        lat_rad = np.arcsin(z)
        lon_rad = np.arctan2(y, x)

        lat_deg = np.degrees(lat_rad)
        lon_deg = np.degrees(lon_rad)

        gt = np.column_stack([lat_deg, lon_deg])
        return gt[:prediction_steps]

    # Nothing we can do: raise clear error for the user
    raise KeyError(
        "DataFrame is missing 'Latitude'/'Longitude' columns and cannot reconstruct them. "
        "Expected either 'Latitude' and 'Longitude', or 'X','Y','Z' unit-sphere columns. "
        f"Available columns: {list(df.columns)}"
    )



