import pandas as pd
import pyarrow
import pyarrow.parquet
import numpy as np
from pathlib import Path
import os
import glob

def interpolate(group, freq="1min"):
    #vi sætter index til timestamp i stedet for 0,1,2...
    group = group.set_index("Timestamp")
    #man skal være på passelig med cog da det er i grader og gennemsnittet af 350 og 10 er 180 men det er det ikke det er 0
    rads = np.deg2rad(group["COG"])
    group["COG_sin"] = np.sin(rads)
    group["COG_cos"] = np.cos(rads)
    group = group.drop(columns=["COG"])
    #vi resampler så får en observation hvert freq med gennemsnit, eller nan.

    resampled = group.resample(freq).mean(numeric_only=True)
    # vi interpolerer kolonnerne for at udfylde nan værdier
    cols_to_interp = ["Latitude", "Longitude", "SOG","COG_sin", "COG_cos"]
    resampled[cols_to_interp] = resampled[cols_to_interp].interpolate(method="linear", limit_direction='both')
    resampled["COG"] = np.rad2deg(np.arctan2(resampled["COG_sin"], resampled["COG_cos"]))
    resampled["COG"] = (resampled["COG"] + 360) % 360
    resampled = resampled.drop(columns=["COG_sin", "COG_cos"])
    resampled["MMSI"] = group["MMSI"].iloc[0]
    resampled["Segment"] = group["Segment"].iloc[0]
    return resampled.dropna().reset_index()

def haversine_distance(lat1, lon1, lat2, lon2):
    #https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def filter_anomalies(df):
    df = df.sort_values(['MMSI', 'Timestamp'])
    prev_lat = df.groupby('MMSI')['Latitude'].shift(1)
    prev_lon = df.groupby('MMSI')['Longitude'].shift(1)
    prev_time = df.groupby('MMSI')['Timestamp'].shift(1)
    dist_meters = haversine_distance(df['Latitude'], df['Longitude'], prev_lat, prev_lon)
    time_diff = (df['Timestamp'] - prev_time).dt.total_seconds()
    calc_speed = dist_meters / time_diff.replace(0, np.nan)

    MAX_SPEED_kMS = 0.05 # 50 meter per second

    return df[(calc_speed < MAX_SPEED_kMS) | calc_speed.isna()]


def filter_raw_data(df):
    # Remove errors 
    #måske skal vi slette denne 
    bbox = [60, 0, 50, 20]
    north, west, south, east = bbox
    df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & (df["Longitude"] >= west) & (
    df["Longitude"] <= east)]

    df = df[df["Type of mobile"].isin(["Class A", "Class B"])].drop(columns=["Type of mobile"])
    df = df[df["MMSI"].str.len() == 9] # Adhere to MMSI format
    
    # Handle potential non-numeric MMSI prefixes safely
    # df = df[df["MMSI"].str[:3].astype(int).between(219, 220)] 
    # Safer alternative if needed, but sticking to original logic for now:
    try:
        df = df[df["MMSI"].str[:3].astype(int).between(219, 220)]
    except ValueError:
        # Fallback if conversion fails (e.g. empty strings or non-digits)
        pass

    df["Ship type"] = df["Ship type"].fillna("Unknown").astype(str).str.lower()
    df["Navigational status"] = df["Navigational status"].fillna("Unknown").astype(str).str.lower()

    is_fishing = (
    df["Ship type"].str.contains("fishing") |
    df["Navigational status"].str.contains("fishing")
    )
    df = df[is_fishing]
    return df


def remove_long_stationary_periods(df, max_duration_hours=4, speed_threshold=1.0):
    print(f"Filtering stationary periods > {max_duration_hours} hours with speed < {speed_threshold} knots...")
    # df is already sorted by MMSI, Timestamp before this call
    # df = df.sort_values(['MMSI', 'Timestamp']) 
    
    is_stationary = df['SOG'] < speed_threshold
    
    # Identify groups of consecutive stationary/moving points
    condition_change = (is_stationary != is_stationary.shift())
    mmsi_change = (df['MMSI'] != df['MMSI'].shift())
    group_ids = (condition_change | mmsi_change).cumsum()
    
    # Calculate duration for each group
    group_stats = df.groupby(group_ids).agg(
        is_stat=('SOG', lambda x: (x < speed_threshold).all()),
        duration=('Timestamp', lambda x: x.max() - x.min())
    )
    
    # Identify groups to drop
    groups_to_drop = group_stats[
        group_stats['is_stat'] & 
        (group_stats['duration'] > pd.Timedelta(hours=max_duration_hours))
    ].index
    
    # Filter
    df['temp_group_id'] = group_ids
    df_filtered = df[~df['temp_group_id'].isin(groups_to_drop)].drop(columns=['temp_group_id'])
    
    print(f"Removed {len(df) - len(df_filtered)} rows (stationary periods).")
    return df_filtered


def preprocess(input_path, out_path, raw_out_path=None, test_limit=None, max_days=None):
    dtypes = {
    "MMSI": "object",
    "SOG": float,
    "COG": float,
    "Longitude": float,
    "Latitude": float,
    "# Timestamp": "object",
    "Type of mobile": "object",
    "Navigational status": "object",
    "Ship type": "object",
    }
    usecols = list(dtypes.keys())

    if os.path.isdir(input_path):
        all_files = sorted(glob.glob(os.path.join(input_path, "*.csv")))
        if max_days:
            all_files = all_files[:max_days]
            print(f"Limiting to first {max_days} days (files): {all_files}")
            
        df_list = []
        for f in all_files:
            try:
                # Check if file has the required columns by reading just the header
                header = pd.read_csv(f, nrows=0)
                if not set(usecols).issubset(header.columns):
                    print(f"Skipping {f}: Missing required columns")
                    continue
                
                print(f"Reading {f}...")
                df_chunk = pd.read_csv(f, usecols=usecols, dtype=dtypes)
                
                # Apply filtering immediately to reduce memory usage
                df_chunk = filter_raw_data(df_chunk)
                
                if not df_chunk.empty:
                    df_list.append(df_chunk)
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
        if not df_list:
            print("No valid CSV files found.")
            return
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.read_csv(input_path, usecols=usecols, dtype=dtypes)
        df = filter_raw_data(df)

    if test_limit:
        unique_mmsis = df["MMSI"].unique()[:test_limit]
        df = df[df["MMSI"].isin(unique_mmsis)]
        print(f"Test mode: Limiting to {test_limit} MMSIs: {unique_mmsis}")

    df = df.rename(columns={"# Timestamp": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")

    df = df.drop_duplicates(["Timestamp", "MMSI", ], keep="first")
    df = df.sort_values(['MMSI', 'Timestamp'])

    df = remove_long_stationary_periods(df, max_duration_hours=4, speed_threshold=1.0)

    # Divide track into segments based on timegap
    df['Segment'] = df.groupby('MMSI')['Timestamp'].transform(
    lambda x: (x.diff().dt.total_seconds().fillna(0) >= 15 * 60).cumsum()) # Max allowed timegap

    if raw_out_path:
        print(f"Saving raw segmented data to {raw_out_path}...")
        table_raw = pyarrow.Table.from_pandas(df, preserve_index=False)
        pyarrow.parquet.write_to_dataset(
            table_raw,
            root_path=raw_out_path,
            partition_cols=["MMSI", "Segment"]
        )

    df = filter_anomalies(df)
    def track_filter(g):
        # Filter out tracks that are too short, stationary, or span too little time
        len_filt = len(g) > 256 # Min required length of track/segment
        sog_filt = 1 <= g["SOG"].max() <= 50 # Remove stationary tracks/segments
        time_filt = (g["Timestamp"].max() - g["Timestamp"].min()).total_seconds() >= 60 * 60 # Min required timespan
        return len_filt and sog_filt and time_filt

    # Track filtering
    df = df.groupby("MMSI").filter(track_filter)
    df = df.sort_values(['MMSI', 'Timestamp'])

    # Segment filtering
    df = df.groupby(["MMSI", "Segment"]).filter(track_filter)
    df = df.reset_index(drop=True)

    #
    knots_to_ms = 0.514444
    df["SOG"] = knots_to_ms * df["SOG"]

    df_processed = df.groupby(["MMSI", "Segment"]).apply(interpolate).reset_index(drop=True)

    table = pyarrow.Table.from_pandas(df_processed, preserve_index=False)
    pyarrow.parquet.write_to_dataset(
    table,
    root_path=out_path,
    partition_cols=["MMSI", # "Date",
    "Segment", # "Geocell"
    ]
    )

    print(f"saving {df_processed['MMSI'].nunique()} unique MMSI to {out_path}")

    # Summary statistics
    segments_per_ship = df_processed.groupby("MMSI")["Segment"].nunique()
    avg_segments = segments_per_ship.mean()

    segment_lengths = df_processed.groupby(["MMSI", "Segment"]).size()
    avg_length = segment_lengths.mean()

    # Calculate duration in hours
    segment_durations = df_processed.groupby(["MMSI", "Segment"])["Timestamp"].apply(
        lambda x: (x.max() - x.min()).total_seconds() / 3600
    )
    avg_duration = segment_durations.mean()

    print("\n=== Summary Statistics ===")
    print(f"Average segments per ship: {avg_segments:.2f}")
    print(f"Average segment length (data points): {avg_length:.2f}")
    print(f"Average segment duration (hours): {avg_duration:.2f}")
    print("==========================")


if __name__ == "__main__":
    preprocess(
        "../../data/Raw/data", 
        "../../data/preprocessed/done4.parquet",
        "../../data/preprocessed/raw_segmented.parquet",
)