import pandas as pd
import pyarrow
import pyarrow.parquet
import numpy as np
from pathlib import Path
import os

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


def preprocess(file_path, out_path):
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
    df = pd.read_csv(file_path, usecols=usecols, dtype=dtypes)

    # Remove errors
    bbox = [60, 0, 50, 20]
    north, west, south, east = bbox
    df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & (df["Longitude"] >= west) & (
    df["Longitude"] <= east)]

    df = df[df["Type of mobile"].isin(["Class A", "Class B"])].drop(columns=["Type of mobile"])
    df = df[df["MMSI"].str.len() == 9] # Adhere to MMSI format
    df = df[df["MMSI"].str[:3].astype(int).between(219, 220)] # Adhere to MID standard

    df["Ship type"] = df["Ship type"].fillna("Unknown").astype(str).str.lower()
    df["Navigational status"] = df["Navigational status"].fillna("Unknown").astype(str).str.lower()

    is_fishing = (
    df["Ship type"].str.contains("fishing") |
    df["Navigational status"].str.contains("fishing")
    )
    df = df[is_fishing]

    df = df.rename(columns={"# Timestamp": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")

    df = df.drop_duplicates(["Timestamp", "MMSI", ], keep="first")
    df = df.sort_values(['MMSI', 'Timestamp'])

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

    # Divide track into segments based on timegap
    df['Segment'] = df.groupby('MMSI')['Timestamp'].transform(
    lambda x: (x.diff().dt.total_seconds().fillna(0) >= 15 * 60).cumsum()) # Max allowed timegap

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

    print(f"saving {df['MMSI'].nunique()} unique MMSI to {out_path}")


preprocess("maritime_domain_awareness/data/Raw/aisdk-2025-03-01.csv", "maritime_domain_awareness/data/preprocessed/done3.parquet")