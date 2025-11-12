from pathlib import Path

import typer
from torch.utils.data import Dataset
import pandas
import pyarrow
import pyarrow.parquet
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import torch


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

        

def fn(file_path, out_path):
    # ---- read ----
    dtypes = {
        "MMSI": "object",
        "SOG": float,
        "COG": float,
        "Longitude": float,
        "Latitude": float,
        "# Timestamp": "object",
        "Type of mobile": "object",
        # fishing-related columns
        "Navigational status": "object",
        "Ship type": "object",
    }
    usecols = list(dtypes.keys())
    # read only columns that exist in the file
    df = pandas.read_csv(file_path, usecols=usecols, dtype=dtypes)
    
    # ---- base geographic and quality filters ----
    bbox = [60, 0, 50, 20]
    north, west, south, east = bbox
    df = df[
        (df["Latitude"] <= north) & (df["Latitude"] >= south) &
        (df["Longitude"] >= west) & (df["Longitude"] <= east)
    ]

    # keep only AIS Class A/B
    df = df[df["Type of mobile"].isin(["Class A", "Class B"])]
    df = df.drop(columns=["Type of mobile"])

    # MMSI sanity + Danish MMSIs (219*, 220*)
    df = df[df["MMSI"].astype(str).str.len() == 9]
    df = df[df["MMSI"].astype(str).str.isnumeric()]
    df = df[df["MMSI"].astype(str).str.startswith(("219", "220"))]

    # ---- fishing vessel filters ----
    # TODO: Currently we are dismissing any AIS messages that do not include the word 'fish' in ship type or
    # navigational status. We probably should save the MMSI of any vessel where the word 'fish' occurs even once,
    # and then save all messages from that vessel.
    # Include filtering for word "Trawl Fishing" or a subset of that string in the destination column
    # Include heading in the filtered fields?

    # Normalize helper
    def _norm_str_col(s):
        return s.fillna("").astype(str).str.strip().str.lower()

    # Start with False, then OR in evidence
    fishing_mask = pandas.Series(False, index=df.index)

    # Navigational status: look for phrases like "engaged in fishing"
    if "Navigational status" in df.columns:
        nav_norm = _norm_str_col(df["Navigational status"])
        fishing_mask = fishing_mask | nav_norm.str.contains("fish")

    # Ship type: can be text "Fishing"
    if "Ship type" in df.columns:
        # two paths: numeric-like vs text-like
        st = df["Ship type"]
        # text detection
        text_norm = _norm_str_col(st)
        is_fishing_text = text_norm.str.contains("fish")
        fishing_mask = fishing_mask | is_fishing_text

    # Apply fishing mask if either column existed. If neither existed, mask is all False → drop all.
    has_any_fishing_field = ("Navigational status" in df.columns) or ("Ship type" in df.columns)
    if has_any_fishing_field:
        df = df[fishing_mask]
    else:
        # No fields to identify fishing → nothing to keep.
        df = df.iloc[0:0]

    # ---- timestamps, filters, segmentation ----
    df = df.rename(columns={"# Timestamp": "Timestamp"})
    df["Timestamp"] = pandas.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df = df.drop_duplicates(["Timestamp", "MMSI"], keep="first")

    def track_filter(g):
        len_filt = len(g) > 256
        # still in knots here
        sog_filt = 1 <= g["SOG"].max() <= 50
        time_filt = (g["Timestamp"].max() - g["Timestamp"].min()).total_seconds() >= 60 * 60
        return len_filt and sog_filt and time_filt

    df = df.groupby("MMSI").filter(track_filter)
    df = df.sort_values(["MMSI", "Timestamp"])

    # segment by ≥15 min gap
    df["Segment"] = df.groupby("MMSI")["Timestamp"].transform(
        lambda x: (x.diff().dt.total_seconds().fillna(0) >= 15 * 60).cumsum()
    )

    # filter segments
    df = df.groupby(["MMSI", "Segment"]).filter(track_filter).reset_index(drop=True)

    # ---- units ----
    df["SOG"] = 0.514444 * df["SOG"]  # knots → m/s

    # ---- write parquet, partitioned by MMSI and Segment ----
    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    pyarrow.parquet.write_to_dataset(
        table,
        root_path=out_path,
        partition_cols=["MMSI", "Segment"],
    )




def preprocess(data_path: Path = None, output_folder: Path = None) -> None:
    print("Preprocessing data...")
    
    # Get the project root (maritime_domain_awareness folder)
    project_root = Path(__file__).resolve().parents[3]
    
    if data_path is None:
        data_path = project_root / "maritime_domain_awareness" / "data" / "Raw" / "aisdk-2025-03-01.csv"
    
    if output_folder is None:
        output_folder = project_root / "maritime_domain_awareness" / "data" / "Processed"
    
    fn(data_path, output_folder)
    
    #dataset = MyDataset(data_path)
    #dataset.preprocess(output_folder)

if __name__ == "__main__":
    typer.run(preprocess)
