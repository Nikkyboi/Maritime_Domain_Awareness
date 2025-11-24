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

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon



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
    # ----- read -----
    dtypes = {
        "MMSI": "object",
        "SOG": float,
        "COG": float,
        "Longitude": float,
        "Latitude": float,
        "Heading": float,
        "# Timestamp": "object",
        "Type of mobile": "object",
        # fishing-related columns for filtering
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

    # TODO: 
    # Include filtering for word "Trawl Fishing" or a subset of that string in the destination column
    # Consider whether simply dropping observations where any feature has a null value is the right move.
    # Alternatively, could simply drop the features that frequently has null values
    # Normalize features

    # ----- fishing vessel filters -----
    # Normalize helper
    def _norm_str_col(s):
        return s.fillna("").astype(str).str.strip().str.lower()

    fishing_mask_row = pandas.Series(False, index=df.index)

    nav_norm = _norm_str_col(df["Navigational status"])
    fishing_mask_row |= nav_norm.str.contains("fish")

    type_norm = _norm_str_col(df["Ship type"])
    fishing_mask_row |= type_norm.str.contains("fish")

    fishing_mmsi = df.loc[fishing_mask_row, "MMSI"].unique()

    df = df[df["MMSI"].isin(fishing_mmsi)]

    # ----- timestamps, filters, segmentation -----
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

    # ----- Drop observations where any feature has null value -----
    feature_cols = ["Latitude", "Longitude", "SOG", "COG", "Heading"]
    df = df.dropna(subset=feature_cols)

    # ----- Add Δt feature (seconds between messages within each segment) -----
    df["DeltaT"] = (
        df.groupby(["MMSI", "Segment"])["Timestamp"]
          .diff()
          .dt.total_seconds()
    )

    # for the first message in each segment, diff() is NaN → set to 0
    df["DeltaT"] = df["DeltaT"].fillna(0.0)

    # ----- units -----
    df["SOG"] = 0.514444 * df["SOG"]  # knots → m/s


    # ----- normalization of features -----
    norm_cols = ["SOG", "COG", "Heading", "DeltaT"]
    for col in norm_cols:
        mean = df[col].mean()
        std = df[col].std()

        # avoid division-by-zero / NaN for constant or degenerate columns
        if std == 0 or not np.isfinite(std):
            std = 1.0

        df[col] = (df[col] - mean) / std

    # ----- filter for ships at port -----
    def make_port_boundaries(data_path: Path = "data/Raw/port_locodes.csv"):
        df = pandas.read_csv(data_path,sep=";")
        dictionary = dict()
        for row in df.itertuples(index = False, name = None):
                name, code, coords = row
                if not code.startswith("DK"):
                    continue
                this_coord = []
                for coord in coords.split(","):
                    longitude,latitude = coord.split()
                    this_coord.append((float(longitude),float(latitude)))
                dictionary[code] = this_coord
        return dictionary

    def point_in_polygon(row):
        point = Point(row["Longitude"],row["Latitude"])
        return not polygon.contains(point)

    ports = make_port_boundaries()

    for port in ports:
        polygon = Polygon(ports[port])
        df = df[df.apply(point_in_polygon,axis=1)]



    # ---- ECEF unit vectors from lat/lon ----
    # degrees → radians
    lat_rad = np.deg2rad(df["Latitude"].values)
    lon_rad = np.deg2rad(df["Longitude"].values)

    # unit sphere, origin at Earth's center
    # assuming spherical Earth; height ignored
    cos_lat = np.cos(lat_rad)
    df["X"] = cos_lat * np.cos(lon_rad)
    df["Y"] = cos_lat * np.sin(lon_rad)
    df["Z"] = np.sin(lat_rad)

    
    # ----- Drop columns we don't want model training on ------
    df = df.drop(columns=["Navigational status", "Ship type", "Timestamp", "Latitude", "Longitude"])

    # ----- write parquet, partitioned by MMSI and Segment -----
    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    pyarrow.parquet.write_to_dataset(
        table,
        root_path=out_path,
        partition_cols=["MMSI", "Segment"],
    )




def preprocess(data_path: Path = "data/Raw/aisdk-2025-03-01.csv", output_folder: Path = "data/Processed/") -> None:
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
