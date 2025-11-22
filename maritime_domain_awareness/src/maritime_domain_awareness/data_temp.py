from pathlib import Path
import pandas as pd
import typer
from torch.utils.data import Dataset
import pandas
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from zipfile import ZipFile

class PreprocessDataset(Dataset):
    """ 
    Custom Dataset for maritime domain awareness data.

    Takes in a path to CSV, then preprocesses the data to parquet format.
    Options:
    - See length of dataset
    - Get a sample by index
    - Preprocess the data and save to output folder in Parquet format
    """
    def __init__(self, data_path: Path) -> None:
        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} does not exist.")
        
        if data_path.suffix == ".csv":
            raise ValueError("CSV files will run out of memory.")
        
        self.data_path = data_path
        self._len = None
        
        if data_path.suffix == ".parquet":
            self._pf = pq.ParquetFile(self.data_path)
            
            # Length of the dataset (number of rows)
            self._len = sum(self._pf.metadata.row_group(i).num_rows
                            for i in range(self._pf.metadata.num_row_groups))
        
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self._len

    def __getitem__(self, index: int):
        """Return a given sample from the dataset.
        
        Currently only return the MMSI, latitude, longitude, and timestamp.
        """
        df = pd.read_parquet(self.data_path) 
        row = df.iloc[index]
        return {
            "mmsi": int(row.mmsi),
            "lat": float(row.lat),
            "lon": float(row.lon),
            "ts": row.timestamp,
        }

    def preprocess(self, output_path: str | Path) -> None:
        """
        Preprocess the raw data and save it to the output folder.
        
        Preprocessing steps:
        - Filter for Danish ships (MMSI starting with 219 or 220)
        - Filter for fishing vessels (to be implemented)
        - Remove duplicates (to be implemented)
        - regularize timestamps to 1-minute intervals by interpolation (to be implemented)
        - Save the preprocessed data as a Parquet file
        
        """
        # input and output paths
        input_path = Path(input_path)
        output_path = Path(output_path)

        # load data
        df = pd.read_csv(input_path)
        df_danish = filterForDanishShips(df)
        # df_fishing = filterForFishingVessels(df_danish)
        # df_no_duplicates = remove_duplicates(input_file, output_file, cols_to_check=["# Timestamp", "MMSI"])
        # df_regularized = resample_to_freq_grid(input_file, output_file, freq="10S")

        # Save the preprocessed data as a Parquet file
        df_danish.to_parquet(output_path, index=False)

def filterForDanishShips(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    # Keep only ships with Danish MMSI (starting with 219 and 220)

    Based on the guidelines from the Danish Maritime Authority:
    https://www.dma.dk/ship-survey-and-registration/apply-for-a-certificate/guidance-for-ship-station-license
    """

    # Filter for Danish MMSI (219 and 220)
    df_danish = df[df["MMSI"].astype(str).str.startswith(("219", "220"))].copy()
    return df_danish

def filterForFishingVessels(df: pandas.DataFrame) -> pandas.DataFrame:
    """
        # Keep only fishing vessels

    ???
    """
    pass
    return df


def extract_single_fishing_ship(
    input_path: str | Path,
    output_path: str | Path,
    mmsi_col: str = "MMSI",
    status_col: str = "Navigational status",
    fishing_status: str = "Engaged in fishing",
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    # Normalize nav status text just in case of spaces
    df[status_col] = df[status_col].astype(str).str.strip()

    # all rows where this ship is engaged in fishing
    fishing_rows = df[df[status_col] == fishing_status]

    if fishing_rows.empty:
        raise ValueError(f"No rows with {status_col} == {fishing_status!r} found.")

    # MMSI from the FIRST such row in the file
    chosen_mmsi = fishing_rows[mmsi_col].iloc[0]
    print(f"Chosen MMSI: {chosen_mmsi} (first to have '{fishing_status}')")

    # Keep ONLY this MMSI
    df_ship = df[df[mmsi_col] == chosen_mmsi].copy()
    print(f"Rows for MMSI {chosen_mmsi}: {len(df_ship)}")

    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_ship.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    
def remove_duplicates(
    input_path: str | Path,
    output_path: str | Path,
    cols_to_check: list[str] | None = None,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    # Remove exact duplicates
    df_no_duplicates = df.drop_duplicates()

    # Remove duplicates based on MMSI and Timestamp
    df_no_duplicates = df_no_duplicates.drop_duplicates(subset=cols_to_check)

    df_no_duplicates.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

def resample_to_10s_grid(
    input_path: str | Path,
    output_path: str | Path,
    timestamp_col: str = "# Timestamp",
    freq: str = "10S",
) -> None:
    """
    Take a single-ship AIS CSV and resample it to a regular time grid
    (e.g. every 10 seconds). No duplicates, evenly spaced timestamps.

    - Parses the timestamp column.
    - Sorts by time.
    - Resamples to `freq` with forward-fill for missing values.
    - Keeps full datetime and also adds a HH:MM:SS column.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)

    # Parse timestamp
    df["Timestamp"] = pd.to_datetime(
        df[timestamp_col],
        format="%d/%m/%Y %H:%M:%S",
    )

    # Sort by time (important before resampling)
    df = df.sort_values("Timestamp")

    # Set index to timestamp for resampling
    df = df.set_index("Timestamp")

    # Resample to a regular 10-second grid
    # - ffill: forward-fill last known values into gaps
    #   (you can change to .interpolate() for lat/lon if you want)
    df_resampled = df.resample(freq).ffill()

    # Bring the timestamp back as a column
    df_resampled = df_resampled.reset_index()

    # Optional: add a HH:MM:SS-only column (no date)
    df_resampled["time_10s"] = df_resampled["Timestamp"].dt.strftime("%H:%M:%S")

    # Remove the original timestamp column if needed
    df_resampled = df_resampled.drop(columns=[timestamp_col], errors='ignore')

    # Remove date so only time remains in the timestamp column
    df_resampled["Timestamp"] = df_resampled["Timestamp"].dt.strftime("%H:%M:%S")
    
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_resampled.to_csv(output_path, index=False)
    print(f"Resampled data saved to: {output_path}")

def preprocess(data_path: Path = "data/Processed/2025-03-01/danish_219_220.parquet", output_folder: Path = "data/Processed/2025-03-01") -> None:
    # "data/Processed/2025-03-01/aisdk-2025-03-01.zip"
    #print("Preprocessing data...")
    
    #my_dataset = MyDataset(data_path)
    #my_dataset.preprocess(output_folder)
    #print("len(my_dataset):", len(my_dataset))
    
    
    #dataset = MyDataset(data_path)
    #dataset.preprocess(output_folder)
    #input_file = "data/Raw/2025-03-01/aisdk-2025-03-01_single_fishing_ship.csv"
    input_file = "data/Raw/2025-03-01/training_example.csv"
    output_file = "data/Raw/2025-03-01/training_example_temp.csv"
    #keep_only_danish_ships(input_file, output_file)
    #extract_single_fishing_ship(input_file, output_file)
    #remove_duplicates(output_file, output_file, cols_to_check=["# Timestamp", "MMSI", "Latitude", "Longitude", "SOG", "COG", "Heading"])
    #resample_to_10s_grid(input_path=input_file,output_path=output_file,freq="10S")

if __name__ == "__main__":
    typer.run(preprocess)