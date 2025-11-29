import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

class AISTrajectorySeq2Seq(Dataset):
    """
    Dataset for AIS trajectory sequence-to-sequence modeling.
    Many to many model.
    
    It simply stores the full dataset in memory as tensors X and y,
    then returns sequences of length seq_len for each index.
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor, seq_len: int):
        """
        X: [N, n_in]
        y: [N, n_out]
        seq_len: number of timesteps per input sequence

        For each i, returns:
          x_seq: X[i : i+seq_len]         -> [seq_len, n_in]
          y_seq: y[i+1 : i+1+seq_len]     -> [seq_len, n_out]
        """
        assert X.shape[0] == y.shape[0], "X and y must have same length"
        self.X = X
        self.y = y
        self.seq_len = seq_len

        # we need i+1+seq_len <= N  ->  i <= N - seq_len - 1
        self.N = X.shape[0] - seq_len + 1

    def __len__(self) -> int:
        return max(self.N, 0)

    def __getitem__(self, idx: int):
        x_seq = self.X[idx : idx + self.seq_len]          # [T, n_in]
        y_seq = self.y[idx : idx + self.seq_len]          # [T, 2] deltas
        return x_seq, y_seq
    
    
def load_and_split_data(
    input_file: str | Path,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    in_mean = None,
    in_std = None,
    delta_mean = None,
    delta_std = None,
):
    """
    Load and split the dataset into train, validation, and test sets.
    """
    input_file = Path(input_file)
    
    if input_file.suffix == ".csv":
        df = pd.read_csv(input_file)
    elif input_file.suffix == ".parquet":
        df = pd.read_parquet(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")
    

    # ---- Select input + output columns ----
    in_cols  = ["Latitude", "Longitude", "SOG", "COG"]
    #out_cols = ["Latitude", "Longitude"]
    out_cols = ["dLatitude", "dLongitude"] # predict deltas of lat, lon
    
    # ----- Split into train, val, test -----
    # Compute deltas on raw df
    df["dLatitude"]  = df["Latitude"].diff().fillna(0.0)
    df["dLongitude"] = df["Longitude"].diff().fillna(0.0)

    N = len(df)
    train_end = int(train_frac * N)
    val_end   = int((train_frac + val_frac) * N)

    # ----- normalization -----
    if in_mean is None or in_std is None or delta_mean is None or delta_std is None:
        raise ValueError("in_mean, in_std, delta_mean, delta_std must be provided for normalization.")

    # Inputs
    mean_in = pd.Series(in_mean, index=in_cols)
    std_in  = pd.Series(in_std,  index=in_cols)

    # Deltas
    mean_delta = pd.Series(delta_mean, index=out_cols)
    std_delta  = pd.Series(delta_std,  index=out_cols)

    df_X = (df[in_cols]    - mean_in)   / std_in
    df_y = (df[out_cols]   - mean_delta) / std_delta

    # chronological splits (no shuffling)
    df_train_X = df_X.iloc[:train_end]
    df_val_X   = df_X.iloc[train_end:val_end]
    df_test_X  = df_X.iloc[val_end:]

    df_train_y = df_y.iloc[:train_end]
    df_val_y   = df_y.iloc[train_end:val_end]
    df_test_y  = df_y.iloc[val_end:]

    print(f"N={N} -> train={len(df_train_X)}, val={len(df_val_X)}, test={len(df_test_X)}")

    X_train = df_train_X.to_numpy(dtype="float32")
    y_train = df_train_y.to_numpy(dtype="float32")

    X_val   = df_val_X.to_numpy(dtype="float32")
    y_val   = df_val_y.to_numpy(dtype="float32")

    X_test  = df_test_X.to_numpy(dtype="float32")
    y_test  = df_test_y.to_numpy(dtype="float32")

    # Convert to torch tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t   = torch.from_numpy(X_val)
    y_val_t   = torch.from_numpy(y_val)
    X_test_t  = torch.from_numpy(X_test)
    y_test_t  = torch.from_numpy(y_test)

    return (X_train_t, y_train_t), (X_val_t, y_val_t), (X_test_t, y_test_t)

# Global column definitions
IN_COLS = ["Latitude", "Longitude", "SOG", "COG"]
DELTA_COLS = ["dLatitude", "dLongitude"]

def compute_global_norm_stats(base_folder: Path, train_frac: float = 0.7, IN_COLS = IN_COLS, DELTA_COLS = DELTA_COLS):
    """
    Compute global mean/std for:
      - IN_COLS  = [Latitude, Longitude, SOG, COG]   (inputs)
      - DELTA_COLS = [dLatitude, dLongitude]         (targets = deltas)
    over the TRAIN portions of all files.
    """
    # Inputs
    count_in = 0
    sum_in = np.zeros(len(IN_COLS), dtype=np.float64)
    sum_sq_in = np.zeros(len(IN_COLS), dtype=np.float64)

    # Deltas
    count_delta = 0
    sum_delta = np.zeros(len(DELTA_COLS), dtype=np.float64)
    sum_sq_delta = np.zeros(len(DELTA_COLS), dtype=np.float64)

    for ship_folder in base_folder.iterdir():
        if not ship_folder.is_dir():
            continue
        for inner_folder in ship_folder.iterdir():
            if not inner_folder.is_dir():
                continue
            for parquet_file in inner_folder.glob("*.parquet"):
                df = pd.read_parquet(parquet_file)[IN_COLS].dropna()
                N = len(df)
                if N < 2:
                    continue

                train_end = int(train_frac * N)
                df_train = df.iloc[:train_end].copy()

                # Compute deltas on the train part
                df_train["dLatitude"]  = df_train["Latitude"].diff().fillna(0.0)
                df_train["dLongitude"] = df_train["Longitude"].diff().fillna(0.0)

                arr_in = df_train[IN_COLS].to_numpy(dtype=np.float64)
                arr_delta = df_train[DELTA_COLS].to_numpy(dtype=np.float64)

                # Accumulate for inputs
                count_in += len(arr_in)
                sum_in   += arr_in.sum(axis=0)
                sum_sq_in += (arr_in ** 2).sum(axis=0)

                # Accumulate for deltas
                count_delta += len(arr_delta)
                sum_delta   += arr_delta.sum(axis=0)
                sum_sq_delta += (arr_delta ** 2).sum(axis=0)

    # Inputs stats
    mean_in = sum_in / count_in
    var_in  = sum_sq_in / count_in - mean_in**2
    std_in  = np.sqrt(np.maximum(var_in, 1e-12))

    # Deltas stats
    mean_delta = sum_delta / count_delta
    var_delta  = sum_sq_delta / count_delta - mean_delta**2
    std_delta  = np.sqrt(np.maximum(var_delta, 1e-12))

    return (
        mean_in.astype("float32"),
        std_in.astype("float32"),
        mean_delta.astype("float32"),
        std_delta.astype("float32"),
    )
    
def find_all_parquet_files(base_folder: Path):
    """
    Find all Parquet files in the given base folder and its subfolders.
    """
    training_sequences = []
    for ship_folder in base_folder.iterdir():
            if not ship_folder.is_dir():
                continue
            for inner_folder in ship_folder.iterdir():
                if not inner_folder.is_dir():
                    continue
                for parquet_file in inner_folder.glob("*.parquet"):
                    training_sequences.append(parquet_file)
    
    return training_sequences

if __name__ == "__main__":
    # Example usage
    data_path = "data/Processed/2025-03-01/danish_219_220.parquet"
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_split_data(
        data_path,
        in_mean=[56.0, 10.0, 5.0, 180.0],
        in_std=[1.0, 1.0, 2.0, 90.0],
        delta_mean=[0.0, 0.0],
        delta_std=[0.0001, 0.0001],
    )
    print("Train X shape:", X_train.shape)
    print("Train y shape:", y_train.shape)
    print("Val X shape:", X_val.shape)
    print("Val y shape:", y_val.shape)
    print("Test X shape:", X_test.shape)
    print("Test y shape:", y_test.shape)