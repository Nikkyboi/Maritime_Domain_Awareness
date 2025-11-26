#from maritime_domain_awareness.model import Model
#from maritime_domain_awareness.data import MyDataset
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch import nn
from models import Load_model
from PlotToWorldMap import PlotToWorldMap
from KalmanFilterWrapper import KalmanFilterWrapper

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
        self.N = X.shape[0] - seq_len - 1

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        x_seq = self.X[idx : idx + self.seq_len]          # [T, n_in]
        y_seq = self.y[idx + 1 : idx + 1 + self.seq_len]  # [T, n_out]
        return x_seq, y_seq

def load_and_split_data(
    input_file: str | Path,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
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
    

    # Select input + output columns
    in_cols  = ["X", "Y", "Z", "SOG", "COG", "Heading", "DeltaT"]
    # in_cols  = ["Latitude", "Longitude"]
    out_cols = ["X", "Y", "Z"]

    # Basic sanity check
    for c in in_cols + out_cols:
        if c not in df.columns:
            raise KeyError(f"Column {c!r} not found in {input_file}. Got: {list(df.columns)}")

    X = df[in_cols].to_numpy(dtype="float32")   # [N, 5]
    y = df[out_cols].to_numpy(dtype="float32")  # [N, 2]

    N = len(df)
    train_end = int(train_frac * N)
    val_end   = int((train_frac + val_frac) * N)

    # chronological splits (no shuffling)
    X_train, y_train = X[:train_end],      y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],        y[val_end:]

    print(f"N={N} -> train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Convert to torch tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t   = torch.from_numpy(X_val)
    y_val_t   = torch.from_numpy(y_val)
    X_test_t  = torch.from_numpy(X_test)
    y_test_t  = torch.from_numpy(y_test)

    return (X_train_t, y_train_t), (X_val_t, y_val_t), (X_test_t, y_test_t)

def train(model : nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        num_epochs: int = 10,
        learning_rate: float = 1e-3, 
        device: str | torch.device | None = None,):
    
    # Set device (Use GPU if available)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    # Loss function and optimizer
    criterion = torch.nn.MSELoss() # Mean Squared Error
    #criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Track loss
    training_loss, validation_loss = [], []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for x_batch, y_batch in train_loader:
            # [B, T, Features]
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Batch first then transpose
            if getattr(model, 'batch_first', False) is False:
                x_batch = x_batch.transpose(0, 1)  # [T, B, Features]
                y_batch = y_batch.transpose(0, 1)  # [T, B, Features]

            preds = model(x_batch)
            
            loss = criterion(preds, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)
        training_loss.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Batch first then transpose
                if getattr(model, 'batch_first', False) is False:
                    x_batch = x_batch.transpose(0, 1)  # [T, B, Features]
                    y_batch = y_batch.transpose(0, 1)  # [T, B, Features]

                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        validation_loss.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}")

    return training_loss, validation_loss, 

if __name__ == "__main__":
    """
    
    This is an example of how to use the training function with the AIS trajectory dataset.
    
    Same setup should work for:
    - RNN_models.myRNN
    - RNN_models.myLSTM
    - RNN_models.myGRU
    - Transformer_model.myTransformer
    - Mamba_model.Mamba
    - Kalman filter
    
    Because AISTrajectorySeq2Seq dataset returns sequences of shape:
    [B, T, n_in]
    [B, T, n_out]
    It gets transposed to [T, B, n_in] if model.batch_first is False.
    
    car2pi
    
    carophy
    """
    device = None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --------------------------
    
    # Choose the name of the model to train
    # Options: "rnn", "lstm", "gru", "transformer", "kalman" & "mamba"
    model_name = "transformer"

    # Look for the existing model
    if model_name == "kalman":
        models = "kalman"
    else:
        models = "models/ais_" + model_name + "_model.pth"
    
    # Inputs, Hidden, Outputs
    n_in = 7    # lat, lon
    n_hid = 64  # hidden size
    n_out = 3   # lat, lon
    
    # Epochs and learning rate
    epochs = 5
    lr = 1e-3
    
    # -------------------------
    if Path(models).exists():
        print("Loading existing model:")
        model = Load_model.load_model(model_name, n_in, n_out, n_hid)
        model.load_state_dict(torch.load(models))
    elif models == "kalman":
        print("Using Kalman Filter model...")
    else:
        print("Training new model...")
        model = Load_model.load_model(model_name, n_in, n_out, n_hid)



    training_sequences = []

    # Find all training sequences in the data folder
    base_folder = Path("data/Processed")
    # base_folder = Path("maritime_domain_awareness/data/Processed")

    for ship_folder in base_folder.iterdir():
        if not ship_folder.is_dir():
            continue
        for inner_folder in ship_folder.iterdir():
            if not inner_folder.is_dir():
                continue
            for parquet_file in inner_folder.glob("*.parquet"):
                training_sequences.append(parquet_file)

    print("Found training sequences:", len(training_sequences))

    train_loss_total = []
    val_loss_total = []
    avg_test_loss = []
    i = 0
    for seq in training_sequences:
        
        print("Training on sequence:", seq)
        # Load data
        input_file = seq
        
        # split into train, test and validation sets:
        # Input = latitude, longitude, sog, cog, heading
        # Output = latitude, longitude
        # split 70/15/15
        (train_X, train_y), (val_X, val_y), (test_X, test_y) = load_and_split_data(input_file)
        print("X_train.shape:", train_X.shape)
        print("y_train.shape:", train_y.shape)

        # Choose sequence length (number of timesteps per sample)
        seq_len = 50
        
        # Check if we have enough data for at least one sequence
        # We need at least seq_len + 1 samples
        if len(train_X) <= seq_len + 1:
            print(f"Skipping sequence {seq}: Train set too small ({len(train_X)} <= {seq_len + 1})")
            continue
        
        if len(val_X) <= seq_len + 1:
             # If validation is too small, we can either skip validation or skip the whole file.
             # Usually better to skip the file to avoid errors in val_loader
             print(f"Skipping sequence {seq}: Val set too small ({len(val_X)} <= {seq_len + 1})")
             continue

        # Create datasets and dataloaders
        train_dataset = AISTrajectorySeq2Seq(train_X, train_y, seq_len)
        val_dataset   = AISTrajectorySeq2Seq(val_X,   val_y,   seq_len)
        test_dataset  = AISTrajectorySeq2Seq(test_X,  test_y,  seq_len)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)
        
        if models == "kalman":
            model = KalmanFilterWrapper(dt=1.0, process_variance=1e-5, measurement_variance=0.1, init_error=1.0)
            train_loss = []
            val_loss = []
            # Validation loss for Kalman Filter
            model.eval()
            with torch.no_grad():
                val_err = 0.0
                for x_batch, y_batch in val_loader:
                    outputs = model(x_batch)
                    error = nn.MSELoss()(outputs, y_batch)
                    val_err += error.item()
                avg_val_loss = val_err / len(val_loader)
                val_loss.append(avg_val_loss)
        else:
            train_loss, val_loss = train(
                model,
                train_loader,
                val_loader,
                num_epochs=epochs,
                learning_rate=lr,
            )
        predictedPoint = []
        actualPoint = []
        
        # Predict the test set vs true values
        model.to(device)
        model.eval()
        err = 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                # Move tensors to the same device as the model
                inputs = inputs.to(device)
                targets = targets.to(device)

                # If the model expects sequences with time-first (T, B, F)
                # transpose like in the training/validation loops
                if getattr(model, 'batch_first', False) is False:
                    inputs = inputs.transpose(0, 1)
                    targets = targets.transpose(0, 1)
                outputs = model(inputs)
                # Ensure both outputs and targets are [batch_size, seq_len, n_out]
                if outputs.dim() == 2:
                    outputs = outputs.unsqueeze(0)
                if targets.dim() == 2:
                    targets = targets.unsqueeze(0)
                predictedPoint.append(outputs)
                actualPoint.append(targets)
                error = nn.MSELoss()(outputs, targets)
                err += error.item()
        predictedPoint = torch.cat(predictedPoint, dim=0)
        actualPoint = torch.cat(actualPoint, dim=0)
        avg_err = err / len(test_loader)

        train_loss_total.extend(train_loss)
        val_loss_total.extend(val_loss)
        avg_test_loss.append(avg_err)
        if i == 20:
            break
        i += 1
    # Save model
    PlotToWorldMap(actualPoint = actualPoint, predictedPoint = predictedPoint)
    torch.save(model.state_dict(), models)
    plot = True
    if plot == True:
        # Show training and validation loss
        plt.plot(train_loss_total, label="Train Loss")
        plt.plot(val_loss_total, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("reports/Training and Validation Loss")
        #plt.savefig("reports/training_validation_loss_temp.png")
        plt.show()

        # Show test loss
        plt.figure()
        plt.plot(avg_test_loss, label="Test Loss")
        plt.xlabel("Sequence")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("reports/Test Loss")
        #plt.savefig("reports/test_loss_temp.png")
        plt.show()