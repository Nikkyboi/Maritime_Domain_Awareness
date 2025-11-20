#from maritime_domain_awareness.model import Model
#from maritime_domain_awareness.data import MyDataset
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch import nn
from . import RNN_models
from . import Transformer_model

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
    df = pd.read_csv(input_file)

    # Select input + output columns
    #in_cols  = ["Latitude", "Longitude", "SOG", "COG", "Heading"]
    in_cols  = ["Latitude", "Longitude"]
    out_cols = ["Latitude", "Longitude"]

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
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return training_loss, validation_loss

if __name__ == "__main__":
    """
    
    This is an example of how to use the training function with the AIS trajectory dataset.
    
    Same setup should work for:
    - RNN_models.myRNN
    - RNN_models.myLSTM
    - RNN_models.myGRU
    - Transformer_model.myTransformer
    
    Because AISTrajectorySeq2Seq dataset returns sequences of shape:
    [B, T, n_in]
    [B, T, n_out]
    It gets transposed to [T, B, n_in] if model.batch_first is False.
    
    """
    # Only training on one ship
    input_file = "data/Raw/2025-03-01/training_example_temp.csv"
    
    # split into train, test and validation sets:
    # Input = latitude, longitude, sog, cog, heading
    # Output = latitude, longitude
    # split 70/15/15
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = load_and_split_data(input_file)
    print("X_train.shape:", train_X.shape)
    print("y_train.shape:", train_y.shape)

    seq_len = 50
    
    train_dataset = AISTrajectorySeq2Seq(train_X, train_y, seq_len)
    val_dataset   = AISTrajectorySeq2Seq(val_X,   val_y,   seq_len)
    test_dataset  = AISTrajectorySeq2Seq(test_X,  test_y,  seq_len)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)
    
    #models = "models/ais_RNN_model.pth"
    #models = "models/ais_lstm_model.pth"
    #models = "models/ais_gru_model.pth"
    models = "models/ais_transformer_model.pth"
    
    if Path(models).exists():
        print("Loading existing model...")
        """
        RNN_models = RNN_models.myLSTM(
            n_in=2,
            n_hid=64,
            n_out=2,
            num_layers=2,
            batch_first=False,
            dropout=0.1,
        )
        RNN_models.load_state_dict(torch.load("models/ais_lstm_model.pth"))
        """
        n_in = 2    # lat, lon
        n_hid = 64  # hidden size
        n_out = 2   # lat, lon
        models = Transformer_model.myTransformer(
        n_in=n_in,
        n_hid=n_hid,
        n_out=n_out,
        num_layers=2,
        n_heads=4,
        dim_feedforward=128,
        dropout=0.1,
        batch_first=False,
        )
        models.load_state_dict(torch.load("models/ais_transformer_model.pth"))
    else:
        print("Training new model...")
        # Define model
        n_in = 2    # lat, lon
        n_hid = 64  # hidden size
        n_out = 2   # lat, lon
        model = Transformer_model.myTransformer(
            n_in=n_in,
            n_hid=n_hid,
            n_out=n_out,
            num_layers=2,
            n_heads=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=False,
            )

        train_loss, val_loss = train(
            model,
            train_loader,
            val_loader,
            num_epochs=11,
            learning_rate=1e-3,
        )
        
        # Save model
        torch.save(model.state_dict(), models)
    
        # Show training and validation loss
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("reports/Training and Validation Loss")
        plt.savefig("reports/training_validation_loss_temp.png")
        plt.show()

    