# Standard library imports
from pathlib import Path
import random

# Third-party imports
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Local project imports
from .data import (
    load_and_split_data,
    AISTrajectorySeq2Seq,
    compute_global_norm_stats,
    find_all_parquet_files,
)
from .evaluate import evaluate_model, trajectory_prediction
from .models import Load_model
from .KalmanFilterWrapper import KalmanFilterWrapper
#from PlotToWorldMap import PlotToWorldMap

def train(model : nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        dynamic_epochs: bool = False,
        device: str | torch.device | None = None,):
    
    # Set device (Use GPU if available)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    # Loss function and optimizer
    # Prediction for regression tasks
    criterion = torch.nn.MSELoss() # Mean Squared Error
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Track loss
    training_loss, validation_loss = [], []
    
    if dynamic_epochs:
        epochs = 100  # big upper bound, early stopping will cut it
        patience = 5        # how many epochs to wait for improvement
        min_delta = 1e-8    # minimum change to count as improvement
    else:
        epochs = num_epochs
        patience = None
        min_delta = None
        
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
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
        
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}")
        
        # ----- Early stopping -----
        if dynamic_epochs:
            if avg_val_loss + min_delta < best_val_loss:
                # significant improvement
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1} due to validation loss plateau.")
                    break

    return training_loss, validation_loss

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
    # ----------------------------
    # Set device (Use GPU if available)
    device = None
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --------------------------
    # Choose the name of the model to train
    # Options: "rnn", "lstm", "gru", "transformer", "kalman"
    #model_name = "Transformer"
    #models = ["rnn", "lstm", "gru", "transformer"]
    models = ["rnn", "gru", "transformer"]
    #models = ["lstm"]
    
    for model_name in models:
        # Look for the existing model
        if model_name == "kalman":
            models = "kalman"
        else:
            models = "models/ais_" + model_name + "_model.pth"
        
        # Inputs, Hidden, Outputs
        n_in = 4    # lat, lon
        n_hid = 64  # hidden size
        n_out = 2   # lat, lon
        
        # Epochs and learning rate
        epochs = 20
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

        # ----------------------------
        # Find all training sequences
        
        training_sequences = []
        # Find all training sequences in the data folder
        base_folder = Path("data/Processed/")
        training_sequences = find_all_parquet_files(base_folder)
        print("Found training sequences:", len(training_sequences))
        
        
        # ----------------------------
        # Compute global normalization stats
        global_in_mean, global_in_std, global_delta_mean, global_delta_std = compute_global_norm_stats(
            base_folder,
            train_frac=0.7,
            IN_COLS = ["Latitude", "Longitude", "SOG", "COG"],
            DELTA_COLS = ["dLatitude", "dLongitude"],
        )
        print("Global input mean:", global_in_mean)
        print("Global input std:",  global_in_std)
        print("Global delta mean:", global_delta_mean)
        print("Global delta std:",  global_delta_std)
        
        # ----------------------------
        # Training loop over all sequences
        train_loss_total = []
        val_loss_total = []
        avg_test_loss = []
        i = 0
        
        tests_to_run = []
        
        # ----------------------------
        # Training sequence loop
        # For each training sequence file, train the model
        for seq in training_sequences:
            print("Training on sequence:", seq)
            # Load data
            input_file = seq
            
            # split into train, test and validation sets:
            # Input = latitude, longitude, sog, cog, heading
            # Output = latitude, longitude
            # split 70/15/15
            (train_X, train_y), (val_X, val_y), (test_X, test_y) = load_and_split_data(
                input_file,
                in_mean=global_in_mean,
                in_std=global_in_std,
                delta_mean=global_delta_mean,
                delta_std=global_delta_std,
            )
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

            # Save test loaders for later evaluation
            tests_to_run.append((seq, test_loader))
            
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
                    dynamic_epochs = True
                )

                # Show under or overfitting
                train_loss_total.extend(train_loss)
                val_loss_total.extend(val_loss)
        
        # ----------------------------
        # Save model
        torch.save(model.state_dict(), models)
        
        # Show plots?
        plot = True
        plot_trajectory = False
        
        # ----------------------------
        # Plot training and validation loss
        # This shows under or overfitting
        if plot:
            # Show training and validation loss
            plt.plot(train_loss_total, label="Train Loss")
            plt.plot(val_loss_total, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("reports/Training and Validation Loss")
            #plt.savefig("reports/training_validation_loss_temp.png")
            plt.show()
                
            # ----------------------------
            # Evaluate on all test sets
            evaluate_model(model, tests_to_run, device)

            
            # ----------------------------
        if plot_trajectory:
            # pick 5 random test sequences and do full rollout
            random_runs = random.sample(tests_to_run, min(5, len(tests_to_run)))
            for example_seq, _ in random_runs:
                (train_X, train_y), (val_X, val_y), (test_X, test_y) = load_and_split_data(
                    example_seq,
                    in_mean=global_in_mean,
                    in_std=global_in_std,
                    delta_mean=global_delta_mean,
                    delta_std=global_delta_std,
                )
                trajectory_prediction(
                    model,
                    test_X,
                    in_mean=global_in_mean,
                    in_std=global_in_std,
                    delta_mean=global_delta_mean,
                    delta_std=global_delta_std,
                    device=device,
                    seq_len=50,   # or whatever seq_len you trained with
                )
