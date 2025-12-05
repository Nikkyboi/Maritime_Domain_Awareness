# Standard library imports
from pathlib import Path
import random
import copy
import logging
from typing import Optional

# Third-party imports
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

# Local project imports
from .data import (
    load_and_split_data,
    AISTrajectorySeq2Seq,
    compute_global_norm_stats,
    find_all_parquet_files,
)
from .evaluate import evaluate_model
from .models import Load_model
from .KalmanFilterWrapper import KalmanFilterWrapper
#from PlotToWorldMap import PlotToWorldMap

def train(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    dynamic_epochs: bool = False,
    device: str | torch.device | None = None,
   logger: Optional[logging.Logger] = None,
   ):
    # Set device (Use GPU if available)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"{device}")
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Track loss
    training_loss, validation_loss = [], []
    
    # Early stopping config
    max_epochs = num_epochs
    if dynamic_epochs:
        patience = 10        # epochs with no significant improvement
        min_delta = 1e-2     # improvement threshold (in loss units)
    else:
        patience = None
        min_delta = None
    
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # ----------------------
        # Training
        # ----------------------
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Batch first then transpose if needed
            batch_first = getattr(model, "batch_first", False)
            if not batch_first:
                x_batch = x_batch.transpose(0, 1)  # [T, B, F]
                y_batch = y_batch.transpose(0, 1)  # [T, B, F]

            preds = model(x_batch)
            #loss = criterion(preds, y_batch)
            
            if batch_first:
                # [B, T, F] -> [B, F]
                preds_last = preds[:, -1, :]
                y_last     = y_batch[:, -1, :]
            else:
                # [T, B, F] -> [B, F]
                preds_last = preds[-1]   # preds[-1, :, :]
                y_last     = y_batch[-1] # y_batch[-1, :, :]

            loss = criterion(preds_last, y_last)

            optimizer.zero_grad()
            loss.backward()
            # ----------------------
            # Gradient clipping (?) to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # ----------------------
            
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)
        training_loss.append(avg_train_loss)
        
        # ----------------------
        # Validation
        # ----------------------
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                batch_first = getattr(model, "batch_first", False)

                if not batch_first:
                    x_batch = x_batch.transpose(0, 1)
                    y_batch = y_batch.transpose(0, 1)

                preds = model(x_batch)
                
                if batch_first:
                    preds_last = preds[:, -1, :]
                    y_last     = y_batch[:, -1, :]
                else:
                    preds_last = preds[-1]
                    y_last     = y_batch[-1]

                
                loss = criterion(preds_last, y_last)
                
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        validation_loss.append(avg_val_loss)
        
        logger.info(
            f"Epoch {epoch+1}/{max_epochs}, "
            f"Train Loss: {avg_train_loss:.8f}, "
            f"Val Loss: {avg_val_loss:.8f}"
        )
        
        # ----------------------
        # Early stopping logic
        # ----------------------
        if dynamic_epochs:
            if avg_val_loss < best_val_loss - min_delta:
                # Significant improvement
                best_val_loss = avg_val_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        f"Early stopping at epoch {epoch+1} "
                        f"(no val improvement for {patience} epochs)."
                    )
                    break

    # Restore best model if early stopping was used
    if dynamic_epochs and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return training_loss, validation_loss

def main():
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
    # dataset = MyDataset(...)
    
    device = None
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Simple logging setup
    logger = logging.getLogger("ais_training")
    logger.setLevel(logging.INFO)
    # Avoid adding handlers twice if this file is imported
    if not logger.handlers:
        # Log to stdout (so you see it in HPC job output)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Log to file (so you can inspect later)
        file_handler = logging.FileHandler("training.log")
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    logger.info(f"{device}")
    # --------------------------
    # Choose the name of the model to train
    # Options: "rnn", "lstm", "gru", "transformer", "kalman"
    models = ["rnn", "lstm", "gru", "transformer"]
    #models = ["transformer"]
    
    for model_name in models:
        # Look for the existing model
        if model_name == "kalman":
            models = "kalman"
        else:
            models = "models/ais_" + model_name + "_model.pth"
        
        # Inputs, Hidden, Outputs
        n_in = 4    # lat, lon
        n_hid = 256  # hidden size
        n_out = 4   # lat, lon
        
        # Epochs and learning rate
        epochs = 200
        lr = {"rnn": 1e-5, "lstm": 1e-5, "gru": 1e-5, "transformer": 1e-3}[model_name]
        logger.info(f"Using learning rate: {lr}")
        # -------------------------
        if Path(models).exists():
            logger.info("Loading existing model:")
            model = Load_model.load_model(model_name, n_in, n_out, n_hid)
            model.load_state_dict(torch.load(models))
        elif models == "kalman":
            logger.info("Using Kalman Filter model...")
        else:
            logger.info("Training new model...")
            model = Load_model.load_model(model_name, n_in, n_out, n_hid)

        # ----------------------------
        # Find all training sequences
        
        training_sequences = []
        # Find all training sequences in the data folder
        base_folder = Path("data/Processed/")
        training_sequences = find_all_parquet_files(base_folder)
        logger.info(f"Found training sequences:{len(training_sequences)}")
        
        def split_into_n_chunks(seq_list, n_chunks):
            """Split seq_list into n_chunks parts as evenly as possible."""
            k, m = divmod(len(seq_list), n_chunks)
            # First m chunks will have size k+1, the rest size k
            return [
                seq_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
                for i in range(n_chunks)
            ]

        # Split training sequences into 8 chunks
        chunks = split_into_n_chunks(training_sequences, 60)
        
        # ----------------------------
        # Compute global normalization stats
        global_in_mean, global_in_std, global_delta_mean, global_delta_std = compute_global_norm_stats(
            base_folder,
            train_frac=0.7,
            IN_COLS = ["Latitude", "Longitude", "SOG", "COG"],
            DELTA_COLS = ["dLatitude", "dLongitude", "dSOG", "dCOG"],
        )
        logger.info(f"Global input mean:{global_in_mean}")
        logger.info(f"Global input std:{global_in_std}")
        logger.info(f"Global delta mean:{global_delta_mean}")
        logger.info(f"Global delta std:{global_delta_std}")

        print("Global delta std:",  global_delta_std)
        
        import json
        Path("models").mkdir(parents=True, exist_ok=True)
        with open(f"models/norm_params_{model_name}.json", "w") as f:
            json.dump({
                "delta_mean": global_delta_mean.tolist(),
                "delta_std": global_delta_std.tolist()
            }, f)
        
        # ----------------------------
        # Training loop over all sequences
        train_loss_total = []
        val_loss_total = []
        avg_test_loss = []
        i = 0
        
        tests_to_run = []

        # Split the training_sequences into k equal sequences
        
        # ----------------------------
        # For each training sequence file, train the model
        for chunk_idx, seq_files in enumerate(chunks):
            #print("Training on sequence:", seq)
            # Load data
            #input_file = seq
            
            # split into train, test and validation sets:
            # Input = latitude, longitude, sog, cog, heading
            # Output = latitude, longitude
            # split 70/15/15
            (train_X, train_y), (val_X, val_y), (test_X, test_y) = load_and_split_data(
                input_files = seq_files,
                in_mean=global_in_mean,
                in_std=global_in_std,
                delta_mean=global_delta_mean,
                delta_std=global_delta_std,
                logger = logger,
            )
            logger.info(f"X_train.shape:{train_X.shape}")
            logger.info(f"y_train.shape:{train_y.shape}")

            # Choose sequence length (number of timesteps per sample)
            seq_len = 50
        
            # Create datasets and dataloaders
            train_dataset = AISTrajectorySeq2Seq(train_X, train_y, seq_len)
            val_dataset   = AISTrajectorySeq2Seq(val_X,   val_y,   seq_len)
            test_dataset  = AISTrajectorySeq2Seq(test_X,  test_y,  seq_len)

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
            test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

            # Save test loaders for later evaluation
            tests_to_run.append((f"chunk_{chunk_idx}", test_loader))
            
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
                    dynamic_epochs = True,
                    logger = logger,
                )

                # Show under or overfitting
                train_loss_total.extend(train_loss)
                val_loss_total.extend(val_loss)
            if chunk_idx == 1:
                break
        # ----------------------------
        # Save model
        torch.save(model.state_dict(), models)
        
        # Show plots?
        plot = True
        
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
            plt.title("Training and Validation Loss for " + model_name)
            plt.savefig("reports/training_validation_loss_" + model_name + ".png")
            plt.close()
            # ----------------------------
            # Evaluate on all test sets
            evaluate_model(model, tests_to_run, device, model_name)


if __name__ == "__main__":
    main()