# Standard library imports
from pathlib import Path
import random
import copy

# Third-party imports
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

# Optimize CPU performance
torch.set_num_threads(os.cpu_count())  # Use all CPU cores
torch.set_num_interop_threads(os.cpu_count())

# Optimize GPU performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Auto-tune cuDNN algorithms
    torch.backends.cudnn.enabled = True

# Local project imports
from data import (
    load_and_split_data,
    AISTrajectorySeq2Seq,
    compute_global_norm_stats,
    find_all_parquet_files,
)
from evaluate import evaluate_model, evaluate_multiple_models
from models import Load_model
from KalmanFilterWrapper import KalmanFilterWrapper
from PlotToWorldMap import PlotToWorldMap
#from PlotToWorldMap import PlotToWorldMap

def train(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    dynamic_epochs: bool = False,
    device: str | torch.device | None = None,
):
    # Set device (GPU preferred, CPU fallback)
    if device is None:
        if torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                # Test if CUDA actually works with the model
                print(f"Attempting to use GPU: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(f"Warning: GPU detected but error occurred: {e}")
                print("Falling back to CPU for training...")
                device = torch.device("cpu")
        else:
            print("Warning: No GPU available. Using CPU for training...")
            device = torch.device("cpu")
    
    try:
        model.to(device)
    except RuntimeError as e:
        if "CUDA" in str(e) or "kernel" in str(e):
            print(f"\nError moving model to GPU: {e}")
            print("GPU is incompatible with this PyTorch version.")
            print("Falling back to CPU for training...\n")
            device = torch.device("cpu")
            model.to(device)
        else:
            raise
    
    # Loss function - focus on position (lat/lon) only
    # COG has std=20.46 which dominates loss even when normalized
    # For trajectory prediction, position accuracy is what matters most
    # Outputs: [dLat, dLon, dSOG, dCOG] - we only compute loss on first 2
    def position_mse_loss(pred, target):
        """MSE loss on position deltas (lat/lon) only, ignoring SOG/COG"""
        return torch.nn.functional.mse_loss(pred[..., :2], target[..., :2])
    
    criterion = position_mse_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # AdamW with weight decay
    
    # Learning rate scheduler - reduce LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Mixed precision training for faster GPU training
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    use_amp = device.type == 'cuda'

    # Track loss and position error
    training_loss, validation_loss = [], []
    training_pos_error, validation_pos_error = [], []  # Track lat/lon error separately
    
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

    # Create tqdm progress bar for epochs
    pbar = tqdm(range(max_epochs), desc="Training", unit="epoch")
    
    for epoch in pbar:
        # ----------------------
        # Training
        # ----------------------
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            # Batch first then transpose if needed
            batch_first = getattr(model, "batch_first", False)
            if not batch_first:
                x_batch = x_batch.transpose(0, 1)  # [T, B, F]
                y_batch = y_batch.transpose(0, 1)  # [T, B, F]

            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if use_amp:
                with torch.amp.autocast('cuda'):
                    preds = model(x_batch)
                    
                    # CRITICAL FIX: Train on ALL timesteps, not just the last one
                    # This teaches the model trajectory dynamics over the full sequence
                    if batch_first:
                        # preds: [B, T, F], y_batch: [B, T, F]
                        loss = criterion(preds.reshape(-1, preds.shape[-1]), 
                                       y_batch.reshape(-1, y_batch.shape[-1]))
                    else:
                        # preds: [T, B, F], y_batch: [T, B, F]
                        loss = criterion(preds.reshape(-1, preds.shape[-1]), 
                                       y_batch.reshape(-1, y_batch.shape[-1]))
                
                # Backward with gradient scaling
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(x_batch)
                
                # CRITICAL FIX: Train on ALL timesteps, not just the last one
                if batch_first:
                    loss = criterion(preds.reshape(-1, preds.shape[-1]), 
                                   y_batch.reshape(-1, y_batch.shape[-1]))
                else:
                    loss = criterion(preds.reshape(-1, preds.shape[-1]), 
                                   y_batch.reshape(-1, y_batch.shape[-1]))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                batch_first = getattr(model, "batch_first", False)

                if not batch_first:
                    x_batch = x_batch.transpose(0, 1)
                    y_batch = y_batch.transpose(0, 1)

                # Use mixed precision for validation
                with torch.amp.autocast('cuda', enabled=use_amp):
                    preds = model(x_batch)
                    
                    # Train on ALL timesteps for validation too
                    if batch_first:
                        loss = criterion(preds.reshape(-1, preds.shape[-1]), 
                                       y_batch.reshape(-1, y_batch.shape[-1]))
                    else:
                        loss = criterion(preds.reshape(-1, preds.shape[-1]), 
                                       y_batch.reshape(-1, y_batch.shape[-1]))
                
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        validation_loss.append(avg_val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Update tqdm progress bar with current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar.set_postfix({
            'train': f'{avg_train_loss:.6f}',
            'val': f'{avg_val_loss:.6f}',
            'lr': f'{current_lr:.1e}'
        })
        
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
                    print(
                        f"Early stopping at epoch {epoch+1} "
                        f"(no val improvement for {patience} epochs)."
                    )
                    break

    # Restore best model if early stopping was used
    if dynamic_epochs and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

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
    # Set device (GPU preferred, CPU fallback) - BEFORE creating models
    
    device = torch.device("cpu")  # Default to CPU
    
    if torch.cuda.is_available():
        try:
            test_device = torch.device("cuda")
            # Test if CUDA actually works with a simple RNN
            test_model = nn.RNN(input_size=2, hidden_size=4, num_layers=1, batch_first=False)
            test_model.to(test_device)
            test_tensor = torch.zeros(1, 1, 2).to(test_device)
            _ = test_model(test_tensor)
            # If we get here, GPU works!
            device = test_device
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            del test_model, test_tensor  # Clean up
        except Exception as e:
            print(f"✗ GPU detected but not compatible: {str(e)[:100]}")
            print("→ Falling back to CPU for training...")
            device = torch.device("cpu")
    else:
        print("✗ No GPU available. Using CPU for training...")
    
    print(f"Device: {device} ({'CPU' if device.type == 'cpu' else 'GPU'})\n")
    
    # --------------------------
    # Choose the name of the model to train
    # Options: "rnn", "lstm", "gru", "transformer", "kalman"
    #model_name = "Transformer"
    # models_list = ["rnn", "lstm", "gru", "transformer"]
    models_list = ["transformer"]
    
    trained_models = {}  # Store all trained models for comparison
    
    for model_name in models_list:
        # Look for the existing model
        if model_name == "kalman":
            models = "kalman"
        else:
            models = "models/ais_" + model_name + "_model.pth"
        
        # Inputs, Hidden, Outputs
        n_in = 4    # lat, lon, SOG, COG
        n_hid = 64 if device.type == 'cpu' else 256  # Larger hidden size for GPU
        n_out = 4   # dLat, dLon, dSOG, dCOG
        
        # Epochs and learning rate
        epochs = 200  # Shorter run for testing (~1 hour), change to 1000 for full training
        # Learning rates - lower for transformer to prevent instability
        lr = {"rnn": 5e-3, "lstm": 5e-3, "gru": 5e-3, "transformer": 1e-4}[model_name]
        print(f"Using learning rate: {lr}")
        # -------------------------
        if models == "kalman":
            print("Using Kalman Filter model...")
        else:
            print("Training new model...")
            model = Load_model.load_model(model_name, n_in, n_out, n_hid)
            
            # Try to load existing model if it exists and matches architecture
            if Path(models).exists():
                try:
                    model.load_state_dict(torch.load(models, weights_only=True))
                    print(f"Loaded existing model from {models}")
                except Exception as e:
                    print(f"Could not load existing model (architecture changed): {e}")
                    print("Training from scratch...")

        # ----------------------------
        # Find all training sequences
        
        training_sequences = []
        # Find all training sequences in the data folder
        base_folder = Path(__file__).parent.parent.parent / "data" / "Raw" / "processed"
        all_sequences = find_all_parquet_files(base_folder)
        
        # Use subset of data on CPU for faster training
        if device.type == 'cpu':
            import random
            training_sequences = random.sample(all_sequences, min(50, len(all_sequences)))  # Use only 50 files on CPU
            print(f"CPU Mode: Using subset of {len(training_sequences)}/{len(all_sequences)} sequences for faster training")
        else:
            training_sequences = all_sequences
            print("Found training sequences:", len(training_sequences))
        
        # Use all training sequences at once (no chunking)
        print(f"Training on all {len(training_sequences)} sequences simultaneously...")
        
        # ----------------------------
        # Compute global normalization stats
        global_in_mean, global_in_std, global_delta_mean, global_delta_std = compute_global_norm_stats(
            base_folder,
            train_frac=0.7,
            IN_COLS = ["Latitude", "Longitude", "SOG", "COG"],
            DELTA_COLS = ["dLatitude", "dLongitude", "dSOG", "dCOG"],
        )
        print("Global input mean:", global_in_mean)
        print("Global input std:",  global_in_std)
        print("Global delta mean:", global_delta_mean)
        print("Global delta std:",  global_delta_std)
        
        # ----------------------------
        # Load all data at once
        print("Loading all data...")
        (train_X, train_y), (val_X, val_y), (test_X, test_y) = load_and_split_data(
            input_files=training_sequences,
            in_mean=global_in_mean,
            in_std=global_in_std,
            delta_mean=global_delta_mean,
            delta_std=global_delta_std,
        )
        print("X_train.shape:", train_X.shape)
        print("y_train.shape:", train_y.shape)

        # Choose sequence length (number of timesteps per sample)
        seq_len = 50  # Changed to 50 to match trajectory_prediction in visualize.py

        # Create datasets and dataloaders
        train_dataset = AISTrajectorySeq2Seq(train_X, train_y, seq_len)
        val_dataset   = AISTrajectorySeq2Seq(val_X,   val_y,   seq_len)
        test_dataset  = AISTrajectorySeq2Seq(test_X,  test_y,  seq_len)

        # Create data loaders with optimized batch size for CPU/GPU
        batch_size = 64 if device.type == 'cpu' else 500  # 500 sequences of same length trained together
        num_workers = 0  # Disable multiprocessing for Windows to avoid overhead
        pin_memory = device.type == 'cuda'  # Pin memory for faster GPU transfer
        # CRITICAL: shuffle=True mixes data from different ships/segments in each batch
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        # Save test loader for later evaluation
        tests_to_run = [("all_data", test_loader)]
        
        if models == "kalman":
            model = KalmanFilterWrapper(dt=1.0, process_variance=1e-5, measurement_variance=0.1, init_error=1.0)
            train_loss_total = []
            val_loss_total = []
            # Validation loss for Kalman Filter
            model.eval()
            with torch.no_grad():
                val_err = 0.0
                for x_batch, y_batch in val_loader:
                    outputs = model(x_batch)
                    error = nn.MSELoss()(outputs, y_batch)
                    val_err += error.item()
                avg_val_loss = val_err / len(val_loader)
                val_loss_total.append(avg_val_loss)
        else:
            train_loss_total, val_loss_total = train(
                model,
                train_loader,
                val_loader,
                num_epochs=epochs,
                learning_rate=lr,
                dynamic_epochs=False,  # Set to False to train for full 1000 epochs
                device=device
            )
        
        # ----------------------------
        # Save model
        # Create models directory if it doesn't exist
        models_path = Path(models)
        models_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), models)
        print(f"Model saved to: {models}")
        
        # Store trained model for comparison
        trained_models[model_name] = model
        
        # Show plots?
        plot = True
        
        # ----------------------------
        # Plot training and validation loss
        # This shows under or overfitting
        if plot:
            # Create reports directory if it doesn't exist
            Path("reports").mkdir(parents=True, exist_ok=True)
            
            # Show training and validation loss
            plt.figure()
            plt.plot(train_loss_total, label="Train Loss")
            plt.plot(val_loss_total, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Weighted Loss (10x lat/lon, 1x SOG, 0.1x COG)")
            plt.legend()
            plt.title("Training and Validation Loss - " + model_name)
            plt.yscale('log')  # Log scale to see improvements better
            plt.savefig("reports/training_validation_loss_" + model_name + ".png")
            plt.close()

            # ----------------------------
            # Evaluate on all test sets
            evaluate_model(model, tests_to_run, device, model_name)
            
            # ----------------------------
            # Trajectory visualization: predict next 50 steps
            print(f"\n{'='*60}")
            print(f"Trajectory Prediction Visualization for {model_name}")
            print(f"{'='*60}")
            
            # Load a sample trajectory for visualization
            sample_paths = [
                "data/Processed/MMSI=219000617/Segment=11/513774f9fb5b4cabba2085564bb84c5c-0.parquet",
                "data/Processed/MMSI=219002906/Segment=2/513774f9fb5b4cabba2085564bb84c5c-0.parquet",
                "data/Processed/MMSI=219001258/Segment=1/513774f9fb5b4cabba2085564bb84c5c-0.parquet",
            ]
            
            for sample_path in sample_paths:
                sample_file = Path(sample_path)
                if sample_file.exists():
                    print(f"\\nVisualizing trajectory from: {sample_path}")
                    df_sample = pd.read_parquet(sample_file)
                    X_seq = torch.from_numpy(df_sample[["Latitude", "Longitude", "SOG", "COG"]].to_numpy("float32"))
                    
                    try:
                        from visualize import trajectory_prediction
                        result = trajectory_prediction(
                            model=model,
                            X_seq_raw=X_seq,
                            device=device,
                            seq_len=seq_len,
                            future_steps=50,
                            sog_cog_mode="predicted"
                        )
                        print(f"Prediction error: {result['error_m']:.1f} meters")
                        
                        # Save the figure
                        plt.savefig(f"reports/trajectory_prediction_{model_name}_{sample_file.parent.parent.name}.png")
                        print(f"Saved visualization to: reports/trajectory_prediction_{model_name}_{sample_file.parent.parent.name}.png")
                        plt.close()
                        break  # Only visualize the first available trajectory
                    except Exception as e:
                        print(f"Could not visualize trajectory: {e}")
                        continue
    
    # ----------------------------
    # Evaluate all models together and plot comparisons
    if len(trained_models) > 1:
        print("\n" + "="*60)
        print("Comparing all trained models...")
        print("="*60)
        global_stats = (global_in_mean, global_in_std, global_delta_mean, global_delta_std)
        evaluate_multiple_models(trained_models, tests_to_run, device, global_stats)