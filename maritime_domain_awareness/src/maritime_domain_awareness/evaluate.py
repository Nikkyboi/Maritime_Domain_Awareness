import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from .models import Load_model

def evaluate_model_on_sequence(model, test_loader, device):
    """
    Evaluate the model on a single test dataset.
    """
    model.to(device)
    model.eval()

    predicted_batches = []
    actual_batches = []

    total_loss = 0.0
    num_batches = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        for inputs, targets in test_loader:
            # inputs, targets: [B, T, F]
            inputs = inputs.to(device)
            targets = targets.to(device)

            raw_targets = targets  # keep batch-first

            # If the model expects time-first (T, B, F), transpose inputs
            if getattr(model, 'batch_first', False) is False:
                inputs_for_model = inputs.transpose(0, 1)  # [T, B, n_in]
            else:
                inputs_for_model = inputs                   # [B, T, n_in]

            outputs = model(inputs_for_model)

            # Normalize outputs to [B, T, F]
           if outputs.dim() == 3:
                if getattr(model, 'batch_first', False) is False:
                    outputs = outputs.transpose(0, 1)
            elif outputs.dim() == 2:
                outputs = outputs.unsqueeze(1)

            if raw_targets.dim() == 2:
                raw_targets = raw_targets.unsqueeze(1)

            if outputs.shape[1] > 1:  
                outputs_last = outputs[:, -1, :]      # [B, F]
                targets_last = raw_targets[:, -1, :]  # [B, F]
            else:
                outputs_last = outputs.squeeze(1)
                targets_last = raw_targets.squeeze(1)

            loss = criterion(outputs_last, targets_last)  
            total_loss += loss.item()
            num_batches += 1

            # store for plotting, move to CPU
            predicted_batches.append(outputs.detach().cpu())
            actual_batches.append(raw_targets.detach().cpu())

    if num_batches > 0:
        avg_loss = total_loss / num_batches
    else:
        avg_loss = float("nan")

    predicted = torch.cat(predicted_batches, dim=0)  # [N_total, T, F]
    actual    = torch.cat(actual_batches,    dim=0)

    return predicted, actual, avg_loss

def evaluate_model(model : nn.Module, tests_to_run : list, device : torch.device, name : str) -> None:
    """
    Evaluate the model on multiple test datasets.

    tests_to_run: list of (seq_path, test_loader) tuples
    """
    norm_path = f"models/norm_params_{name}.json"
    if Path(norm_path).exists():
        with open(norm_path) as f:
            norm = json.load(f)
        delta_std = np.array(norm["delta_std"])
    else:
        print(f"Warning: {norm_path} not found, showing normalized errors only")
        delta_std = None
    
    all_test_losses = []
    all_position_errors = []

    for seq, test_loader in tests_to_run:
        print(f"Evaluating on test set for sequence: {seq}")
        pred, actual, avg_loss = evaluate_model_on_sequence(model, test_loader, device)
        all_test_losses.append(avg_loss)
        
        if delta_std is not None:
            pred_np = pred.numpy()
            actual_np = actual.numpy()
            
            error = pred_np - actual_np
            dlat_error = error[..., 0] * delta_std[0]
            dlon_error = error[..., 1] * delta_std[1]
            
            lat_error_m = dlat_error * 111000
            lon_error_m = dlon_error * 111000 * np.cos(np.radians(56.0))
            
            position_error_m = np.sqrt(lat_error_m**2 + lon_error_m**2)
            all_position_errors.append(position_error_m.mean())

    plt.figure()
    plt.plot(all_test_losses, marker="o")
    plt.xlabel("Boat sequence index")
    plt.ylabel("Average Test MSE")
    plt.title("Test loss per boat sequence")
    plt.tight_layout()
    plt.savefig("reports/test_loss_per_sequence_" + name + ".png")
    plt.close()
    #plt.show()

    print(f"\nGlobal average test loss (normalized MSE): {np.mean(all_test_losses):.6f}")
    print(f"Average position error: {np.mean(all_position_errors):.1f} meters")

if __name__ == "__main__":
    # Parameters
    n_in = 4      # Latitude, Longitude, SOG, COG
    n_out = 4      # predict dLatitude, dLongitude, dSOG, dCOG
    n_hid = 256    # hidden size for RNN/LSTM/GRU/Transformer
    
    # Sequence length for training and rollout
    seq_len = 50

    # load a model and evaluate on test data
    model_name = "lstm"
    model_path = "models/ais_" + model_name + "_model.pth"
    if Path(model_path).exists():
        print("Loading existing model:")
        model = Load_model.load_model(model_name, n_in, n_out, n_hid)
        model.load_state_dict(torch.load(model_path))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else :
        print(f"Model file not found: {model_path}")

    # Load and split data
    data_path = "data/ais_data.parquet"