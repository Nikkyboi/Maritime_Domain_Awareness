import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from .models import Load_model
from .visualize import haversine
from .data import AISTrajectorySeq2Seq, load_and_split_data, compute_global_norm_stats

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
                    outputs = outputs.transpose(0, 1)      # [B, T, F]
            elif outputs.dim() == 2:
                outputs = outputs.unsqueeze(1)             # [B, 1, F]

            if raw_targets.dim() == 2:
                raw_targets = raw_targets.unsqueeze(1)      # [B, 1, F]

            loss = criterion(outputs, raw_targets)
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

def evaluate_model(model : nn.Module, tests_to_run : list, device : torch.device) -> None:
    """
    Evaluate the model on multiple test datasets.

    tests_to_run: list of (seq_path, test_loader) tuples
    """
    all_test_losses = []

    for seq, test_loader in tests_to_run:
        print(f"Evaluating on test set for sequence: {seq}")
        _, _, avg_loss = evaluate_model_on_sequence(model, test_loader, device)
        all_test_losses.append(avg_loss)

    plt.figure()
    plt.plot(all_test_losses, marker="o")
    plt.xlabel("Boat sequence index")
    plt.ylabel("Average Test MSE")
    plt.title("Test loss per boat sequence")
    plt.tight_layout()
    plt.show()

    global_avg = sum(all_test_losses) / len(all_test_losses)
    print(f"Global average test loss over {len(all_test_losses)} sequences: {global_avg:.6f}")


def trajectory_prediction(
    model,
    X_seq,                # torch tensor [N, 4] normalized inputs for ONE file
    in_mean,
    in_std,
    delta_mean,
    delta_std,
    device,
    seq_len: int = 50,
):
    """
    Perform an autoregressive rollout over a full AIS sequence.

    It simply:
    - starts from the first seq_len positions
    for seq = 50 it has x[0..49]
    
    - based on the x[0..49] it predicts delta lat/lon at time 50
    - it adds the predicted delta to the last known position x[49] to get position at time 50
    - it appends the new position to the context window, removes the oldest position
      so now the context window is x[1..50]
    - it repeats this up until x[49] thuss x[50 .. 99] are pure predictions
    """
    model.to(device)
    model.eval()
    
    X_seq = X_seq.to(device)
    N = X_seq.shape[0]

    # ----- denormalize true positions for plotting -----
    latlon_mean = np.asarray(in_mean[:2], dtype=np.float32)
    latlon_std  = np.asarray(in_std[:2],  dtype=np.float32)

    X_np = X_seq.cpu().numpy()
    true_lat = X_np[:, 0] * latlon_std[0] + latlon_mean[0]
    true_lon = X_np[:, 1] * latlon_std[1] + latlon_mean[1]

    # Delta stats for denormalization
    delta_mean = np.asarray(delta_mean, dtype=np.float32)   # [2]
    delta_std  = np.asarray(delta_std,  dtype=np.float32)   # [2]

    # ----- autoregressive rollout -----
    # store predicted deltas here
    preds = []

    with torch.no_grad():
        for t in range(seq_len, N):
            # context window [t-seq_len .. t-1]
            window = X_seq[t - seq_len : t]         # [T, 4]
            window = window.unsqueeze(1)            # [T, 1, 4] (batch_first = False)

            out = model(window)                     # [T, 1, 2] (normalized deltas)
            delta_t = out[-1, 0]                    # last time step, batch 0 -> [2]

            preds.append(delta_t.cpu().numpy())

    preds = np.stack(preds, axis=0)                # [N - seq_len, 2]
    pred_delta = preds * delta_std + delta_mean    # denormalize deltas (back to lat/lon space)

    # ----- reconstruct predicted positions -----
    # start from the true position at time seq_len-1
    pred_lat = [true_lat[seq_len - 1]]
    pred_lon = [true_lon[seq_len - 1]]

    for dlat, dlon in pred_delta:
        pred_lat.append(pred_lat[-1] + dlat)
        pred_lon.append(pred_lon[-1] + dlon)

    pred_lat = np.array(pred_lat)
    pred_lon = np.array(pred_lon)

    # align true segment to same time span
    true_lat_seg = true_lat[seq_len - 1 : seq_len - 1 + len(pred_lat)]
    true_lon_seg = true_lon[seq_len - 1 : seq_len - 1 + len(pred_lon)]

    # ----- plot one actual vs one predicted trajectory -----
    plt.figure(figsize=(7, 7))

    # Trajectories
    plt.plot(true_lon_seg, true_lat_seg, label="Actual", linewidth=1)
    plt.plot(pred_lon,     pred_lat,     label="Predicted", linewidth=1)

    # Start marker
    plt.scatter(true_lon_seg[0], true_lat_seg[0],
                marker="o", color="green", s=80, label="Start")

    # Final markers
    plt.scatter(true_lon_seg[-1], true_lat_seg[-1],
                marker="x", color="blue",  s=80, label="End (Actual)")
    plt.scatter(pred_lon[-1],     pred_lat[-1],
                marker="x", color="red",   s=80, label="End (Predicted)")

    # Line between actual end and predicted end
    plt.plot(
        [true_lon_seg[-1], pred_lon[-1]],
        [true_lat_seg[-1], pred_lat[-1]],
        color="purple", linestyle="--", linewidth=1.5, label="Final error"
    )

    # Compute haversine distance (meters)
    error_m = haversine(
        true_lat_seg[-1], true_lon_seg[-1],
        pred_lat[-1],     pred_lon[-1]
    )

    # ---- add text with distance next to the line ----
    mid_lon = (true_lon_seg[-1] + pred_lon[-1]) / 2
    mid_lat = (true_lat_seg[-1] + pred_lat[-1]) / 2

    plt.text(
        mid_lon,
        mid_lat,
        f"{error_m:.1f} m",
        ha="center",
        va="bottom",
        fontsize=10,
        color="purple",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="purple", alpha=0.8),
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Trajectory Prediction ({seq_len} min)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parameters
    n_in = 4      # Latitude, Longitude, SOG, COG
    n_out = 2     # predict dLatitude, dLongitude
    n_hid = 64    # hidden size for RNN/LSTM/GRU/Transformer
    
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
    
    # Find all training sequences    
    base_folder = Path("data/Processed/")
    from .data import find_all_parquet_files
    training_sequences = find_all_parquet_files(base_folder)
    
    # Evaluate the biggest error sequences
    global_in_mean, global_in_std, global_delta_mean, global_delta_std = compute_global_norm_stats(
            base_folder,
            train_frac=0.7,
            IN_COLS = ["Latitude", "Longitude", "SOG", "COG"],
            DELTA_COLS = ["dLatitude", "dLongitude"],
        )
    
    each_sequence_losses = []
    sequence_paths = []
    for seq_path in training_sequences:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_split_data(
            seq_path,
            in_mean=global_in_mean,
            in_std=global_in_std,
            delta_mean=global_delta_mean,
            delta_std=global_delta_std,
        )
        test_dataset = AISTrajectorySeq2Seq(X_test, y_test, seq_len)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        _, _, avg_loss = evaluate_model_on_sequence(model, test_loader, device)
        each_sequence_losses.append(avg_loss)
        sequence_paths.append(seq_path)
    
    # Get top 5 worst sequences
    each_sequence_losses = np.array(each_sequence_losses)
    worst_indices = np.argsort(-each_sequence_losses)[:5]
    tests_to_run = []
    for idx in worst_indices:
        seq_path = sequence_paths[idx]
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_split_data(
            seq_path,
            in_mean=global_in_mean,
            in_std=global_in_std,
            delta_mean=global_delta_mean,
            delta_std=global_delta_std,
        )
        test_dataset = AISTrajectorySeq2Seq(X_test, y_test, seq_len)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        tests_to_run.append((seq_path, test_loader))
        
    # Load one of the worst sequences and do trajectory prediction
    test_file = sequence_paths[worst_indices[0]]
    base_folder = Path("data/Processed/")
    global_in_mean, global_in_std, global_delta_mean, global_delta_std = compute_global_norm_stats(
            base_folder,
            train_frac=0.7,
            IN_COLS = ["Latitude", "Longitude", "SOG", "COG"],
            DELTA_COLS = ["dLatitude", "dLongitude"],
        )
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_split_data(
        test_file,
        in_mean=global_in_mean,
        in_std=global_in_std,
        delta_mean=global_delta_mean,
        delta_std=global_delta_std,
    )
    
    test_dataset = AISTrajectorySeq2Seq(X_test, y_test, seq_len)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    tests_to_run = [(Path(test_file), test_loader)]
    trajectory_prediction(
        model,
        X_seq=torch.tensor(X_test, dtype=torch.float32),
        in_mean=global_in_mean,
        in_std=global_in_std,
        delta_mean=global_delta_mean,
        delta_std=global_delta_std,
        device=device,
        seq_len=seq_len,
    )