import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
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


def evaluate_model(
    model,
    tests_to_run,
    device,
    in_mean=None,
    in_std=None,
    delta_mean=None,
    delta_std=None,):
    """
    Evaluate the model on multiple test datasets.

    tests_to_run: list of (seq_path, test_loader) tuples
    """
    
    # Input stats for Lat/Lon (first two entries of in_mean/std)
    latlon_mean = np.asarray(in_mean[:2], dtype=np.float32)
    latlon_std  = np.asarray(in_std[:2],  dtype=np.float32)

    # Delta stats for [dLatitude, dLongitude]
    delta_mean = np.asarray(delta_mean, dtype=np.float32)
    delta_std  = np.asarray(delta_std,  dtype=np.float32)
    
    all_test_losses = []

    for seq, test_loader in tests_to_run:
        print(f"Evaluating on test set for sequence: {seq}")
        predictedPoint, actualPoint, avg_loss = evaluate_model_on_sequence(model, test_loader, device)
        all_test_losses.append(avg_loss)
        print(f"  -> Test MSE: {avg_loss:.6f}")

        # ------------------------
        # Plot actual vs predicted for THIS boat
        # ------------------------
        # Shape: [B_total, T, n_out] where n_out = 2 (Latitude, Longitude)
        pred_np = predictedPoint.detach().cpu().numpy()  # [B, T, 2]
        act_np  = actualPoint.detach().cpu().numpy()

        # Denormalize
        pred_denorm = pred_np * latlon_std + latlon_mean   # [B, T, 2]
        act_denorm  = act_np  * latlon_std + latlon_mean   # [B, T, 2]

        # Split into lat/lon, keep batch + time dimensions
        pred_lat = pred_denorm[..., 0]   # [B, T]
        pred_lon = pred_denorm[..., 1]
        true_lat = act_denorm[..., 0]
        true_lon = act_denorm[..., 1]

        B, T = pred_lat.shape
        plot = False
        if plot:
            fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True, sharey=True)

            for b in range(B):
                # plot actual segment b
                if b == 0:
                    axes[0].plot(true_lon[b], true_lat[b], linewidth=1, alpha=0.6, label="Actual")
                else:
                    axes[0].plot(true_lon[b], true_lat[b], linewidth=1, alpha=0.6)

                # mark start (circle) and end (cross) for actual
                axes[0].scatter(true_lon[b, 0],   true_lat[b, 0],   marker="o", s=20, color="green",
                                label="Actual start" if b == 0 else None)
                axes[0].scatter(true_lon[b, -1],  true_lat[b, -1],  marker="x", s=20, color="red",
                                label="Actual end" if b == 0 else None)

                # plot predicted segment b
                if b == 0:
                    axes[1].plot(pred_lon[b], pred_lat[b], linewidth=1, alpha=0.6, label="Predicted")
                else:
                    axes[1].plot(pred_lon[b], pred_lat[b], linewidth=1, alpha=0.6)

                # mark start (circle) and end (cross) for predicted
                axes[1].scatter(pred_lon[b, 0],   pred_lat[b, 0],   marker="o", s=20, color="green",
                                label="Pred start" if b == 0 else None)
                axes[1].scatter(pred_lon[b, -1],  pred_lat[b, -1],  marker="x", s=20, color="red",
                                label="Pred end" if b == 0 else None)

            axes[0].set_ylabel("Latitude")
            axes[0].set_title(f"Boat sequence: {seq.parent.name}/{seq.name} - Actual")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].set_xlabel("Longitude")
            axes[1].set_ylabel("Latitude")
            axes[1].set_title(f"Boat sequence: {seq.parent.name}/{seq.name} - Predicted")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            plt.tight_layout()
            plt.show()
        
        
    # ----------------------------
    # Plot test loss per sequence + global average
    # ----------------------------
    if all_test_losses:
        plt.figure()
        plt.plot(all_test_losses, marker="o")
        plt.xlabel("Boat sequence index")
        plt.ylabel("Average Test MSE")
        plt.title("Test loss per boat sequence")
        plt.tight_layout()
        plt.show()

        global_avg = sum(all_test_losses) / len(all_test_losses)
        print(f"Global average test loss over {len(all_test_losses)} sequences: {global_avg:.6f}")

def rollout_full_sequence(
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
    Use the model to roll forward along one full AIS sequence.

    At each time t >= seq_len, use the last `seq_len` points to predict
    the delta for the NEXT point. Then reconstruct one continuous
    predicted trajectory and plot it against the actual.
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

    # delta stats
    delta_mean = np.asarray(delta_mean, dtype=np.float32)   # [2]
    delta_std  = np.asarray(delta_std,  dtype=np.float32)   # [2]

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
    pred_delta = preds * delta_std + delta_mean    # denormalize deltas

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
    plt.figure(figsize=(6, 6))
    plt.plot(true_lon_seg, true_lat_seg, label="Actual", linewidth=1)
    plt.plot(pred_lon,     pred_lat,     label="Predicted", linewidth=1)
    plt.scatter(true_lon_seg[0], true_lat_seg[0], marker="o", color="green", label="Start")
    plt.scatter(true_lon_seg[-1], true_lat_seg[-1], marker="x", color="red", label="End")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Full-sequence rollout (one continuous trajectory)")
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
        
    # Load a random test dataset and evaluate
    test_file = "data/Processed/MMSI=219005901/Segment=0/513774f9fb5b4cabba2085564bb84c5c-0.parquet"
    from .data import AISTrajectorySeq2Seq, load_and_split_data, compute_global_norm_stats
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
    rollout_full_sequence(
        model,
        X_seq=torch.tensor(X_test, dtype=torch.float32),
        in_mean=global_in_mean,
        in_std=global_in_std,
        delta_mean=global_delta_mean,
        delta_std=global_delta_std,
        device=device,
        seq_len=seq_len,
    )