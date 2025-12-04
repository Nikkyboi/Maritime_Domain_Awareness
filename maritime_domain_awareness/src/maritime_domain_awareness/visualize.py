from os import path
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import radians, cos, sin, sqrt, atan2
import torch
from .models import Load_model
from .data import AISTrajectorySeq2Seq, load_and_split_data, compute_global_norm_stats
from .PlotToWorldMap import PlotToWorldMap

def haversine(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two points (in degrees).
    Returns distance in meters.
    """
    R = 6371000  # Earth radius in meters

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def wrap_360(angle_deg: float) -> float:
        """Wrap angle to [0, 360) degrees."""
        return (angle_deg % 360.0 + 360.0) % 360.0

def trajectory_prediction(
    model,
    X_seq_raw,                # torch tensor [N, 4] normalized inputs for ONE file
    device,
    seq_len: int = 50,    # context window length
    future_steps: int = 50,    # how many steps to predict ahead
    sog_cog_mode: str = "predicted",  # "predicted", "true", or "constant"
    model_name: str = "",
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
    
    # Inputs:
    - X_seq_raw is RAW data in degrees / knots / deg, not normalized.
    - The LAST `future_steps` points are the window we want to predict.
      We use the `seq_len` samples immediately BEFORE that window as context.
    - The model outputs 4 normalized deltas: [dLat, dLon, dSOG, dCOG].
    
    sog_cog_mode:
        "predicted" : use the model's dSOG, dCOG to update SOG/COG autoregressively.
        "true"      : keep the TRUE future SOG/COG from the AIS data (cheating).
        "constant"  : keep SOG/COG fixed at the last true value before predicting future values.
    """
    model.to(device)
    model.eval()

    # RAW input: [N, 4] = [lat_deg, lon_deg, SOG, COG]
    X_seq_raw = X_seq_raw.to(device)
    N = X_seq_raw.shape[0]

    # Need at least seq_len context + future_steps to predict
    if N < seq_len + future_steps:
        raise ValueError(
            f"Not enough points in sequence (N={N}) for "
            f"seq_len={seq_len} and future_steps={future_steps}."
        )

    # ----- extract normalization stats -----
    in_mean, in_std, delta_mean, delta_std = compute_global_norm_stats()
    in_mean_np = np.asarray(in_mean, dtype=np.float32)   # [4]
    in_std_np  = np.asarray(in_std,  dtype=np.float32)   # [4]

    # Now length 4: [dLat, dLon, dSOG, dCOG]
    delta_mean_np = np.asarray(delta_mean, dtype=np.float32)  # [4]
    delta_std_np  = np.asarray(delta_std,  dtype=np.float32)  # [4]

    # Tensors on device for input normalization
    in_mean_t = torch.from_numpy(in_mean_np).to(device)  # [4]
    in_std_t  = torch.from_numpy(in_std_np).to(device)   # [4]

    # ----- full TRUE lat/lon/SOG/COG in REAL units -----
    X_np_raw = X_seq_raw.detach().cpu().numpy()
    true_lat_full = X_np_raw[:, 0]  # degrees
    true_lon_full = X_np_raw[:, 1]  # degrees
    true_sog_full = X_np_raw[:, 2]  # knots
    true_cog_full = X_np_raw[:, 3]  # degrees (0..360)

    # ----- create NORMALIZED copy for the model -----
    X_in = (X_seq_raw - in_mean_t) / in_std_t    # [N, 4] normalized

    # lat/lon normalization stats (first two dimensions of input)
    latlon_mean = in_mean_np[:2]  # [lat_mean, lon_mean]
    latlon_std  = in_std_np[:2]   # [lat_std, lon_std]

    # ---- indices ----
    first_future_idx = N - future_steps   # first index we want to PREDICT
    base_idx = first_future_idx - 1       # last TRUE point before the future window

    # Store predicted lat/lon/SOG/COG per index for the full sequence
    pred_lat_full = np.full(N, np.nan, dtype=np.float64)
    pred_lon_full = np.full(N, np.nan, dtype=np.float64)
    pred_sog_full = np.full(N, np.nan, dtype=np.float64)
    pred_cog_full = np.full(N, np.nan, dtype=np.float64)
    
    # SOG/COG mode setup
    if sog_cog_mode == "constant":
        base_sog_cog_norm = X_in[base_idx, 2:4].clone()   # normalized
    elif sog_cog_mode in ("true", "predicted"):
        base_sog_cog_norm = None
    else:
        raise ValueError(
            f"Unknown sog_cog_mode: {sog_cog_mode!r} (use 'predicted', 'true', or 'constant')"
        )

    # ----- autoregressive rollout -----
    with torch.no_grad():
        for step_idx in range(first_future_idx, first_future_idx + future_steps):
            # Context window is the previous `seq_len` points:
            context_start = step_idx - seq_len
            context_end   = step_idx            # exclusive
            window = X_in[context_start:context_end]     # [seq_len, 4]
            window = window.unsqueeze(1)                 # [seq_len, 1, 4]

            out = model(window)               # [seq_len, 1, 4] normalized deltas
            delta_t_norm = out[-1, 0]         # last time step, batch 0 -> [4]

            # Denormalize deltas to REAL units
            delta_np = delta_t_norm.detach().cpu().numpy()
            dlat = float(delta_np[0] * delta_std_np[0] + delta_mean_np[0])
            dlon = float(delta_np[1] * delta_std_np[1] + delta_mean_np[1])
            dSOG = float(delta_np[2] * delta_std_np[2] + delta_mean_np[2])
            dCOG = float(delta_np[3] * delta_std_np[3] + delta_mean_np[3])

            # ----- base LAT/LON -----
            prev_idx = step_idx - 1
            if prev_idx >= first_future_idx and not np.isnan(pred_lat_full[prev_idx]):
                # use PREVIOUS PREDICTION
                base_lat = pred_lat_full[prev_idx]
                base_lon = pred_lon_full[prev_idx]
            else:
                # use TRUE lat/lon
                base_lat = true_lat_full[prev_idx]
                base_lon = true_lon_full[prev_idx]

            # Predicted position at `step_idx` (degrees)
            lat_pred = base_lat + dlat
            lon_pred = base_lon + dlon

            pred_lat_full[step_idx] = lat_pred
            pred_lon_full[step_idx] = lon_pred

            # ----- update X_in for future context: LAT/LON -----
            lat_norm = float((lat_pred - latlon_mean[0]) / latlon_std[0])
            lon_norm = float((lon_pred - latlon_mean[1]) / latlon_std[1])
            X_in[step_idx, 0] = lat_norm
            X_in[step_idx, 1] = lon_norm

            # ----- SOG/COG -----
            if sog_cog_mode == "constant":
                # keep SOG/COG fixed at base_idx value for all predicted steps
                X_in[step_idx, 2:4] = base_sog_cog_norm

            elif sog_cog_mode == "true":
                # use the true future SOG/COG (already in X_in from normalization above)
                # nothing to change: X_in[step_idx, 2:4] already holds normalized true values
                pass

            elif sog_cog_mode == "predicted":
                # base SOG/COG = previous predicted if available, else true
                if prev_idx >= first_future_idx and not np.isnan(pred_sog_full[prev_idx]):
                    base_sog = pred_sog_full[prev_idx]
                    base_cog = pred_cog_full[prev_idx]
                else:
                    base_sog = true_sog_full[prev_idx]
                    base_cog = true_cog_full[prev_idx]

                sog_pred = base_sog + dSOG
                cog_pred = wrap_360(base_cog + dCOG)

                pred_sog_full[step_idx] = sog_pred
                pred_cog_full[step_idx] = cog_pred

                # feed back normalized predicted SOG/COG
                sog_norm = float((sog_pred - in_mean_np[2]) / in_std_np[2])
                cog_norm = float((cog_pred - in_mean_np[3]) / in_std_np[3])
                X_in[step_idx, 2] = sog_norm
                X_in[step_idx, 3] = cog_norm

    # ---- collect the segment for plotting ----
    idx_range_future = np.arange(first_future_idx, first_future_idx + future_steps)

    # base true point (time t0)
    base_lat = true_lat_full[base_idx]
    base_lon = true_lon_full[base_idx]

    # future true and predicted
    true_lat_future = true_lat_full[idx_range_future]
    true_lon_future = true_lon_full[idx_range_future]
    pred_lat_future = pred_lat_full[idx_range_future]
    pred_lon_future = pred_lon_full[idx_range_future]

    # prepend base point to both true and predicted tracks
    true_lat = np.concatenate([[base_lat], true_lat_future])
    true_lon = np.concatenate([[base_lon], true_lon_future])
    pred_lat = np.concatenate([[base_lat], pred_lat_future])
    pred_lon = np.concatenate([[base_lon], pred_lon_future])

    # ----- compute final error (last predicted point) -----
    error_m = haversine(
        true_lat[-1], true_lon[-1],
        pred_lat[-1], pred_lon[-1],
    )

    past_start = first_future_idx - seq_len   # inclusive
    past_end   = base_idx                     # inclusive

    true_lat_past = true_lat_full[past_start:past_end + 1]
    true_lon_past = true_lon_full[past_start:past_end + 1]

    # ----- plot -----
    plt.figure(figsize=(7, 7))
    plt.plot(true_lon_past, true_lat_past, label=f"Previous Trajectory ({seq_len} steps)", linewidth=1, color="blue", alpha=0.5)
    plt.plot(true_lon, true_lat, label=f"True: {future_steps} steps", linewidth=2)
    plt.plot(pred_lon, pred_lat, label=f"Predicted: {future_steps} steps", linewidth=2)

    # Start marker: base point (where we start predicting from)
    plt.scatter(true_lon[0], true_lat[0],
                marker="o", color="green", s=80, label="Start")

    # Final markers
    plt.scatter(true_lon[-1], true_lat[-1],
                marker="x", color="blue",  s=80, label="End (True)")
    plt.scatter(pred_lon[-1], pred_lat[-1],
                marker="x", color="red",   s=80, label="End (Predicted)")

    # Line between true end and predicted end
    plt.plot(
        [true_lon[-1], pred_lon[-1]],
        [true_lat[-1], pred_lat[-1]],
        color="purple", linestyle="--", linewidth=1.5, label="Final error"
    )

    # Distance label next to line
    mid_lon = (true_lon[-1] + pred_lon[-1]) / 2
    mid_lat = (true_lat[-1] + pred_lat[-1]) / 2

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

    dt_minutes = 1  # adjust if AIS sampling is not 1 minute
    future_minutes = future_steps * dt_minutes

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(
        f"Predicting {future_steps} steps from last known point "
        f"(error = {error_m:.1f} m, steps = {future_minutes} min)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/plot_" + model_name + "2.png")
    #plt.show()

    return {
        "true_lat": true_lat,
        "true_lon": true_lon,
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "error_m": error_m,
    }



if __name__ == "__main__":
    # Example usage of haversine function
    lat1, lon1 = 52.2296756, 21.0122287  # Warsaw
    lat2, lon2 = 41.8919300, 12.5113300  # Rome

    distance = haversine(lat1, lon1, lat2, lon2)
    print(f"Distance between Warsaw and Rome: {distance:.2f} meters")
    
    # ------ Load a trained model (example) ------
    # Parameters
    n_in = 4      # Latitude, Longitude, SOG, COG
    n_out = 4     # predict dLatitude, dLongitude
    n_hid = 256    # hidden size for RNN/LSTM/GRU/Transformer
    
    # Sequence length for training and rollout
    seq_len = 50

    # load a model and evaluate on test data
    model_name = "rnn"
    model_path = "models/ais_" + model_name + "_model.pth"
    if Path(model_path).exists():
        print("Loading existing model:")
        model = Load_model.load_model(model_name, n_in, n_out, n_hid)
        model.load_state_dict(torch.load(model_path))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else :
        print(f"Model file not found: {model_path}")
    
    # ------------------------------------------
    
    # Trajectory_prediction function
    #path = "data/Processed/MMSI=219002906/Segment=2/513774f9fb5b4cabba2085564bb84c5c-0.parquet"
    #path = "data/Processed/MMSI=219001258/Segment=1/513774f9fb5b4cabba2085564bb84c5c-0.parquet"
    #path = "data/Processed/MMSI=219001204/Segment=0/513774f9fb5b4cabba2085564bb84c5c-0.parquet"
    path = "data/Processed/MMSI=219000617/Segment=11/513774f9fb5b4cabba2085564bb84c5c-0.parquet"
    #path = "data/Processed/MMSI=219000617/Segment=25/513774f9fb5b4cabba2085564bb84c5c-0.parquet"
    #path = "data/Processed/MMSI=219005931/Segment=1/513774f9fb5b4cabba2085564bb84c5c-0.parquet"
    #path = "data/Processed/MMSI=219005941/Segment=0/513774f9fb5b4cabba2085564bb84c5c-0.parquet"
    
    df = pd.read_parquet(path)
    X_seq = torch.from_numpy(df[["Latitude", "Longitude", "SOG", "COG"]].to_numpy("float32"))

    true_lat, true_lon, pred_lat, pred_lon, error_m = trajectory_prediction(model, X_seq, device, seq_len=50, future_steps=50, sog_cog_mode="predicted",model_name=model_name)
    
    # Plot on world map
    #actualPoint = [torch.tensor(np.vstack((true_lat, true_lon)).T)]
    #predictedPoint = [torch.tensor(np.vstack((pred_lat, pred_lon)).T)]
    #PlotToWorldMap([true_lat,true_lon], [pred_lat,pred_lon])