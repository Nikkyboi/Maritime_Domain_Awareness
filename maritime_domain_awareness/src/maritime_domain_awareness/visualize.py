from os import path
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import radians, cos, sin, sqrt, atan2
import torch
from .CompareModels import compare_models
from .data import find_all_parquet_files
from .KalmanTrajectoryPrediction import kalman_trajectory_prediction
from .models import Load_model
from .data import AISTrajectorySeq2Seq, load_and_split_data, compute_global_norm_stats
import pandas as pd

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
    save_plot: bool = True,   # whether to generate and save the plot
    norm_stats: tuple = None,  # optional pre-computed (in_mean, in_std, delta_mean, delta_std)
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
    
    norm_stats:
        Optional tuple of (in_mean, in_std, delta_mean, delta_std) to avoid recomputing.
        If None, will call compute_global_norm_stats() (slower).
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
    if norm_stats is not None:
        in_mean, in_std, delta_mean, delta_std = norm_stats
    else:
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
    if save_plot:
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


def main(in_path = "maritime_domain_awareness/data/Processed/MMSI=220507000/Segment=0/c7215d18afa84486a1f009dd4dd86dd8-0.parquet"):
    
  
    COMPUTE_TEST_SET_ERROR = False 

    
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
    n_hid = 256    # hidden size for RNN/LSTM/GRU/Transformer
    
    # Sequence length for training and rollout
    seq_len = 60
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_types = ["transformer", "rnn", "lstm", "gru"]
    loaded_models = {}

    print("\nLoading available models...")
    for model_type in model_types:
        model_path = f"models/ais_{model_type}_model.pth"
        if Path(model_path).exists():
            try:
                model = Load_model.load_model(model_type, n_in, n_out, n_hid)
                model.load_state_dict(torch.load(model_path, map_location=device))
                loaded_models[model_type] = model
                print(f"✓ {model_type.upper()} loaded")
            except Exception as e:
                print(f"✗ Failed to load {model_type}: {e}")
        else:
            print(f"✗ {model_type.upper()} not found")
    
    # ------------------------------------------
    
    if COMPUTE_TEST_SET_ERROR:
        
        base_folder = Path("maritime_domain_awareness/src/maritime_domain_awareness/done44")
        all_files = find_all_parquet_files(base_folder)
        print(f"Found {len(all_files)} parquet files")
        
        # Compute normalization stats ONCE (only needed for neural networks, but compute anyway)
        print("\nComputing normalization statistics...")
        norm_stats = compute_global_norm_stats()
        print("✓ Normalization stats computed")
        
        # Define time intervals to test (in minutes/steps)
        # 1 step = 1-step ahead (standard forecasting), then longer horizons
        time_intervals = [1, 5, 15, 30, 45, 60]
        
        # Store results for all models and time intervals
        all_results = {}
        
        # Add Kalman filter to the models to evaluate
        models_to_evaluate = dict(loaded_models)  # Copy neural network models
        models_to_evaluate['kalman'] = None  # Add Kalman (doesn't need a model object)
        
        # Loop over each model type (including Kalman)
        for model_name, model in models_to_evaluate.items():
                print("\n" + "="*70)
                print(f"EVALUATING MODEL: {model_name.upper()}")
                print("="*70)
                
                all_results[model_name] = {}
                
                # Loop over different time intervals
                for future_steps in time_intervals:
                    print(f"\n--- Testing {future_steps}-step prediction ---")
                    
                    # Create test sequences for this time interval
                    test_sequences = []
                    min_seq_length = seq_len + future_steps
                    
                    files_used = 0
                    for file_path in all_files:
                        df = pd.read_parquet(file_path)
                        in_cols = ["Latitude", "Longitude", "SOG", "COG"]
                        X_raw = torch.from_numpy(df[in_cols].to_numpy("float32"))
                        
                        N = len(X_raw)
                        if N < min_seq_length:
                            continue
                        
                        val_end = int(0.85 * N)
                        X_test_raw = X_raw[val_end:]
                        
                        i = 0
                        sequences_from_file = 0
                        while i + min_seq_length <= len(X_test_raw):
                            seq = X_test_raw[i:i + min_seq_length]
                            test_sequences.append(seq)
                            i += min_seq_length
                            sequences_from_file += 1
                        
                        if sequences_from_file > 0:
                            files_used += 1
                    
                    print(f"Created {len(test_sequences)} test sequences from {files_used} files")
                    
                    # Evaluate model on all test sequences
                    all_errors = []
                    for idx, seq in enumerate(test_sequences):
                        if (idx + 1) % 50 == 0:
                            print(f"  Processing sequence {idx + 1}/{len(test_sequences)}...")
                        
                        # Use appropriate prediction function based on model type
                        if model_name == 'kalman':
                            # Use Kalman filter prediction
                            result = kalman_trajectory_prediction(
                                X_seq_raw=seq,
                                seq_len=seq_len,
                                future_steps=future_steps,
                                haversine=haversine,
                            )
                        else:
                            
                            # Use neural network prediction
                            result = trajectory_prediction(
                                model=model,
                                X_seq_raw=seq,
                                device=device,
                                seq_len=seq_len,
                                future_steps=future_steps,
                                sog_cog_mode="predicted",
                                model_name="",
                                save_plot=False,
                                norm_stats=norm_stats,
                            )
                        all_errors.append(result["error_m"])
                    
                    all_errors = np.array(all_errors)
                    results = {
                        "mean_error_m": np.mean(all_errors),
                        "std_error_m": np.std(all_errors),
                        "median_error_m": np.median(all_errors),
                        "min_error_m": np.min(all_errors),
                        "max_error_m": np.max(all_errors),
                        "num_sequences": len(all_errors),
                    }
                    
                    all_results[model_name][future_steps] = results
                    
                    # Print results for this time interval
                    print(f"  Mean: {results['mean_error_m']:.1f}m | "
                          f"Std: {results['std_error_m']:.1f}m | "
                      f"Median: {results['median_error_m']:.1f}m")
        
        # Print summary comparison table
        print("\n" + "="*70)
        print("SUMMARY: MEAN ERROR (meters) BY MODEL AND TIME INTERVAL")
        print("="*70)
        print(f"{'Model':<15}", end="")
        for interval in time_intervals:
            print(f"{interval:>10} steps", end="")
        print()
        print("-"*70)
        
        for model_name in models_to_evaluate.keys():
            print(f"{model_name.upper():<15}", end="")
            for interval in time_intervals:
                mean_err = all_results[model_name][interval]["mean_error_m"]
                print(f"{mean_err:>15.1f}", end="")
            print()
        print("="*70)
    
    else:
        
        # Trajectory_prediction function
        path = in_path
        #path = "maritime_domain_awareness/data/Raw/processed/MMSI=219002906/Segment=2/513774f9fb5b4cabba2085564bb84c5c-0.parquet"
        #path = "maritime_domain_awareness/data/Raw/processed/MMSI=219001258/Segment=1/513774f9fb5b4cabba2085564bb84c5c-0.parquet"
        #path = "maritime_domain_awareness/data/Raw/processed/MMSI=219001204/Segment=0/513774f9fb5b4cabba2085564bb84c5c-0.parquet"
        # path = "data/Processed/MMSI=219018158/Segment=0/513774f9fb5b4cabba2085564bb84c5c-0.parquet"
        
        #path = "/data/Processed/MMSI=219000617/Segment=25/513774f9fb5b4cabba2085564bb84c5c-0.parquet"
        #path = "maritime_domain_awareness/data/Raw/processed/MMSI=219005931/Segment=1/513774f9fb5b4cabba2085564bb84c5c-0.parquet"
        #path = "maritime_domain_awareness/data/Raw/processed/MMSI=219005941/Segment=0/513774f9fb5b4cabba2085564bb84c5c-0.parquet"
        #path = "data/Processed/MMSI=220507000/Segment=0/c7215d18afa84486a1f009dd4dd86dd8-0.parquet"

        
        df = pd.read_parquet(path)
        X_seq = torch.from_numpy(df[["Latitude", "Longitude", "SOG", "COG"]].to_numpy("float32"))

        models_to_compare = []

        # Add all loaded neural network models with different colors
        model_colors = {
            "transformer": "orange",
            "gru": "red",
            "lstm": "purple",
            "rnn": "brown"
        }

        for model_type, model in loaded_models.items():
            models_to_compare.append({
                "name": model_type.upper(),
                "predictor": lambda X, m=model, **kw: trajectory_prediction(m, X, device, **kw),
                "kwargs": {"sog_cog_mode": "predicted"},
                "color": model_colors[model_type]
            })
        # Add Kalman filter
        models_to_compare.append({
            "name": "Kalman Filter",
            "predictor": kalman_trajectory_prediction,
            "kwargs": {"haversine": haversine},
            "color": "green"
        })

        # EXAMPLE: Add more models here if you want
        # models_to_compare.append({
        #     "name": "LSTM",
        #     "predictor": lstm_trajectory_prediction,  # your function
        #     "kwargs": {},
        #     "color": "red"
        # })

        # Run comparison
        if len(models_to_compare) > 0:
            results = compare_models(X_seq, models_to_compare, seq_len=seq_len, future_steps=50, device=device)

            # Print summary
            print("\n" + "="*50)
            print("SUMMARY OF RESULTS")
            print("="*50)
            for model_name, model_data in results.items():
                error = model_data["result"]["error_m"]
                print(f"{model_name:20s}: {error:8.1f} meters")
            print("="*50)

            # Show all plots
            plt.show()
        else:
            print("No models available to compare. Please train a model first.")


if __name__ == "__main__":
    main()