from os import path
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import radians, cos, sin, sqrt, atan2
import torch
from models import Load_model
from data import compute_global_norm_stats
from PlotToWorldMap import PlotToWorldMap
from models.KalmanFilter import KalmanFilter

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

def kalman_trajectory_prediction(
    X_seq_raw,
    seq_len: int = 50,
    future_steps: int = 50,
    dt: float = 60.0,  # 1 minute in seconds
):
    """
    Perform Kalman filter prediction for trajectory forecasting.
    Similar interface to trajectory_prediction but uses Kalman filter.
    
    Args:
        X_seq_raw: torch tensor [N, 4] with [Lat, Lon, SOG, COG] in raw units
        seq_len: context window (not heavily used by Kalman, but kept for consistency)
        future_steps: how many steps to predict ahead
        dt: time step in seconds (default 60s = 1 minute)
    
    Returns:
        dict with predicted lat/lon arrays
    """
    X_np_raw = X_seq_raw.cpu().numpy()
    N = X_np_raw.shape[0]
    
    if N < seq_len + future_steps:
        raise ValueError(f"Not enough points in sequence (N={N})")
    
    # Earth radius for lat/lon to meters conversion (approximate)
    R_earth = 6371000  # meters
    lat_to_m = R_earth * np.pi / 180.0
    
    # Indices
    first_future_idx = N - future_steps
    base_idx = first_future_idx - 1
    
    # Convert lat/lon to meters (approximate local projection)
    # Use mean latitude for longitude conversion
    mean_lat = np.mean(X_np_raw[:first_future_idx, 0])
    lon_to_m = lat_to_m * np.cos(np.radians(mean_lat))
    
    # Initialize Kalman filter with first observation
    lat0, lon0 = X_np_raw[0, 0], X_np_raw[0, 1]
    x0_m = lon0 * lon_to_m  # east in meters
    y0_m = lat0 * lat_to_m  # north in meters
    
    df_init = {'x': x0_m, 'y': y0_m, 'v_e': 0.0, 'v_n': 0.0}
    kf = KalmanFilter(
        df=df_init,
        dt=dt,
        process_variance=1e-3,  # tune this for smoothness
        measurement_variance=10.0,  # tune based on GPS accuracy
        init_error=100.0
    )
    kf.x_est = np.array([[x0_m], [y0_m], [0.0], [0.0]])
    
    # Run filter through all historical data up to prediction point
    for t in range(first_future_idx):
        lat, lon = X_np_raw[t, 0], X_np_raw[t, 1]
        x_m = lon * lon_to_m
        y_m = lat * lat_to_m
        z = np.array([x_m, y_m])
        kf.step(z)
    
    # Now predict future without measurements (dead reckoning)
    pred_lat_future = []
    pred_lon_future = []
    
    for step in range(future_steps):
        # Predict next state
        kf.predict()
        # Use prediction as next estimate (no measurement update)
        kf.x_est = kf.x_pred.copy()
        kf.P = kf.P_pred.copy()
        
        # Convert back to lat/lon
        x_m, y_m = kf.x_est[0, 0], kf.x_est[1, 0]
        lon_pred = x_m / lon_to_m
        lat_pred = y_m / lat_to_m
        
        pred_lat_future.append(lat_pred)
        pred_lon_future.append(lon_pred)
    
    # Get base point and true future
    base_lat = X_np_raw[base_idx, 0]
    base_lon = X_np_raw[base_idx, 1]
    
    true_lat_future = X_np_raw[first_future_idx:first_future_idx + future_steps, 0]
    true_lon_future = X_np_raw[first_future_idx:first_future_idx + future_steps, 1]
    
    # Prepend base point
    pred_lat = np.concatenate([[base_lat], pred_lat_future])
    pred_lon = np.concatenate([[base_lon], pred_lon_future])
    true_lat = np.concatenate([[base_lat], true_lat_future])
    true_lon = np.concatenate([[base_lon], true_lon_future])
    
    # Compute error
    error_m = haversine(true_lat[-1], true_lon[-1], pred_lat[-1], pred_lon[-1])
    
    return {
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "true_lat": true_lat,
        "true_lon": true_lon,
        "error_m": error_m,
    }

def trajectory_prediction(
    model,
    X_seq_raw,                # torch tensor [N, 4] normalized inputs for ONE file
    device,
    seq_len: int = 50,    # context window length
    future_steps: int = 50,    # how many steps to predict ahead
    sog_cog_mode: str = "predicted",  # "predicted", "true", or "constant"
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
    # Don't call plt.show() to avoid blocking - let caller handle figure display/save
    
    # Also create world map visualization comparing predicted vs actual
    # Convert to torch tensors in [N, 2] format (lat, lon)
    actual_trajectory = torch.tensor(np.column_stack((true_lat, true_lon)), dtype=torch.float32)
    predicted_trajectory = torch.tensor(np.column_stack((pred_lat, pred_lon)), dtype=torch.float32)
    
    # Create dictionary for multiple model predictions format
    model_predictions = {
        "Predicted": predicted_trajectory
    }
    model_names = ["Predicted"]
    
    # Plot on world map
    PlotToWorldMap(actual_trajectory, model_predictions, model_names)

    return {
        "true_lat": true_lat,
        "true_lon": true_lon,
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "error_m": error_m,
    }



def compare_models(X_seq, models_to_compare, seq_len=50, future_steps=50, device=None):
    """
    Compare multiple trajectory prediction models.
    
    Args:
        X_seq: torch tensor [N, 4] with [Lat, Lon, SOG, COG] in raw units
        models_to_compare: list of dicts with format:
            [
                {
                    "name": "Model Name",
                    "predictor": function that takes (X_seq, ...) and returns result dict,
                    "kwargs": dict of additional arguments for the predictor,
                    "color": matplotlib color string (optional, auto-assigned if not provided)
                },
                ...
            ]
        seq_len: context window length
        future_steps: number of steps to predict
        device: torch device (for neural network models)
    
    Returns:
        dict with all model results
    """
    # Default color cycle for models (extended color palette)
    default_colors = ['orange', 'green', 'red', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'olive', 'navy']
    
    # Run all model predictions
    results = {}
    print(f"\nRunning predictions for {len(models_to_compare)} models...\n")
    
    for i, model_config in enumerate(models_to_compare):
        model_name = model_config["name"]
        predictor = model_config["predictor"]
        kwargs = model_config.get("kwargs", {})
        
        # Assign color if not provided
        if "color" not in model_config:
            model_config["color"] = default_colors[i % len(default_colors)]
        
        print(f"Running {model_name}...")
        result = predictor(X_seq, seq_len=seq_len, future_steps=future_steps, **kwargs)
        results[model_name] = {
            "result": result,
            "color": model_config["color"]
        }
        print(f"{model_name} error: {result['error_m']:.1f} meters")
    
    # Get past trajectory for context (same for all models)
    N = X_seq.shape[0]
    first_future_idx = N - future_steps
    past_start = first_future_idx - seq_len
    past_end = first_future_idx - 1
    X_np = X_seq.numpy() if isinstance(X_seq, torch.Tensor) else X_seq
    true_lat_past = X_np[past_start:past_end + 1, 0]
    true_lon_past = X_np[past_start:past_end + 1, 1]
    
    # Create comparison plot
    plt.figure(figsize=(12, 10))
    
    # Plot past trajectory
    plt.plot(true_lon_past, true_lat_past, label=f"Previous Trajectory ({seq_len} steps)", 
             linewidth=1, color="gray", alpha=0.5)
    
    # Get true trajectory (same from any model result)
    first_result = next(iter(results.values()))["result"]
    true_lat = first_result["true_lat"]
    true_lon = first_result["true_lon"]
    
    # Plot true future
    plt.plot(true_lon, true_lat, label="True: 50 steps", 
             linewidth=2.5, color="blue", zorder=3)
    
    # Plot all model predictions
    for model_name, model_data in results.items():
        result = model_data["result"]
        color = model_data["color"]
        
        plt.plot(result['pred_lon'], result['pred_lat'], 
                label=f"{model_name}: {result['error_m']:.0f}m error", 
                linewidth=2, color=color, linestyle='--', zorder=2)
        
        # Plot end marker for this model
        plt.scatter(result['pred_lon'][-1], result['pred_lat'][-1],
                   marker="X", color=color, s=100, 
                   label=f"End ({model_name})", zorder=4)
    
    # Markers for start and true end
    plt.scatter(true_lon[0], true_lat[0],
                marker="o", color="darkgreen", s=120, label="Start", zorder=5, edgecolors='black', linewidths=2)
    plt.scatter(true_lon[-1], true_lat[-1],
                marker="X", color="blue", s=120, label="End (True)", zorder=5, edgecolors='black', linewidths=2)
    
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    
    # Create title with all model names
    model_names_str = " vs ".join(results.keys())
    plt.title(f"Maritime Trajectory Prediction: {model_names_str}", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Also create world map with all predictions
    actual_trajectory = torch.tensor(np.column_stack((true_lat, true_lon)), dtype=torch.float32)
    model_predictions = {}
    model_names_list = []
    
    for model_name, model_data in results.items():
        result = model_data["result"]
        model_predictions[model_name] = torch.tensor(
            np.column_stack((result['pred_lat'], result['pred_lon'])), 
            dtype=torch.float32
        )
        model_names_list.append(model_name)
    
    PlotToWorldMap(actual_trajectory, model_predictions, model_names_list)
    
    return results


if __name__ == "__main__":
    # Example usage of haversine function
    lat1, lon1 = 52.2296756, 21.0122287  # Warsaw
    lat2, lon2 = 41.8919300, 12.5113300  # Rome

    distance = haversine(lat1, lon1, lat2, lon2)
    print(f"Distance between Warsaw and Rome: {distance:.2f} meters")
    
    # ------ Load trained models ------
    # Parameters
    n_in = 4      # Latitude, Longitude, SOG, COG
    n_out = 4     # predict dLatitude, dLongitude, dSOG, dCOG
    n_hid = 256    # hidden size - must match the trained model (256 for GPU training)
    
    # Sequence length for training and rollout
    seq_len = 50   # Changed to 50 to match training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all available models
    model_types = ["transformer", "gru", "lstm", "rnn"]
    loaded_models = {}
    
    print("\nLoading available models...")
    for model_type in model_types:
        model_path = f"models/ais_{model_type}_model.pth"
        if Path(model_path).exists():
            try:
                model = Load_model.load_model(model_type, n_in, n_out, n_hid)
                model.load_state_dict(torch.load(model_path))
                loaded_models[model_type] = model
                print(f"✓ {model_type.upper()} loaded")
            except Exception as e:
                print(f"✗ Failed to load {model_type}: {e}")
        else:
            print(f"✗ {model_type.upper()} not found")
    
    # ------------------------------------------
    
    # Load test trajectory data
    data_base = Path(__file__).parent.parent.parent / "data" / "Raw" / "processed"
    path = data_base / "MMSI=219000617/Segment=11/513774f9fb5b4cabba2085564bb84c5c-0.parquet"
    
    df = pd.read_parquet(path)
    X_seq = torch.from_numpy(df[["Latitude", "Longitude", "SOG", "COG"]].to_numpy("float32"))

    # ------ Configure models to compare ------
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
        "kwargs": {},
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