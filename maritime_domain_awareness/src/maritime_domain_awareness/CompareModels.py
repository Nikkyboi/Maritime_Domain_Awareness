import numpy as np
import torch
import matplotlib.pyplot as plt
from PlotToWorldMap import PlotToWorldMap

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