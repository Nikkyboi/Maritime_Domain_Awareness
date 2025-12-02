import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from pathlib import Path
from models import Load_model
from PlotToWorldMap import PlotToWorldMap

def evaluate_model_on_sequence(model, test_loader, device, return_inputs=False):
    """
    Evaluate the model on a single test dataset.
    """
    model.to(device)
    model.eval()

    predicted_batches = []
    actual_batches = []
    input_batches = []

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
            if return_inputs:
                input_batches.append(inputs.detach().cpu())

    if num_batches > 0:
        avg_loss = total_loss / num_batches
    else:
        avg_loss = float("nan")

    predicted = torch.cat(predicted_batches, dim=0)  # [N_total, T, F]
    actual    = torch.cat(actual_batches,    dim=0)

    if return_inputs:
        inputs_all = torch.cat(input_batches, dim=0)
        return predicted, actual, avg_loss, inputs_all
    
    return predicted, actual, avg_loss

def evaluate_multiple_models(models_dict, tests_to_run, device, global_stats=None):
    """
    Evaluate multiple models on test datasets and plot all predictions vs actual.
    
    Args:
        models_dict: Dictionary of {model_name: model} to evaluate
        tests_to_run: list of (seq_name, test_loader) tuples
        device: torch device for computation
        global_stats: tuple of (in_mean, in_std, delta_mean, delta_std) for denormalization
    """
    import numpy as np
    from data import compute_global_norm_stats
    
    # Get normalization stats if not provided
    if global_stats is None:
        in_mean, in_std, delta_mean, delta_std = compute_global_norm_stats()
    else:
        in_mean, in_std, delta_mean, delta_std = global_stats
    
    all_predictions = {}
    all_actuals = {}
    all_inputs = {}
    
    for model_name, model in models_dict.items():
        print(f"\nEvaluating {model_name}...")
        model_predictions = []
        model_actuals = []
        model_inputs = []
        
        for seq_name, test_loader in tests_to_run:
            predicted, actual, avg_loss, inputs = evaluate_model_on_sequence(model, test_loader, device, return_inputs=True)
            model_predictions.append(predicted)
            model_actuals.append(actual)
            model_inputs.append(inputs)
            print(f"  {seq_name}: Test Loss = {avg_loss:.6f}")
        
        all_predictions[model_name] = torch.cat(model_predictions, dim=0)
        all_inputs[model_name] = torch.cat(model_inputs, dim=0)
        if 'actual' not in all_actuals:
            all_actuals['actual'] = torch.cat(model_actuals, dim=0)
    
    # Reconstruct trajectories from deltas
    # Take first sequence from first model as example
    first_model = list(models_dict.keys())[0]
    inputs = all_inputs[first_model][:100]  # Take first 100 sequences
    actual_deltas = all_actuals['actual'][:100]  # [N, T, 4] - deltas
    
    # Denormalize inputs to get initial positions
    inputs_np = inputs.numpy()  # [N, T, 4]
    in_mean_np = np.array(in_mean)
    in_std_np = np.array(in_std)
    delta_mean_np = np.array(delta_mean)
    delta_std_np = np.array(delta_std)
    
    # Denormalize inputs: actual = normalized * std + mean
    inputs_denorm = inputs_np * in_std_np + in_mean_np  # [N, T, 4] - [lat, lon, SOG, COG]
    
    # Get initial position (last position in input sequence)
    initial_positions = inputs_denorm[:, -1, :2]  # [N, 2] - [lat, lon]
    
    # Denormalize actual deltas
    actual_deltas_np = actual_deltas.numpy()  # [N, T, 4]
    actual_deltas_denorm = actual_deltas_np * delta_std_np + delta_mean_np  # Real deltas
    
    # Reconstruct actual trajectory
    actual_trajectories = []
    for i in range(len(initial_positions)):
        traj = [initial_positions[i]]  # Start with initial position
        current_pos = initial_positions[i].copy()
        for t in range(actual_deltas_denorm.shape[1]):
            current_pos = current_pos + actual_deltas_denorm[i, t, :2]  # Add lat/lon deltas
            traj.append(current_pos.copy())
        actual_trajectories.append(np.array(traj))
    
    # Reconstruct predicted trajectories for each model
    model_trajectories = {}
    for model_name, preds in all_predictions.items():
        preds_np = preds[:100].numpy()  # [N, T, 4]
        preds_denorm = preds_np * delta_std_np + delta_mean_np
        
        pred_trajectories = []
        for i in range(len(initial_positions)):
            traj = [initial_positions[i]]  # Start with same initial position
            current_pos = initial_positions[i].copy()
            for t in range(preds_denorm.shape[1]):
                current_pos = current_pos + preds_denorm[i, t, :2]  # Add predicted deltas
                traj.append(current_pos.copy())
            pred_trajectories.append(np.array(traj))
        
        model_trajectories[model_name] = pred_trajectories
    
    # Plot first trajectory as example
    if len(actual_trajectories) > 0:
        print("\nGenerating comparison plot for first trajectory...")
        actual_traj = actual_trajectories[0]  # [T+1, 2]
        
        model_preds = {}
        for model_name, trajs in model_trajectories.items():
            model_preds[model_name] = trajs[0]  # [T+1, 2]
        
        PlotToWorldMap(actualPoint=actual_traj, model_predictions=model_preds)
    
    return all_predictions, all_actuals

def evaluate_model(model : nn.Module, tests_to_run : list, device : torch.device, name : str) -> None:
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
    plt.savefig("reports/test_loss_per_sequence_" + name + ".png")
    plt.close()
    #plt.show()

    global_avg = sum(all_test_losses) / len(all_test_losses)
    print(f"Global average test loss over {len(all_test_losses)} sequences: {global_avg:.6f}")

if __name__ == "__main__":
    # Parameters
    n_in = 4      # Latitude, Longitude, SOG, COG
    n_out = 4      # predict dLatitude, dLongitude, dSOG, dCOG
    n_hid = 256    # hidden size for RNN/LSTM/GRU/Transformer
    
    # Sequence length for training and rollout
    seq_len = 50

    # load a model and evaluate on test data
    model_name = "transformer"
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