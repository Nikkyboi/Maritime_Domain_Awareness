import torch
import matplotlib.pyplot as plt
import torch.nn as nn
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