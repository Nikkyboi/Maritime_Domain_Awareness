from . import RNN_models
from . import Transformer_model
from . import mamba_model

"""
Interface to load models for training and evaluation.

Connects to RNN_models, Transformer_model and Mamba modules.

"""

def print_model_summary(name : str, model: object) -> None:
    """
    Print a summary of the model architecture.
    """
    print(f"Model Name: {name}")
    if hasattr(model, 'print_summary'):
        model.print_summary()
    else:
        print(model)

def load_model(model_path: str, n_in : int, n_out : int, n_hid : int = 64) -> object:
    """
    Load a model based on the model path.
    """
    if n_in is None or n_hid is None or n_out is None:
        raise ValueError("n_in, n_hid, and n_out must be provided to load the model.")

    model_path = model_path.lower()

    if "rnn" in model_path:
        model = RNN_models.myRecurrent(
            n_in=n_in,
            n_hid=n_hid,
            n_out=n_out,
            num_layers=1,
            batch_first=False,
            dropout=0.0,
        )
        
    elif "lstm" in model_path:
        model = RNN_models.myLSTM(
            n_in=n_in,
            n_hid=n_hid,
            n_out=n_out,
            num_layers=2,
            batch_first=False,
            dropout=0.2,
        )
        
    elif "gru" in model_path:
        model = RNN_models.myGRU(
            n_in=n_in,
            n_hid=n_hid,
            n_out=n_out,
            num_layers=2,
            batch_first=False,
            dropout=0.2,
        )
        
    elif "transformer" in model_path:
        model = Transformer_model.myTransformer(
        n_in=n_in,
        n_hid=n_hid,
        n_out=n_out,
        num_layers=3,
        n_heads=4,
        dim_feedforward=4 * n_hid, # standard transformer: 4*d_model
        dropout=0.1,
        batch_first=False,
    )
        
    elif "mamba" in model_path:
        # To Be Implemented
        model = mamba_model.Mamba.from_pretrained('state-spaces/mamba-370m')
        
        """
        model = Mamba.from_pretrained('state-spaces/mamba-370m')
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
        generate(model, tokenizer, 'Mamba is the')
        """
        
    else:
        raise ValueError(f"Unknown model type in path: \"{model_path}\"")

    print_model_summary(model_path, model)

    return model