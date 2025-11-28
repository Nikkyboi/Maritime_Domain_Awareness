import torch
import torch.nn as nn
import math

# ------------------------------------------------------------
# need to delete
class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Works with shapes:
    - [T, B, d_model] if batch_first=False
    - [B, T, d_model] if batch_first=True
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        # Create positional encodings once [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(1)  # [max_len, 1, d_model] for batch_first=False

        # register as buffer (not a parameter, but moves with .to(device))
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            # x: [B, T, d_model]
            T = x.size(1)
            # pe: [max_len, 1, d_model] -> [1, T, d_model]
            pos = self.pe[:T].transpose(0, 1)  # [1, T, d_model]
            x = x + pos
        else:
            # x: [T, B, d_model]
            T = x.size(0)
            # pe: [T, 1, d_model]
            pos = self.pe[:T]  # [T, 1, d_model]
            x = x + pos

        return self.dropout(x)
# ------------------------------------------------------------

class myTransformer(nn.Module):
    """
    Transformer model for sequence modeling.
    
    n_in:  Number of input features per timestamp (lat, lon, sog, cog, heading, ...)
    n_hid: Transformer d_model (embedding dimension / hidden size)
    n_out: Number of output features per timestamp (e.g. lat, lon)
    """
    
    def __init__(
        self,
        n_in: int,
        n_hid: int,
        n_out: int,
        num_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        batch_first: bool = False,
        max_len: int = 5000,
    ):
        """
        Transformer model
        """
        super(myTransformer, self).__init__()
        
        # Saving parameters for summary
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_first = batch_first
        self.max_len = max_len
        
        # project input to n_hid dimension
        self.input_projection = nn.Linear(self.n_in, self.n_hid)
        
        # Positional Encoder
        self.positional_encoder = PositionalEncoding(
            d_model=self.n_hid,
            dropout=self.dropout,
            max_len=self.max_len,
            batch_first=self.batch_first,
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.n_hid,
            nhead=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=self.batch_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Output layer
        self.output_layer = nn.Linear(self.n_hid, self.n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        x: [B, T, n_in] if batch_first=True
        x: [T, B, n_in] if batch_first=False
        
        returns:
        x: [B, T, n_out] if batch_first=True
        x: [T, B, n_out] if batch_first=False
        """
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)

        # Output layer
        x = self.output_layer(x)

        return x
    
    def print_summary(self) -> None:
        """
        Print a summary of the model architecture.
        """
        print("Input features (n_in):", self.n_in)
        print("Hidden features (n_hid):", self.n_hid)
        print("Output features (n_out):", self.n_out)
        print("Number of layers (num_layers):", self.num_layers)
        print("Number of heads (n_heads):", self.n_heads)
        print("Dimension of feedforward (dim_feedforward):", self.dim_feedforward)
        print("Dropout rate:", self.dropout)
        print("Batch first:", self.batch_first)
        print("Max sequence length (max_len):", self.max_len)

if __name__ == "__main__":
    # Testing the myLSTM model
    n_in = 17    # lat, lon, sog, cog, heading
    n_hid = 128   # hidden size
    n_out = 2   # lat, lon
    
    # test model
    model = myTransformer(
    n_in=n_in,
    n_hid=n_hid,
    n_out=n_out,
    num_layers=4,
    n_heads=8,
    dim_feedforward=512,
    dropout=0.1,
    batch_first=False,
    )
    
    # print model summary
    model.print_summary()

