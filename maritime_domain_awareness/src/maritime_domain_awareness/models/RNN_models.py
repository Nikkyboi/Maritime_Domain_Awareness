from torch import nn
import torch

class myRecurrent(nn.Module):
    """
    Simple recurrent neural network model as baseline.
    
    n_in: Number of input features per timestamp (lat, lon, sog, cog, heading ...)
    n_out: Number of output features per timestamp
    """
    def __init__(self, n_in : int, n_hid: int, n_out: int, num_layers : int =1, batch_first : bool =False, dropout: float =0.0, activation: str ="tanh"):
        super(myRecurrent, self).__init__()
        # Saving parameters for summary
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.num_layers = num_layers
        self.dropout = dropout

        # batch_first:
        # If true changes the input shape to (batch, time-step, feature)
        # If false, the input shape is (time-step, batch, feature)
        self.batch_first = batch_first
        
        # Activation function: "tanh" or "relu"
        if activation not in ["tanh", "relu"]:
            raise ValueError("Activation must be 'tanh' or 'relu'")
        self.activation = activation
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=n_in,
            hidden_size=n_hid,
            num_layers=num_layers,
            nonlinearity=activation,      # "tanh" or "relu"
            bias=True,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output layer 
        self.fc_out = nn.Linear(n_hid, n_out)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        x: [B, T, n_in] if batch_first=True
        x: [T, B, n_in] if batch_first=False
        
        returns:
        x: [B, T, n_out] if batch_first=True
        x: [T, B, n_out] if batch_first=False
        """
        
        # RNN returns output and last hidden state
        x, h = self.rnn(x)

        # Output layer
        x = self.fc_out(x)

        return x

    def print_summary(self) -> None:
        """
        Print a summary of the model architecture.
        """
        print("Input features (n_in):", self.n_in)
        print("Hidden features (n_hid):", self.n_hid)
        print("Output features (n_out):", self.n_out)
        print("Number of layers (num_layers):", self.num_layers)
        print("Batch first:", self.batch_first)
        print("Dropout rate:", self.dropout)
        print("Activation function:", self.activation)
        
class myLSTM(nn.Module):
    """
    LSTM model for sequence modeling.
    
    n_in: Number of input features per timestamp (lat, lon, sog, cog, heading ...)
    n_out: Number of output features per timestamp
    """
    def __init__(self, n_in : int, n_hid: int, n_out: int, num_layers : int =1, batch_first : bool =False, dropout: float =0.0, bias: bool =True):
        super(myLSTM, self).__init__()

        # Saving parameters for summary
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=n_in,
            hidden_size=n_hid,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output layer 
        self.fc_out = nn.Linear(n_hid, n_out)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        x: [B, T, n_in] if batch_first=True
        x: [T, B, n_in] if batch_first=False
        
        returns:
        x: [B, T, n_out] if batch_first=True
        x: [T, B, n_out] if batch_first=False
        """
        
        # LSTM returns output and last hidden state
        x, (h, c) = self.lstm(x)

        # Output layer
        x = self.fc_out(x)

        return x
    
    def print_summary(self) -> None:
        """
        Print a summary of the model architecture.
        """
        print("Input features (n_in):", self.n_in)
        print("Hidden features (n_hid):", self.n_hid)
        print("Output features (n_out):", self.n_out)
        print("Number of layers (num_layers):", self.num_layers)
        print("Bias:", self.bias)
        print("Batch first:", self.batch_first)
        print("Dropout rate:", self.dropout)
    
class myGRU(nn.Module):
    """
    GRU model for sequence modeling.
    
    Difference between GRU and LSTM:
    - GRU has two gates (reset + update) and no separate cell state,
    while LSTM uses three gates (input, forget, output) and maintains
    a separate cell state in addition to the hidden state.
    
    n_in: Number of input features per timestamp (lat, lon, sog, cog, heading ...)
    n_out: Number of output features per timestamp
    """
    def __init__(self, n_in : int, n_hid: int, n_out: int, num_layers : int =1, batch_first : bool =False, dropout: float =0.0, bias: bool =True):
        super(myGRU, self).__init__()
        # Saving parameters for summary
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout

        # GRU layer
        self.gru = nn.GRU(
            input_size = self.n_in,
            hidden_size = self.n_hid,
            num_layers = self.num_layers,
            bias = self.bias,
            batch_first = self.batch_first,
            dropout = self.dropout if self.num_layers > 1 else 0.0,
        )

        # Output layer
        self.fc_out = nn.Linear(self.n_hid, self.n_out)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Same input/output shape as LSTM and RNN.
        """
        
        # GRU returns output and last hidden state
        x, h = self.gru(x)

        # Output layer
        x = self.fc_out(x)

        return x

    def print_summary(self) -> None:
        """
        Print a summary of the model architecture.
        """
        print("Input features (n_in):", self.n_in)
        print("Hidden features (n_hid):", self.n_hid)
        print("Output features (n_out):", self.n_out)
        print("Number of layers (num_layers):", self.num_layers)
        print("Batch first:", self.batch_first)
        print("Dropout rate:", self.dropout)

if __name__ == "__main__":
    # Testing the myLSTM model
    n_in = 5    # lat, lon, sog, cog, heading
    n_hid = 2   # hidden size
    n_out = 2   # lat, lon
    
    # test model
    myRNN_model = myRecurrent(n_in=n_in, n_hid=n_hid, n_out=n_out, num_layers=2)
    myLSTM_model = myLSTM(n_in=n_in, n_hid=n_hid, n_out=n_out, num_layers=2)
    myGRU_model = myGRU(n_in=n_in, n_hid=n_hid, n_out=n_out, num_layers=2)
    
    # print model summary
    print(myRNN_model)
    print(myLSTM_model)
    
    # create dummy input: batch_size=3, seq_length=10, n_in=5
    B, T, F = 3, 5, n_in
    #x_input_1 = torch.randn(B, T, F) # [B, T, n_in] for batch_first=True
    x_input_2 = torch.randn(T, B, F) # [T, B, n_in] for batch_first=False
    
    # forward pass
    y_rnn = myRNN_model(x_input_2)
    y_lstm = myLSTM_model(x_input_2)
    y_gru = myGRU_model(x_input_2)

    # print output shape and values
    print("Output shape:", y_rnn.shape)
    print("Output:", y_rnn)
    print("Output shape:", y_lstm.shape)
    print("Output:", y_lstm)
    print("Output shape:", y_gru.shape)
    print("Output:", y_gru)
    
    print("Testing completed.")
    print(myGRU_model.print_summary())