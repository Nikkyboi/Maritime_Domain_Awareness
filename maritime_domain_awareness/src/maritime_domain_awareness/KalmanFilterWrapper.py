import torch
from .models.KalmanFilter import KalmanFilter
import numpy as np

class KalmanFilterWrapper(torch.nn.Module):
    def __init__(self, dt=1.0, process_variance=1e-5, measurement_variance=0.1, init_error=1.0):
        super().__init__()
        self.dt = dt
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.init_error = init_error

    def forward(self, x_seq):
        # x_seq: [batch_size, seq_len, n_in] or [seq_len, n_in]
        # Output: [batch_size, seq_len, n_out] or [seq_len, n_out]
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(0)  # [1, seq_len, n_in]
        batch_size, seq_len, n_in = x_seq.shape
        preds = []
        for b in range(batch_size):
            seq = x_seq[b].cpu().numpy()
            # Initialize filter with first measurement
            df_init = {'x': seq[0,0], 'y': seq[0,1], 'v_e': 0.0, 'v_n': 0.0}
            kf = KalmanFilter(df=df_init, dt=self.dt, process_variance=self.process_variance,
                              measurement_variance=self.measurement_variance, init_error=self.init_error)
            # Set initial state estimate
            kf.x_est = np.array([[seq[0,0]], [seq[0,1]], [0.0], [0.0]])
            pred_seq = []
            for t in range(seq_len):
                z = seq[t]
                kf.step(z)
                pred_seq.append([kf.x_est[0,0], kf.x_est[1,0]])
            preds.append(pred_seq)
        preds = torch.tensor(preds, dtype=torch.float32)
        return preds