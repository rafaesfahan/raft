import torch
import torch.nn as nn
from torch_utils import normalize_tensor, unnormalize_tensor

import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_channels, num_layers, output_channels, hidden_size=64, out_timesteps=10):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_channels, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_channels)
        self.out_timesteps = out_timesteps
        self.len_coordinate_system = 2  # TODO TODO TODO CONFIGURE 

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x, min_val, max_val = normalize_tensor(x)

        # x shape: [Batch, Channels, Timesteps]
        x = x.permute(0, 2, 1)  # Change to [Batch, Timesteps, Channels]
        x, _ = self.lstm(x)
        x = self.fc(x)  # Apply fully connected layer to the entire output
        x = x[:, :self.out_timesteps, :]  # Adjust the number of timesteps
        x = x.permute(0, 2, 1)  # Change to [Batch, Out_Channels, Timesteps]

        x = unnormalize_tensor(x, min_val[:, :self.len_coordinate_system, :], max_val[:, :self.len_coordinate_system, :])
        return x

# Example usage:
# model = SimpleLSTM(input_channels=10, num_layers=2, output_channels=2, hidden_size=64, out_timesteps=15)
