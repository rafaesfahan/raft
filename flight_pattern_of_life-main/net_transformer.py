import torch
import torch.nn as nn
from torch_utils import normalize_tensor, unnormalize_tensor

class SimpleTimeSeriesTransformer(nn.Module):
    def __init__(self, input_num_channels, input_num_timesteps, num_transformer_blocks_stacked, output_num_channels, output_num_timesteps, hidden_dim, nhead=8):
        super(SimpleTimeSeriesTransformer, self).__init__()
        
        assert hidden_dim % nhead == 0, "hidden_dim must be divisible by nhead"
        
        self.input_num_channels = input_num_channels
        self.input_num_timesteps = input_num_timesteps
        self.num_transformer_blocks_stacked = num_transformer_blocks_stacked
        self.output_num_channels = output_num_channels
        self.output_num_timesteps = output_num_timesteps
        self.hidden_dim = hidden_dim
        
        self.input_projection = nn.Linear(input_num_channels, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_num_timesteps, hidden_dim))
        
        transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_blocks_stacked)
        
        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=input_num_timesteps // output_num_timesteps)
        self.output_projection = nn.Linear(hidden_dim, output_num_channels)

        self.len_coordinate_system = 2 # TODO TODO TODO CONFIGURE 
    
    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x, min_val, max_val = normalize_tensor(x)

        # x shape: [Batch, input_num_channels, input_num_timesteps]
        x = x.permute(0, 2, 1)  # Change to [Batch, input_num_timesteps, input_num_channels]
        x = self.input_projection(x)  # Project input to hidden_dim
        x += self.positional_encoding  # Add positional encoding
        
        x = self.transformer_encoder(x)  # Apply transformer encoder
        x = x.permute(0, 2, 1)  # Change to [Batch, hidden_dim, input_num_timesteps]
        x = self.conv1d(x)  # Apply convolution to reduce timesteps
        x = x.permute(0, 2, 1)  # Change to [Batch, output_num_timesteps, hidden_dim]
        x = self.output_projection(x)  # Project to output_num_channels
        x = x.permute(0, 2, 1)  # Change to [Batch, output_num_channels, output_num_timesteps]
        
        x = unnormalize_tensor(x, min_val[:, :self.len_coordinate_system, :], max_val[:, :self.len_coordinate_system, :])
        return x