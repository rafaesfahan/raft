
import torch
import torch.nn as nn
from torch_utils import normalize_tensor, unnormalize_tensor




class SimpleLSTM(nn.Module):
    def __init__(self, input_channels, num_layers, output_channels, hidden_size=64, conv_out_channels=32, kernel_size=3, out_timesteps=10, normalize_inside_of_network=True, norm_type='layer'):
        super(SimpleLSTM, self).__init__()
        self.conv = nn.Conv1d(input_channels, conv_out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        
        self.input_channels = input_channels
        self.num_layers = num_layers
        self.output_channels = output_channels
        self.norm_type = norm_type
        self.hidden_size = hidden_size
        self.conv_out_channels = conv_out_channels
        self.kernel_size = kernel_size

        if norm_type == 'layer':
            self.norm1 = nn.LayerNorm(conv_out_channels)
            self.norm2 = nn.LayerNorm(hidden_size)
        elif norm_type == 'instance':
            self.norm1 = nn.InstanceNorm1d(conv_out_channels, affine=False)
            self.norm2 = nn.InstanceNorm1d(hidden_size, affine=False)
        elif norm_type == 'batch':
            self.norm1 = nn.BatchNorm1d(conv_out_channels)
            self.norm2 = nn.BatchNorm1d(hidden_size)
        else:
            self.norm1 = None
            self.norm2 = None

        self.lstm = nn.LSTM(input_size=conv_out_channels, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, output_channels)
        self.out_timesteps = out_timesteps
        self.len_coordinate_system = 2
        self.normalize_inside_of_network = normalize_inside_of_network

    def forward(self, x):
        if self.normalize_inside_of_network:
            x, min_val, max_val = normalize_tensor(x)

        # x shape: [Batch, Channels, Timesteps]
        x = self.conv(x)
        if self.norm1 is not None:
            if self.norm_type == 'layer':
                x = x.permute(0, 2, 1)  # Change to [Batch, Timesteps, Channels] for LayerNorm
                x = self.norm1(x)
                x = x.permute(0, 2, 1)  # Change back to [Batch, Channels, Timesteps]
            else:
                x = self.norm1(x)
                
        x = x.permute(0, 2, 1)  # Change to [Batch, Timesteps, Channels]
        x, _ = self.lstm(x)

        if self.norm2 is not None:
            if self.norm_type == 'layer':
                x = self.norm2(x)
            else:
                x = x.permute(0, 2, 1)  # Change to [Batch, Channels, Timesteps]
                x = self.norm2(x)
                x = x.permute(0, 2, 1)  # Change back to [Batch, Timesteps, Channels]
        
        x = self.fc(x)  # Apply fully connected layer to the entire output
        x = x[:, :self.out_timesteps, :]  # Adjust the number of timesteps
        x = x.permute(0, 2, 1)  # Change to [Batch, Out_Channels, Timesteps]

        if self.normalize_inside_of_network:
            x = unnormalize_tensor(x, min_val[:, :self.len_coordinate_system, :], max_val[:, :self.len_coordinate_system, :])
        return x









# import torch
# import torch.nn as nn
# from torch_utils import normalize_tensor, unnormalize_tensor

# import torch
# import torch.nn as nn

# class SimpleLSTM(nn.Module):
#     def __init__(self, input_channels, num_layers, output_channels, hidden_size=64, out_timesteps=10, normaize_inside_of_network=True):
#         super(SimpleLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size=input_channels, 
#                             hidden_size=hidden_size, 
#                             num_layers=num_layers, 
#                             batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_channels)
#         self.out_timesteps = out_timesteps
#         self.len_coordinate_system = 2  # TODO TODO TODO CONFIGURE 
#         self.normaize_inside_of_network = normaize_inside_of_network

#     def forward(self, x):

#         if self.normaize_inside_of_network:
#             x, min_val, max_val = normalize_tensor(x)

#         # x shape: [Batch, Channels, Timesteps]
#         x = x.permute(0, 2, 1)  # Change to [Batch, Timesteps, Channels]
#         x, _ = self.lstm(x)
#         x = self.fc(x)  # Apply fully connected layer to the entire output
#         x = x[:, :self.out_timesteps, :]  # Adjust the number of timesteps
#         x = x.permute(0, 2, 1)  # Change to [Batch, Out_Channels, Timesteps]

#         if self.normaize_inside_of_network:
#             x = unnormalize_tensor(x, min_val[:, :self.len_coordinate_system, :], max_val[:, :self.len_coordinate_system, :])
#         return x

# # Example usage:
# # model = SimpleLSTM(input_channels=10, num_layers=2, output_channels=2, hidden_size=64, out_timesteps=15)
