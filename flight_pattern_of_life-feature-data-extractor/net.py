

import torch
import torch.nn as nn
from torch_utils import normalize_tensor, unnormalize_tensor

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, kernel_size=9, norm_type='batch', stride=1, bias=True):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.norm_type = norm_type

        self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels,
                              kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation,
                              padding="same", groups=self.in_channels, bias=self.bias, padding_mode='replicate')
        self.pw_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1,
                                 stride=1, dilation=self.dilation, padding="same", groups=1, bias=bias,
                                 padding_mode="replicate")
        self.act = nn.PReLU()
        
        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(self.out_channels, affine=False)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(self.out_channels)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm1d(self.out_channels, affine=False)
        else:
            self.norm = None

    def forward(self, x):
        x = self.conv(x)
        x = self.pw_conv(x)
        x = self.act(x)
        if self.norm is not None:
            if self.norm_type == 'layer':
                # Layer normalization for each timestep
                x = x.transpose(1, 2)  # [Batch x Channels x Timesteps] -> [Batch x Timesteps x Channels]
                x = self.norm(x)
                x = x.transpose(1, 2)  # [Batch x Timesteps x Channels] -> [Batch x Channels x Timesteps]
            else:
                x = self.norm(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, kernel_size=9, norm_type='batch', stride=1, bias=True):
        super(ResidualBlock, self).__init__()
        self.block1 = BasicBlock(in_channels, out_channels, dilation, kernel_size, norm_type, stride, bias)
        self.block2 = BasicBlock(in_channels, out_channels, dilation, kernel_size, norm_type, stride, bias)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1) + x1
        return x2

class SimpleNet(nn.Module):
    def __init__(self, in_channels=7, out_channels=4, intermediate_channels=64, num_res_blocks=8, 
                 num_output_rows=1, dilation=1, kernel_size=9, norm_type='batch', stride=1, bias=True, normalize_inside_of_network=True):
        super(SimpleNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.intermediate_channels = intermediate_channels
        self.half_intermidiate_channels = int(self.intermediate_channels / 2.0)
        self.num_res_blocks = num_res_blocks
        self.num_output_rows = num_output_rows
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.stride = stride
        self.bias = bias
        self.normalize_inside_of_network = normalize_inside_of_network

        if isinstance(dilation, int):
            dilation = [dilation for _ in range(num_res_blocks + 2)]
        elif isinstance(dilation, list):
            if len(dilation) < num_res_blocks:
                dilation = (dilation * (num_res_blocks // len(dilation) + 1))[:num_res_blocks]
        self.dilation = dilation

        self.first_conv_block = BasicBlock(self.in_channels, intermediate_channels, int(dilation[0]), kernel_size, norm_type, stride, bias)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(intermediate_channels, intermediate_channels, int(dilation[idx]), kernel_size, norm_type, stride, bias) for idx in range(num_res_blocks)])
        self.last_conv_block = BasicBlock(intermediate_channels, self.out_channels, int(dilation[-1]), kernel_size, norm_type, stride, bias)
        self.one_more_linear = nn.LazyLinear(100)
        self.act_one_more_linear = nn.PReLU()
        self.last_linear = nn.LazyLinear(self.out_channels * self.num_output_rows)
        self.act_final = nn.PReLU()
        self.len_coordinate_system = 2

    def forward(self, x):
        if self.normalize_inside_of_network:
            x, min_val, max_val = normalize_tensor(x)
        b = x.shape[0]
        x = self.first_conv_block(x)

        # diff step implementation
        ord = x[:, :self.half_intermidiate_channels, 1:]
        diff = x[:, self.half_intermidiate_channels:, :]
        diff = torch.diff(diff, dim=-1)  
        x[:, self.half_intermidiate_channels:, 1:] = diff 

        x = self.residual_blocks(x)
        x = self.last_conv_block(x)
        x = x.flatten(start_dim=1, end_dim=2)
        x = self.one_more_linear(x)
        x = self.act_one_more_linear(x)
        x = self.last_linear(x)
        x = self.act_final(x)
        x = torch.reshape(x, (b, self.out_channels, self.num_output_rows))
        if self.normalize_inside_of_network:
            x = unnormalize_tensor(x, min_val[:, :self.len_coordinate_system, :], max_val[:, :self.len_coordinate_system, :])
        return x

def init_weights(m, mean=0.0, std=0.0001):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=mean, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)















# import torch
# import torch.nn as nn
# from torch_utils import normalize_tensor, unnormalize_tensor

# class BasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dilation=1, kernel_size=9, use_bn_norm=True, stride=1, bias=True):
#         super(BasicBlock, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.dilation = dilation
#         self.kernel_size = kernel_size
#         self.use_bn_norm = use_bn_norm
#         self.stride = stride
#         self.bias = bias

#         if stride == 1:
#             self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels,
#                                   kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation,
#                                   padding="same", groups=self.in_channels, bias=self.bias, padding_mode='replicate')
#         else:
#             self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels,
#                                   kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation,
#                                   groups=self.in_channels, bias=self.bias, padding_mode='replicate')
#         self.pw_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1,
#                                  stride=1, dilation=self.dilation, padding="same", groups=1, bias=bias,
#                                  padding_mode="replicate")
#         self.act = nn.PReLU()
#         self.bn = nn.BatchNorm1d(self.out_channels, affine=False)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.pw_conv(x)
#         x = self.act(x)
#         if self.use_bn_norm:
#             x = self.bn(x)
#         return x

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dilation=1, kernel_size=9, use_bn_norm=True, stride=1, bias=True):
#         super(ResidualBlock, self).__init__()
#         self.block1 = BasicBlock(in_channels, out_channels, dilation, kernel_size, use_bn_norm, stride, bias)
#         self.block2 = BasicBlock(in_channels, out_channels, dilation, kernel_size, use_bn_norm, stride, bias)

#     def forward(self, x):
#         x1 = self.block1(x)
#         x2 = self.block2(x1) + x1
#         return x2

# class SimpleNet(nn.Module):
#     def __init__(self, in_channels=7, out_channels=4, intermediate_channels=64, num_res_blocks=8, 
#                  num_output_rows=1, dilation=1, kernel_size=9, use_bn_norm=True, stride=1, bias=True, normalize_inside_of_network=True):
#         super(SimpleNet, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.intermediate_channels = intermediate_channels
#         self.half_intermidiate_channels = int(self.intermediate_channels / 2.0)
#         self.num_res_blocks = num_res_blocks
#         self.num_output_rows = num_output_rows
#         self.dilation = dilation
#         self.kernel_size = kernel_size
#         self.use_bn_norm = use_bn_norm
#         self.stride = stride
#         self.bias = bias

#         self.normalize_inside_of_network = normalize_inside_of_network

#         if isinstance(dilation, int):
#             dilation = [dilation for _ in range(num_res_blocks + 2)]
#         elif isinstance(dilation, list):
#             if len(dilation) < num_res_blocks:
#                 dilation = (dilation * (num_res_blocks // len(dilation) + 1))[:num_res_blocks]
#         self.dilation = dilation

#         self.first_conv_block = BasicBlock(self.in_channels, intermediate_channels, int(dilation[0]), kernel_size, use_bn_norm, stride, bias)
#         self.residual_blocks = nn.Sequential(*[ResidualBlock(intermediate_channels, intermediate_channels, int(dilation[idx]), kernel_size, use_bn_norm, stride, bias) for idx in range(num_res_blocks)])
#         self.last_conv_block = BasicBlock(intermediate_channels, self.out_channels, int(dilation[-1]), kernel_size, use_bn_norm, stride, bias)
#         self.one_more_linear = nn.LazyLinear(100)
#         self.act_one_more_linear = nn.PReLU()
#         self.last_linear = nn.LazyLinear(self.out_channels * self.num_output_rows)
#         self.act_final = nn.PReLU()
#         self.len_coordinate_system = 2

#     def forward(self, x):
#         if self.normalize_inside_of_network:
#             x, min_val, max_val = normalize_tensor(x)
#         b = x.shape[0]
#         x = self.first_conv_block(x)


#         # diff step implementation
#         ord = x[:, :self.half_intermidiate_channels, 1:]
#         diff = x[:, self.half_intermidiate_channels:, :]
#         diff = torch.diff(diff, dim=-1)  
#         x[:, self.half_intermidiate_channels:, 1:] = diff 
        

#         x = self.residual_blocks(x)
#         x = self.last_conv_block(x)
#         x = x.flatten(start_dim=1, end_dim=2)
#         x = self.one_more_linear(x)
#         x = self.act_one_more_linear(x)
#         x = self.last_linear(x)
#         x = self.act_final(x)
#         x = torch.reshape(x, (b, self.out_channels, self.num_output_rows))
#         if self.normalize_inside_of_network:
#             x = unnormalize_tensor(x, min_val[:, :self.len_coordinate_system, :], max_val[:, :self.len_coordinate_system, :])
#         return x

# def init_weights(m, mean=0.0, std=0.0001):
#     if isinstance(m, (nn.Conv2d, nn.Linear)):
#         nn.init.normal_(m.weight, mean=mean, std=std)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)




















# import torch
# import torch.nn as nn

# from torch_utils import normalize_tensor, unnormalize_tensor

# # A very basic net

# class BasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dilation = 1, kernel_size = 9, use_bn_norm=True, stride=1, bias=True):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.dilation = dilation
#         self.kernel_size = kernel_size
#         self.use_bn_norm = use_bn_norm
#         self.stride = stride
#         self.bias = bias

#         if stride == 1:
#             self.conv = torch.nn.Conv1d(in_channels=self.in_channels,
#                                         out_channels=self.in_channels, 
#                                         kernel_size=self.kernel_size,
#                                         stride=self.stride,
#                                         dilation=self.dilation,
#                                         padding="same", 
#                                         groups=self.in_channels,
#                                         bias=self.bias,
#                                         padding_mode='replicate')
#         else:
#             self.conv = torch.nn.Conv1d(in_channels=self.in_channels,
#                                         out_channels=self.in_channels, 
#                                         kernel_size=self.kernel_size,
#                                         stride=self.stride,
#                                         dilation=self.dilation,
#                                         groups=self.in_channels,
#                                         bias=self.bias,
#                                         padding_mode='replicate')

#         self.pw_conv = torch.nn.Conv1d(in_channels=self.in_channels, 
#                                     out_channels=self.out_channels, 
#                                     kernel_size=1, 
#                                     stride=1, 
#                                     dilation=self.dilation, 
#                                     padding="same", 
#                                     groups=1, 
#                                     bias=bias, 
#                                     padding_mode="replicate")
        
#         self.act = nn.PReLU()
#         self.bn = nn.BatchNorm1d(self.out_channels, affine=False)

#         print("debug use batch norm: ", self.use_bn_norm)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.pw_conv(x)
#         x = self.act(x)
#         if self.use_bn_norm:
#             x = self.bn(x)

#         return x

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dilation = 1, kernel_size = 9, use_bn_norm=True, stride=1, bias=True):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.dilation = dilation
#         self.kernel_size = kernel_size
#         self.use_bn_norm = use_bn_norm
#         self.stride = stride
#         self.bias = bias

#         self.block1 = BasicBlock(in_channels=in_channels, out_channels=out_channels, dilation = dilation, kernel_size = kernel_size, use_bn_norm=use_bn_norm, stride=stride, bias=bias)
#         self.block2 = BasicBlock(in_channels=in_channels, out_channels=out_channels, dilation = dilation, kernel_size = kernel_size, use_bn_norm=use_bn_norm, stride=stride, bias=bias)

#     def forward(self, x):
#         x1 = self.block1(x)
#         x2 = self.block2(x1) + x1
#         return x2
        

# class SimpleNet(nn.Module):
#     def __init__(self, 
#                  in_channels=7, 
#                  out_channels=4, 
#                  intermediate_channels=64, 
#                  num_res_blocks=8, 
#                  num_output_rows=1, 
#                  dilation = 1, 
#                  kernel_size = 9, 
#                  use_bn_norm=True, 
#                  stride=1, 
#                  bias=True):
#         super().__init__()
        
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.intermediate_channels = intermediate_channels
#         self.num_res_blocks = num_res_blocks
#         self.num_output_rows = num_output_rows
#         self.dilation = dilation
#         self.kernel_size = kernel_size
#         self.use_bn_norm = use_bn_norm 
#         self.stride = stride
#         self.bias = bias

#         if isinstance(dilation, int):
#             dilation = [dilation for _ in range(num_res_blocks + 2)] # num_res_blocks + one block in begining, one block at end
#         elif isinstance(dilation, list):
#             if len(dilation) < num_res_blocks: # if dilation list is less than num_res_blocks we repeat it untill we have a list that is long enough
#                 dilation = (dilation * (num_res_blocks // len(dilation) + 1))[:num_res_blocks]

#         self.dilation = dilation
#         self.first_conv_block = BasicBlock(self.in_channels, intermediate_channels, int(dilation[0]), kernel_size, use_bn_norm, stride, bias)
#         self.residual_blocks = [ResidualBlock(self.intermediate_channels, self.intermediate_channels, dilation = int(dilation[idx]), kernel_size = kernel_size, use_bn_norm=use_bn_norm, stride=stride, bias=bias) for idx in range(num_res_blocks)]
#         self.residual_blocks = nn.Sequential(*self.residual_blocks)
#         self.last_conv_block = BasicBlock(intermediate_channels, self.out_channels, int(dilation[-1]), kernel_size, use_bn_norm, stride, bias)

#         self.one_more_linear = torch.nn.LazyLinear(100)
#         self.act_one_more_linear = nn.PReLU()
#         self.last_linear = torch.nn.LazyLinear(self.out_channels * self.num_output_rows)
#         self.act_final = nn.PReLU()

#         self.len_coordinate_system = 2 # TODO TODO TODO CONFIGURE 


#     def forward(self, x):
#         x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
#         ####x = torch.diff(x, dim=-1) # feed in the diff along the time step dimension into the model

#         x, min_val, max_val = normalize_tensor(x)
#         b = x.shape[0]
#         x = self.first_conv_block(x)
#         x = self.residual_blocks(x)
#         x = self.last_conv_block(x)
#         x = x.flatten(start_dim=1, end_dim=2)
#         x = self.one_more_linear(x)
#         x = self.act_one_more_linear(x)
#         x = self.last_linear(x)
#         x = self.act_final(x)
#         x = torch.reshape(x, (b, self.out_channels, self.num_output_rows))

#         #print("debug shapes: ", x.shape, min_val.shape, max_val.shape)
#         x = unnormalize_tensor(x, min_val[:, :self.len_coordinate_system, :], max_val[:, :self.len_coordinate_system, :])

#         return x
    



# def init_weights(m, mean=0.0, std=0.0001):
#     if isinstance(m, (nn.Conv2d, nn.Linear)):
#         nn.init.normal_(m.weight, mean=mean, std=std)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)