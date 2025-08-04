import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
from pytorch_lightning import Trainer
from net_transformer import SimpleTimeSeriesTransformer
from datamodule import Datamodule
from model import FlightModel
from file_parsing_utils import create_csv_dict
from coordinate_transform import CoordinateEnum, helper_get_coordinate_system_based_on_enum

from enum import Enum


def inf():

    auxiliary_input_channels = [
                                "diff_time", 
                                "flight_course_corrected", 
                                "flight_course_unknown", 
                                ]
    loss_fn = torch.nn.functional.mse_loss
    optimizer = None #                               TODO 
    mean = None
    std = None
    num_input_rows_total = 100
    min_rows_input = 100
    num_output_rows = 1
    max_num_val_maps = 8
    num_transformer_blocks_stacked = 4
    hidden_dim = 64
    nhead = 8
    coordinate_system_enum = CoordinateEnum.LatLongCoordinates
    coordinate_system = helper_get_coordinate_system_based_on_enum(coordinate_enum=coordinate_system_enum)
    out_channels = len(coordinate_system)
    in_channels = len(coordinate_system) + len(auxiliary_input_channels)
    net = SimpleTimeSeriesTransformer(in_channels, num_input_rows_total, num_transformer_blocks_stacked, out_channels, num_output_rows, hidden_dim, nhead)
    ckpt_path = "/raid/mo0dy/models/A1/model.ckpt"

    model = FlightModel.load_from_checkpoint(checkpoint_path=ckpt_path, 
                                         map_location='cpu',
                                         coordinate_system_enum=coordinate_system_enum,
                                         model=net, 
                                         loss_fn=loss_fn, 
                                         optimizer=optimizer, 
                                         max_num_val_maps=max_num_val_maps, 
                                         n_future_timesteps=10, 
                                         mean=mean, 
                                         std=std)

    # model = FlightModel(net, coordinate_system_enum, loss_fn, optimizer, max_num_val_maps = max_num_val_maps, n_future_timesteps = 10, mean=mean, std=std)
    # model = model.load_from_checkpoint('/raid/mo0dy/models/A1/model.ckpt')


if __name__ == "__main__":
    inf()
