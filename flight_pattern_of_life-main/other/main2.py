
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
from pytorch_lightning import Trainer
import numpy as np

from net import SimpleNet
from net_lstm import SimpleLSTM
from net_transformer import SimpleTimeSeriesTransformer

from datamodule import Datamodule
from model import FlightModel

from file_parsing_utils import create_csv_dict
from coordinate_transform import CoordinateEnum, helper_get_coordinate_system_based_on_enum

from enum import Enum

cuda = torch.cuda.is_available()

class ModelEnum(Enum):
    SimpleCNN = 0
    SimpleLSTM = 1
    SimpleTransformer = 2 
    SimpleMamba = 3
    

    def __str__(self):
        return self.name

def train_func():

    num_workers = 8

    max_num_val_maps = 8


    num_input_rows_total = 100
    min_rows_input = 100
    num_output_rows = 1
    
    coordinate_system_enum = CoordinateEnum.LatLongCoordinates
    coordinate_system = helper_get_coordinate_system_based_on_enum(coordinate_enum=coordinate_system_enum)

    auxiliary_input_channels = [
                                "diff_time", 
                                "flight_course_corrected", 
                                "flight_course_unknown", 
                               ]
    
    auxiliary_output_channels = []

    

    num_res_blocks = 4
    intermediate_channels = 64
    in_channels = len(coordinate_system) + len(auxiliary_input_channels)
    out_channels = len(coordinate_system)

    model_enum = ModelEnum.SimpleTransformer


    if model_enum == ModelEnum.SimpleCNN:
        dilation = [1 if i % 2 == 0 else 2 for i in range(num_res_blocks + 2)]
        net = SimpleNet(in_channels=in_channels, 
                        out_channels=out_channels, 
                        intermediate_channels=intermediate_channels, 
                        num_res_blocks=num_res_blocks, 
                        num_output_rows=num_output_rows, 
                        dilation = dilation, 
                        kernel_size = 27, 
                        use_bn_norm=False, #True, 
                        stride=1, 
                        bias=True)
        
    elif model_enum == ModelEnum.SimpleLSTM:
        num_lstm_layers = 12
        hidden_size = 64
        net = SimpleLSTM(in_channels, num_lstm_layers, out_channels, hidden_size=hidden_size, out_timesteps=num_output_rows) 

    elif model_enum == ModelEnum.SimpleTransformer:
        num_transformer_blocks_stacked = 4
        hidden_dim = 64
        nhead = 8
        net = SimpleTimeSeriesTransformer(in_channels, num_input_rows_total, num_transformer_blocks_stacked, out_channels, num_output_rows, hidden_dim, nhead)

    else:
        raise NotImplementedError(f"Model {model_enum} not yet implemented")
    
    individual_flights_dir = '/raid/mo0dy/F2/FS/'
    flight_dfs = create_csv_dict(individual_flights_dir)
    print(f"\n\n\nNUMBER OF DIRS:\n{len(flight_dfs)}\n\n\n")


    datamodule = Datamodule(all_flight_dataframes_dict = flight_dfs, 
                            num_input_rows_total = num_input_rows_total, 
                            min_rows_input = min_rows_input, 
                            num_output_rows = 1, 
                            coordinate_system_enum = coordinate_system_enum,
                            auxiliary_input_channels = auxiliary_input_channels,
                            auxiliary_output_channels = auxiliary_output_channels,
                            train_prop = 0.8, 
                            batch_size = 32, 
                            num_workers = num_workers, 
                            pin_memory = True,)

    train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.test_dataloader()

    # dummy tensor
    test_dataloader_iterator = test_dataloader.__iter__()
    tensors_dict = next(test_dataloader_iterator)
    dummy_input_tensor = tensors_dict["input_tensor"]
    dummy_output = net(dummy_input_tensor)


    loss_fn = torch.nn.functional.mse_loss
    optimizer = None #                               TODO 
    mean = None
    std = None
    model = FlightModel(net, coordinate_system_enum, loss_fn, optimizer, max_num_val_maps = max_num_val_maps, n_future_timesteps = 10, mean=mean, std=std)

    every_n_train_steps = 5_000 #2000
    val_check_interval = 5_000 #2000
    log_every_n_steps = 1 #2000

    enable_checkpointing = True
    max_steps = 200_000_000
    val_check_interval = val_check_interval
    limit_val_batches = 1
    log_every_n_steps = log_every_n_steps
    accelerator = "gpu" #"mps", 
    devices = 1
    gradient_clip_val = 0.5

    model_save_dir = '/raid/mo0dy/models/A2'
    every_n_train_steps = 5000 
    checkpoint_callback = ModelCheckpoint(dirpath=model_save_dir, save_last=True, every_n_train_steps=every_n_train_steps, save_top_k = -1)
    callbacks = [checkpoint_callback]
    #ckpt_path = '/raid/mo0dy/models/ckpt'
    ckpt_path = None


    trainer = Trainer(enable_checkpointing=True,
                    max_steps = max_steps, 
                    callbacks = callbacks, 
                    logger = None, 
                    val_check_interval = val_check_interval, 
                    limit_val_batches = limit_val_batches, 
                    log_every_n_steps = log_every_n_steps, 
                    accelerator = accelerator,  #"mps", 
                    devices = 1, 
                    gradient_clip_val = 0.5, 
                    )

    trainer.fit(model=model, 
                train_dataloaders=train_dataloader, 
                val_dataloaders = test_dataloader, 
                ckpt_path = ckpt_path, 
            )


if __name__ == "__main__":
    train_func()
