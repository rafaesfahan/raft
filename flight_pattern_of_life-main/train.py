import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
from pytorch_lightning import Trainer
import numpy as np
import matplotlib.pyplot as plt
import os
from net import SimpleNet
from net_lstm import SimpleLSTM
from net_transformer import SimpleTimeSeriesTransformer
from datamodule import Datamodule
from model import FlightModel
from file_parsing_utils import create_csv_dict
from coordinate_transform import CoordinateEnum, helper_get_coordinate_system_based_on_enum
from enum import Enum
import sys

menum  = sys.argv[1]
msteps = int(sys.argv[2])
device = int(sys.argv[3])
model_save_dir = '/raid/mo0dy/models/trans/short'
final_model_path = "/raid/mo0dy/models/trans/short/my_final_model.ckpt"



class ModelEnum(Enum):
    SimpleCNN = 0
    SimpleLSTM = 1
    SimpleTransformer = 2
    SimpleMamba = 3
    def __str__(self):
        return self.name

def train_func(final_model_save_path=None):
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
    
    if menum == 'cnn':
        model_enum = ModelEnum.SimpleCNN
    if menum == 'trans':
        model_enum = ModelEnum.SimpleTransformer
    if menum == 'lstm':
        model_enum = model_enum = ModelEnum.SimpleLSTM

    
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
    
    individual_flights_dir = '/raid/mo0dy/F/flights/'
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
    max_steps = msteps
    val_check_interval = val_check_interval
    limit_val_batches = 1
    log_every_n_steps = log_every_n_steps
    accelerator = "gpu" #"mps",
    devices = 1
    gradient_clip_val = 0.5
    
    # Modified checkpoint callback for every 100,000 steps
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_dir, 
        save_last=True, 
        every_n_train_steps=100_000,  # Changed to 100,000 as requested
        save_top_k=-1,
        filename='checkpoint-{step}'
    )
    
    # CSV Logger for training loss visualization
    csv_logger = loggers.CSVLogger(
        save_dir=model_save_dir,
        name="training_logs"
    )
    
    callbacks = [checkpoint_callback]
    #ckpt_path = '/raid/mo0dy/models/ckpt'
    ckpt_path = None
    
    trainer = Trainer(enable_checkpointing=True,
                    max_steps = max_steps,
                    callbacks = callbacks,
                    logger = csv_logger,  # Added CSV logger
                    val_check_interval = val_check_interval,
                    limit_val_batches = limit_val_batches,
                    log_every_n_steps = log_every_n_steps,
                    accelerator = accelerator,  #"mps",
                    devices = [device],
                    gradient_clip_val = 0.5,
                    )
    
    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders = test_dataloader,
                ckpt_path = ckpt_path,
            )
    
    # Save final model at the end of training
    if final_model_save_path is None:
        final_model_save_path = os.path.join(model_save_dir, 'final_model.ckpt')
    
    trainer.save_checkpoint(final_model_save_path)
    print(f"Final model saved to: {final_model_save_path}")
    
    # Generate training loss visualization
    log_dir = os.path.join(model_save_dir, "training_logs")
    plot_training_loss(log_dir, model_save_dir)
    
    return trainer, model

def plot_training_loss(log_dir, save_dir):
    """
    Plot training loss from CSV logs generated by PyTorch Lightning CSVLogger
    """
    try:
        import pandas as pd
        
        # Find the CSV file with metrics
        version_dirs = [d for d in os.listdir(log_dir) if d.startswith('version_')]
        if not version_dirs:
            print("No training logs found for visualization")
            return
        
        # Get the latest version directory
        latest_version = sorted(version_dirs)[-1]
        csv_path = os.path.join(log_dir, latest_version, 'metrics.csv')
        
        if not os.path.exists(csv_path):
            print(f"No metrics.csv found at {csv_path}")
            return
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Filter for training loss (assuming your model logs 'train_loss' or similar)
        # You might need to adjust this based on what your FlightModel actually logs
        train_loss_cols = [col for col in df.columns if 'train' in col.lower() and 'loss' in col.lower()]
        
        if not train_loss_cols:
            print("No training loss columns found in metrics")
            print(f"Available columns: {df.columns.tolist()}")
            return
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        for loss_col in train_loss_cols:
            # Remove NaN values and plot
            loss_data = df[loss_col].dropna()
            steps = df.loc[loss_data.index, 'step'] if 'step' in df.columns else range(len(loss_data))
            plt.plot(steps, loss_data, label=loss_col, alpha=0.7)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Use log scale for loss
        
        # Save the plot
        plot_path = os.path.join(save_dir, 'training_loss.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training loss plot saved to: {plot_path}")
        
    except ImportError:
        print("pandas not available for loss visualization")
    except Exception as e:
        print(f"Error creating training loss plot: {e}")

if __name__ == "__main__":
    # You can now specify a custom path for the final model
    train_func(final_model_save_path=final_model_path)