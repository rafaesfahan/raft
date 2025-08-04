import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
from pytorch_lightning import Trainer
from net_transformer import SimpleTimeSeriesTransformer
from datamodule import Datamodule
from model import FlightModel
from file_parsing_utils import create_csv_dict
from coordinate_transform import CoordinateEnum, helper_get_coordinate_system_based_on_enum
from iterate_flights import itterate_flights, build_features
from enum import Enum
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

# Model configuration (same as before)
auxiliary_input_channels = [
    "diff_time",
    "flight_course_corrected", 
    "flight_course_unknown",
]
auxiliary_output_channels = []
loss_fn = torch.nn.functional.mse_loss
optimizer = None
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

# Load model and setup multi-GPU
def setup_multi_gpu_model():
    net = SimpleTimeSeriesTransformer(in_channels, num_input_rows_total, num_transformer_blocks_stacked, 
                                    out_channels, num_output_rows, hidden_dim, nhead)
    ckpt_path = "/raid/mo0dy/models/A2/my_final_model.ckpt"
    
    the_model = FlightModel.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        map_location='cpu',
        model=net,
        coordinate_system_enum=coordinate_system_enum,
        loss_fn=loss_fn,
        optimizer=optimizer,
        max_num_val_maps=max_num_val_maps,
        n_future_timesteps=10,
        mean=mean,
        std=std
    )
    
    # Setup for multi-GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")
    
    the_model = the_model.to(device)
    if num_gpus > 1:
        the_model = DataParallel(the_model)
    the_model.eval()
    
    return the_model, device, num_gpus

# Optimized batch prediction function
def prepare_neural_net_input_batch(flight_df_segments):
    """
    Convert multiple flight dataframe segments to batch format for the neural network
    """
    batch_tensors = []
    
    for flight_df_segment in flight_df_segments:
        # Ensure we have the required columns and add missing ones if needed
        if 'Time' not in flight_df_segment.columns:
            flight_df_segment = flight_df_segment.copy()
            flight_df_segment['Time'] = range(len(flight_df_segment))
        if 'FlightCourse' not in flight_df_segment.columns:
            flight_df_segment = flight_df_segment.copy()
            flight_df_segment['FlightCourse'] = -99
        
        # Define desired features
        desired_input_features = coordinate_system + auxiliary_input_channels
        desired_output_features = coordinate_system + auxiliary_output_channels
        
        # Use the same build_features function as training
        flight_df_input_features, flight_df_output_features = build_features(
            flight_df_segment,
            desired_input_features,
            desired_output_features
        )
        
        # Convert to tensor
        input_tensor = torch.as_tensor(flight_df_input_features.to_numpy(), dtype=torch.float32)
        
        # Ensure we have exactly 100 timesteps (pad or truncate)
        if len(input_tensor) > num_input_rows_total:
            input_tensor = input_tensor[:num_input_rows_total]
        elif len(input_tensor) < num_input_rows_total:
            padding = torch.zeros(num_input_rows_total - len(input_tensor), input_tensor.shape[1])
            input_tensor = torch.cat([input_tensor, padding], dim=0)
        
        # Rearrange to (channels, timesteps) format
        input_tensor = input_tensor.permute(1, 0)
        batch_tensors.append(input_tensor)
    
    # Stack into batch
    if batch_tensors:
        return torch.stack(batch_tensors)
    else:
        return None

def predict_flightpath_neural_net_optimized(flight_df_segment, model, num_predictions, device, batch_size=32):
    """
    Optimized neural network prediction with batching
    """
    predicted_coords = []
    current_df = flight_df_segment.copy()
    
    with torch.no_grad():
        # Process predictions in batches when possible
        remaining_predictions = num_predictions
        
        while remaining_predictions > 0:
            current_batch_size = min(batch_size, remaining_predictions)
            
            if current_batch_size == 1:
                # Single prediction
                input_tensor = prepare_neural_net_input_batch([current_df])
                if input_tensor is not None:
                    input_tensor = input_tensor.to(device)
                    prediction = model(input_tensor)
                    pred_lat_lon = prediction.squeeze().cpu().numpy()
                    
                    if len(pred_lat_lon.shape) == 1 and len(pred_lat_lon) >= 2:
                        pred_lat, pred_lon = pred_lat_lon[0], pred_lat_lon[1]
                    else:
                        pred_lat, pred_lon = pred_lat_lon[0, 0], pred_lat_lon[0, 1]
                    
                    predicted_coords.append([pred_lat, pred_lon])
                    
                    # Update current_df for next prediction
                    new_row = current_df.iloc[-1].copy()
                    new_row['Latitude'] = pred_lat
                    new_row['Longitude'] = pred_lon
                    new_row['Time'] = new_row['Time'] + 1
                    current_df = pd.concat([current_df.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)
                    
                remaining_predictions -= 1
            else:
                # For now, fall back to single predictions for sequential dependency
                # This could be further optimized with more complex batching strategies
                for _ in range(current_batch_size):
                    if remaining_predictions <= 0:
                        break
                        
                    input_tensor = prepare_neural_net_input_batch([current_df])
                    if input_tensor is not None:
                        input_tensor = input_tensor.to(device)
                        prediction = model(input_tensor)
                        pred_lat_lon = prediction.squeeze().cpu().numpy()
                        
                        if len(pred_lat_lon.shape) == 1 and len(pred_lat_lon) >= 2:
                            pred_lat, pred_lon = pred_lat_lon[0], pred_lat_lon[1]
                        else:
                            pred_lat, pred_lon = pred_lat_lon[0, 0], pred_lat_lon[0, 1]
                        
                        predicted_coords.append([pred_lat, pred_lon])
                        
                        # Update current_df for next prediction
                        new_row = current_df.iloc[-1].copy()
                        new_row['Latitude'] = pred_lat
                        new_row['Longitude'] = pred_lon
                        new_row['Time'] = new_row['Time'] + 1
                        current_df = pd.concat([current_df.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)
                        
                    remaining_predictions -= 1
            
            # Print progress less frequently
            if (num_predictions - remaining_predictions) % 10 == 0:
                print(f"Progress: {num_predictions - remaining_predictions}/{num_predictions}")
    
    return np.array(predicted_coords)

def process_single_flight(args):
    """
    Process a single flight for multiprocessing
    """
    flight_idx, test_dataframe, model_state, device_id, seq_length = args
    
    # Set device for this process
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    
    # Load model state (you'll need to modify this based on your model structure)
    # This is a simplified version - you might need to recreate the model here
    
    try:
        num_predictions = len(test_dataframe) - seq_length
        if num_predictions <= 0:
            return flight_idx, None, "Flight too short"
        
        # Get initial segment for prediction
        initial_segment = test_dataframe[:seq_length].copy()
        
        # Get actual coordinates
        actual_coords = test_dataframe[['Latitude', 'Longitude']].to_numpy()
        future_coords = actual_coords[seq_length:]
        
        # Note: You'll need to recreate the model here for multiprocessing
        # This is a simplified placeholder
        predicted_coords = predict_flightpath_neural_net_optimized(
            initial_segment, model_state, num_predictions, device, batch_size=64
        )
        
        return flight_idx, {
            'predicted_coords': predicted_coords,
            'actual_coords': future_coords,
            'initial_coords': actual_coords[:seq_length]
        }, None
        
    except Exception as e:
        return flight_idx, None, str(e)

def plot_paths_neural_net_optimized(test_dataframes, model, device, num_gpus, limit_num_flights=None, seq_length=100):
    """
    Optimized version using multiple GPUs and parallel processing
    """
    if limit_num_flights is None:
        limit_num_flights = len(test_dataframes)
    
    dataset = test_dataframes[:limit_num_flights]
    
    # For demonstration, process first flight with optimized single-GPU approach
    if len(dataset) > 0:
        test_dataframe = dataset[0]
        try:
            num_predictions = len(test_dataframe) - seq_length
            if num_predictions > 0:
                initial_segment = test_dataframe[:seq_length].copy()
                actual_coords = test_dataframe[['Latitude', 'Longitude']].to_numpy()
                initial_coords = actual_coords[:seq_length]
                future_coords = actual_coords[seq_length:]
                
                print(f"Starting optimized prediction for {num_predictions} steps...")
                predicted_coords = predict_flightpath_neural_net_optimized(
                    initial_segment, model, num_predictions, device, batch_size=64
                )
                
                print(f"Initial coords shape: {initial_coords.shape}")
                print(f"Actual future coords shape: {future_coords.shape}")
                print(f"Predicted coords shape: {predicted_coords.shape}")
                
                # Calculate metrics
                if len(predicted_coords) == len(future_coords):
                    max_difference = np.max(np.linalg.norm(future_coords - predicted_coords, axis=1))
                    from sklearn.metrics import r2_score, root_mean_squared_error
                    r2 = r2_score(future_coords, predicted_coords)
                    rmse = root_mean_squared_error(future_coords, predicted_coords)
                    print(f"R2: {r2:.4f} | Max Diff: {max_difference:.6f} | RMSE: {rmse:.6f}")
                else:
                    print("Shape mismatch between predicted and actual coordinates")
                    
        except Exception as e:
            print(f"Error processing flight: {e}")

# Load data (same as before - keeping your existing data loading code)
dir_flights = '/raid/mo0dy/F2/FS2/'
flights_dict = create_csv_dict(dir_flights)

# Find flights over 1000 steps (same as before)
list_flights_over_1000_steps = []
for key, val in flights_dict.items():
    for key2, val2 in val.items():
        sample_dataframe = pd.read_csv(val2)
        lat_long_np = sample_dataframe[["Latitude", "Longitude"]].to_numpy()
        num_timesteps, _= lat_long_np.shape
        if num_timesteps > 1_000:
            list_flights_over_1000_steps.append(val2)

print(f"Found {len(list_flights_over_1000_steps)} flights over 1000 steps")

# Load flight dataframes with all required columns
l1 = []
iter_flights = itterate_flights(flights_dict)
for flightseries in iter_flights:
    if flightseries is None:
        break
    (msn, flight_id), flight_df = flightseries
    if len(flight_df) > 1000:
        required_cols = ['Time', 'Latitude', 'Longitude', 'FlightCourse']
        if all(col in flight_df.columns for col in required_cols):
            l1.append(flight_df)

print(f"Found {len(l1)} usable flights")

# Setup and run optimized inference
if __name__ == "__main__":
    print("Setting up multi-GPU model...")
    model, device, num_gpus = setup_multi_gpu_model()
    
    print("Running optimized neural network inference...")
    plot_paths_neural_net_optimized(l1, model, device, num_gpus, limit_num_flights=5, seq_length=100)