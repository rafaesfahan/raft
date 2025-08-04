import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
from pytorch_lightning import Trainer
from net_transformer import SimpleTimeSeriesTransformer
from datamodule import Datamodule
from model import FlightModel
from file_parsing_utils import create_csv_dict
from coordinate_transform import CoordinateEnum, helper_get_coordinate_system_based_on_enum
from iterate_flights import itterate_flights, build_features  # Added build_features import

from enum import Enum

# Model configuration (same as before)
auxiliary_input_channels = [
    "diff_time",
    "flight_course_corrected", 
    "flight_course_unknown",
]
auxiliary_output_channels = []  # Added this - you'll need to define it
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

# Load model
net = SimpleTimeSeriesTransformer(in_channels, num_input_rows_total, num_transformer_blocks_stacked, out_channels, num_output_rows, hidden_dim, nhead)
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
the_model.eval()  # Set to evaluation mode

# Rest of imports (same as before)
from typing import List
import pandas as pd
import numpy as np
import os
import folium
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import random
from folium_utils import create_folium_map

# Helper functions (same as before)
def simple_folium_map(some_flight_df):
    flightpath_compleate = some_flight_df[["Latitude", "Longitude"]].to_numpy()
    
    min_lat, max_lat = np.min(flightpath_compleate[:, 0]), np.max(flightpath_compleate[:, 0])
    min_long, max_long = np.min(flightpath_compleate[:, 1]), np.max(flightpath_compleate[:, 1])
    print("debug simple folium map: ", min_lat, min_long, max_lat, max_long)
    m = create_folium_map(min_lat, min_long, max_lat, max_long, border_lat_prop=0.15, border_long_prop=0.03, tiles=None)
    folium.PolyLine(locations=flightpath_compleate, color='black', weight=2.5, opacity=1).add_to(m)
    return m

# Load flight data (same as before)
dir_flights = '/raid/mo0dy/F2/FS2/'
flights_dict = create_csv_dict(dir_flights)

# Find flights over 1000 steps (same as before)
list_flights_over_1000_steps = []
for key, val in flights_dict.items():
    for key2, val2 in val.items():
        sample_dataframe = pd.read_csv(val2)
        lat_long_np = sample_dataframe[["Latitude", "Longitude"]].to_numpy()
        num_timesteps, _ = lat_long_np.shape
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
    
    if len(flight_df) > 1000:  # Only use long flights
        # Check if required columns exist
        required_cols = ['Time', 'Latitude', 'Longitude', 'FlightCourse']
        if all(col in flight_df.columns for col in required_cols):
            l1.append(flight_df)

print(f"Found {len(l1)} usable flights")

# NEW: Function to convert raw flight data to model input format
def prepare_neural_net_input(flight_df_segment):
    """
    Convert a flight dataframe segment to the 5-channel format expected by the neural network
    """
    # Ensure we have the required columns and add missing ones if needed
    if 'Time' not in flight_df_segment.columns:
        # Create dummy time column if missing
        flight_df_segment = flight_df_segment.copy()
        flight_df_segment['Time'] = range(len(flight_df_segment))
    
    if 'FlightCourse' not in flight_df_segment.columns:
        # Create dummy flight course if missing
        flight_df_segment = flight_df_segment.copy()
        flight_df_segment['FlightCourse'] = -99  # Unknown course
    
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
        # Zero pad to reach required length
        padding = torch.zeros(num_input_rows_total - len(input_tensor), input_tensor.shape[1])
        input_tensor = torch.cat([input_tensor, padding], dim=0)
    
    # Rearrange to (channels, timesteps) format
    input_tensor = input_tensor.permute(1, 0)
    
    return input_tensor

# NEW: Neural network prediction function
def predict_flightpath_neural_net(flight_df_segment, model, num_predictions):
    """
    Use neural network to predict flight path
    """
    predicted_coords = []
    current_df = flight_df_segment.copy()
    
    with torch.no_grad():
        for i in range(num_predictions):
            print(f"\nStep {i}/{num_predictions}\n")
            # Prepare input tensor
            input_tensor = prepare_neural_net_input(current_df)
            
            # Add batch dimension
            input_tensor = input_tensor.unsqueeze(0)  # Shape: (1, 5, 100)
            
            # Get prediction
            prediction = model(input_tensor)  # Should output (1, 2, 1) for lat/lon
            
            # Extract lat/lon prediction
            pred_lat_lon = prediction.squeeze().cpu().numpy()  # Shape: (2,)
            
            # Handle complex coordinate conversion if needed
            # if len(pred_lat_lon) == 2:
                # pred_lat, pred_lon = pred_lat_lon[0], pred_lat_lon[1]
            # else:
                # If using complex encoding, you might need to convert back
            pred_lat, pred_lon = pred_lat_lon[0], pred_lat_lon[1]  # Adjust as needed
            
            predicted_coords.append([pred_lat, pred_lon])
            
            # Update current_df for next prediction
            # Add the predicted point and remove the oldest point
            new_row = current_df.iloc[-1].copy()
            new_row['Latitude'] = pred_lat
            new_row['Longitude'] = pred_lon
            new_row['Time'] = new_row['Time'] + 1  # Increment time
            
            # Add new row and remove first row to maintain window size
            current_df = pd.concat([current_df.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)
    
    return np.array(predicted_coords)

# Modified plot function for neural network
def plot_paths_neural_net(test_dataframes, model, limit_num_flights=None, seq_length=100):
    """
    For every test dataframe, predict the flightpath using neural network
    """
    if limit_num_flights is None:
        limit_num_flights = len(test_dataframes)
    
    fig, axes = plt.subplots(limit_num_flights, 1, figsize=(10, 5 * limit_num_flights))
    if limit_num_flights == 1:
        axes = [axes]
    
    dataset = test_dataframes[:limit_num_flights]
    
    for i, test_dataframe in enumerate(dataset):
        if i>0:
            break
        try:
            num_predictions = len(test_dataframe) - seq_length
            if num_predictions <= 0:
                print(f"Skipping flight {i}: too short")
                continue
                
            # Get initial segment for prediction
            initial_segment = test_dataframe[:seq_length].copy()
            
            # Get actual coordinates
            actual_coords = test_dataframe[['Latitude', 'Longitude']].to_numpy()
            initial_coords = actual_coords[:seq_length]
            future_coords = actual_coords[seq_length:]
            
            # Predict using neural network
            predicted_coords = predict_flightpath_neural_net(initial_segment, model, num_predictions)
            
            # print(f"Flight {i}:")
            print(f"  Initial coords shape: {initial_coords.shape}")
            print(f"  Actual future coords shape: {future_coords.shape}")
            print(f"  Predicted coords shape: {predicted_coords.shape}")
            
            # Calculate metrics
            if len(predicted_coords) == len(future_coords):
                max_difference = np.max(np.linalg.norm(future_coords - predicted_coords, axis=1))
                r2 = r2_score(future_coords, predicted_coords)
                rmse = root_mean_squared_error(future_coords, predicted_coords)
            else:
                max_difference = r2 = rmse = float('nan')
            
            # Plot
            axes[i].plot(initial_coords[:, 1], initial_coords[:, 0], label='Initial Actual Path', marker='o', color='blue')
            axes[i].plot(future_coords[:, 1], future_coords[:, 0], label='Actual Future Path', marker='o', color='green')
            axes[i].plot(predicted_coords[:, 1], predicted_coords[:, 0], label='Predicted Path', linestyle='--', marker='x', color='red')
            axes[i].set_title(f'Flight {i} - R2: {r2:.4f} | Max Diff: {max_difference:.6f} | RMSE: {rmse:.6f}')
            axes[i].set_xlabel('Longitude')
            axes[i].set_ylabel('Latitude')
            axes[i].legend()
            axes[i].grid(True)
            
        except Exception as e:
            print(f"Error processing flight {i}: {e}")
            axes[i].set_title(f'Flight {i} - Error: {str(e)}')
    
    plt.tight_layout()
    plt.show()

# Run the neural network inference
print("Running neural network inference...")
plot_paths_neural_net(l1, the_model, limit_num_flights=5, seq_length=100)