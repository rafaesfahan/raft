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

the_model = FlightModel.load_from_checkpoint(checkpoint_path=ckpt_path, 
                                        map_location='cpu', 
                                        model=net,
                                        coordinate_system_enum=coordinate_system_enum,
                                        loss_fn=loss_fn, 
                                        optimizer=optimizer, 
                                        max_num_val_maps=max_num_val_maps, 
                                        n_future_timesteps=10, 
                                        mean=mean, 
                                        std=std)

# model = FlightModel(net, coordinate_system_enum, loss_fn, optimizer, max_num_val_maps = max_num_val_maps, n_future_timesteps = 10, mean=mean, std=std)
# model = model.load_from_checkpoint('/raid/mo0dy/models/A1/model.ckpt')

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
#from iterate_flights import itterate_flights, helper_determine_coordinate_in_box
from iterate_flights import itterate_flights
from file_parsing_utils import create_csv_dict

from folium_utils import create_folium_map

def simple_folium_map(some_flight_df):
    flightpath_compleate = some_flight_df[["Latitude", "Longitude"]].to_numpy()

    min_lat, max_lat = np.min(flightpath_compleate[:, 0]), np.max(flightpath_compleate[:, 0])
    min_long, max_long = np.min(flightpath_compleate[:, 1]), np.max(flightpath_compleate[:, 1])
    print("debug simple folium map: ", min_lat, min_long, max_lat, max_long)
    m = create_folium_map(min_lat, min_long, max_lat, max_long, border_lat_prop=0.15, border_long_prop=0.03, tiles=None) #"Cartodb dark_matter")
    folium.PolyLine(locations=flightpath_compleate, color='black', weight=2.5, opacity=1).add_to(m)
    return m

def helper_coors_to_airport_fence(coor1, coor2):
    d = {
    'long1': coor1[1],
    'lat1': coor1[0],
    'long2': coor2[1],
    'lat2': coor2[0],
    }
    return d

# Dulles Airport Flightpaths
coor1_dulles = 38.999581, -77.497870   # Geofence rectangle around Dulles Airport
coor2_dulles = 38.910523, -77.417198

# Atlanta airport
coor1_atlanta = 33.662806, -84.468856
coor2_atlanta = 33.613585, -84.392472

# New York Airport(s)
coor1_new_york = 40.787169, -73.906073
coor2_new_york = 40.614613, -73.724726

# Dallas / Fort Worth Airport
coor1_Dallas = 32.943417, -97.108142
coor2_Dallas = 32.839135, -96.975612

airport1 = helper_coors_to_airport_fence(coor1_dulles, coor2_dulles)
airport2 = helper_coors_to_airport_fence(coor1_atlanta, coor2_atlanta)
airport3 = helper_coors_to_airport_fence(coor1_new_york, coor2_new_york)
airport4 = helper_coors_to_airport_fence(coor1_Dallas, coor2_Dallas)

dir_flights = '/raid/mo0dy/F2/FS2/'
#flights_dict = create_csv_dict(dir_flights, break_after=None)
flights_dict = create_csv_dict(dir_flights)

list_flights_over_1000_steps = []
for key, val in flights_dict.items():
    for key2, val2 in val.items():
        sample_dataframe = pd.read_csv(val2)
        lat_long_np = sample_dataframe[["Latitude", "Longitude"]].to_numpy()
        num_timesteps, _ = lat_long_np.shape
        
        if num_timesteps > 1_000:
            list_flights_over_1000_steps.append(val2)

len(list_flights_over_1000_steps)


import random
random.shuffle(list_flights_over_1000_steps)
list_flights_over_1000_steps[:4]


filter_flight_keys = None

l1 = []
l2 = []
target_dest_list = []
iter_flights = itterate_flights(flights_dict)
for flightseries in iter_flights:
    if flightseries is None:
        break
    (msn, flight_id), flight_df = flightseries

    # b1, b2 = get_implied_origin_and_destination(flight_df, airport2, airport4)
    # if b1:
    if len(flight_df) > 1:
        l1.append(flight_df)
    # if b2:
    # l2.append(flight_df)

    # if b1 and b2:
    # target_dest_list.append(flight_df)

len(l1)#, len(l2), len(target_dest_list)


def get_implied_origin_and_destination(dataframe, airport1, airport2):
    """
    Let's get a list of implied flights from airport 1 to airport 2
    Args:
        airport 1/2: dict with lat1/long1/lat2/long2 coordinates
    """
    
    first_row_lat = dataframe.iloc[0]['Latitude']
    first_row_long = dataframe.iloc[0]['Longitude']
    last_row_lat = dataframe.iloc[-1]['Latitude']
    last_row_long = dataframe.iloc[-1]['Longitude']
    
    coor1 = (first_row_lat, first_row_long)
    coor2 = (last_row_lat, last_row_long)
    
    # bool_origin_airport1 = helper_determine_coordinate_in_box(coor1, airport1)
    # bool_origin_airport2 = helper_determine_coordinate_in_box(coor1, airport2)
    # # print("0: ", bool_origin_airport1, bool_origin_airport2)
    
    # bool_destination_airport1 = helper_determine_coordinate_in_box(coor2, airport1)
    # bool_destination_airport2 = helper_determine_coordinate_in_box(coor2, airport2)

    # bool_flight_from_airport1_to_airport2 = bool_origin_airport1 and bool_destination_airport2
    # bool_flight_from_airport2_to_airport1 = bool_origin_airport2 and bool_destination_airport1
    
    #return bool_origin_airport1, bool_origin_airport2, bool_destination_airport1, bool_destination_airport2
    # return bool_flight_from_airport1_to_airport2, bool_flight_from_airport2_to_airport1


plt.figure()
idx = 0
df = l1[idx]
lat = df["Latitude"]
long = df["Longitude"]

combined_df = pd.concat(l1, ignore_index=True)

# Create Min/Max Scaler
scaler = MinMaxScaler()
scaler.fit(combined_df[['Latitude', 'Longitude']])

# Create scaled df list
scaled_df_list = [scaler.transform(df[['Latitude', 'Longitude']]) for df in l1]
random.shuffle(scaled_df_list)
idx_split = int(0.8 * len(scaled_df_list))

# Get train / test sets
train_set = scaled_df_list[:idx_split]
test_set = scaled_df_list[idx_split:]

seq_len = 1000

def create_knn_sequence(dataframe_list, seq_length):
    sequences = []
    targets = []
    counter = 0
    for coords in dataframe_list:

        flightpath_length = len(coords)
        if flightpath_length <= seq_length:  # If fewer coordinates then desired seq length, 
            continue

        counter+=1
        for i in range(flightpath_length - seq_length):
            input_seq = coords[i: i + seq_length]
            output_seq = coords[i + seq_length]
            if counter % 20 == 0:
                sequences.append(input_seq)
                targets.append(output_seq)

    sequences = np.array(sequences)
    targets = np.array(targets)
    return sequences, targets


X_train_knn, y_train_knn = create_knn_sequence(train_set, seq_len)
X_train_knn.shape, y_train_knn.shape

knn_models = [KNeighborsRegressor(n_neighbors=5) for _ in range(12)]

for knn in knn_models:
    knn.fit(X_train_knn.reshape(X_train_knn.shape[0], -1), y_train_knn)

from concurrent.futures import ThreadPoolExecutor
import time, copy

def predict_flightpath(actual_coords, knn, scaler, num_predictions):

    """
    Takes a dataframe or numpy array of (scaled) Lat/Long coordinates and feeds them through the KNN model to get the number 

    ** ASSUMES THE CORRECT seq_length of the numpy array of dataframe being fed it
    """
    current_sequence = actual_coords.copy()
    current_sequence = torch.from_numpy(current_sequence)
    print(f"\n\n\n{the_model(torch.randn(7,5,100).float())}\n\n\n")
    print(f"\n\n\n INFERENCE RAN\n\n\n")
    exit()
    predicted_coords = []
    current_sequence = actual_coords.copy() if isinstance(actual_coords, np.ndarray) else actual_coords.to_numpy().copy()

    for j in range(num_predictions):
        pred = knn.predict(current_sequence.reshape(1, -1))
        predicted_coords.append(pred[0])
        current_sequence = np.vstack([current_sequence[1:], pred])

    predicted_coords_rescaled = scaler.inverse_transform(predicted_coords)
    return predicted_coords_rescaled

def plot_paths(test_dataframes, knn_models, scaler, limit_num_flights = None, seq_length=100):
    """
    For every test dataframe, predict the flightpath given the first N timesteps (seq_length), 
    then plot the flightpaht trajectory for each one (ground truth vs predicted)
    """

    if limit_num_flights is None:
        limit_num_flights = len(test_dataframes)

    fig, axes = plt.subplots(limit_num_flights, 1, figsize=(10, 5 * limit_num_flights))

    dataset = test_dataframes[:limit_num_flights]

    def predict(i, test_dataframe):
        k = knn_models[i%12]
        num_predictions = len(test_dataframe) - seq_length
        actual_coords_rescaled =  scaler.inverse_transform(test_dataframe.copy())
        predicted_coords_rescaled = predict_flightpath(test_dataframe[:seq_length], k, scaler, num_predictions)
        initial_coords_rescaled = actual_coords_rescaled[:seq_length]

        print(f"INDEX:{i}")
        print("actual_coords_rescaled: ", actual_coords_rescaled.shape)
        print("predicted_coords_rescaled: ", predicted_coords_rescaled.shape)
        print("initial_coords_rescaled: ", initial_coords_rescaled.shape)


        #mse = mean_squared_error(actual_coords_rescaled[seq_length:], predicted_coords_rescaled)
        max_difference = np.max(np.linalg.norm(actual_coords_rescaled[seq_length:] - predicted_coords_rescaled, axis=1))
        r2 = r2_score(actual_coords_rescaled[seq_length:], predicted_coords_rescaled)
        rmse = root_mean_squared_error(actual_coords_rescaled[seq_length:], predicted_coords_rescaled)

        axes[i].plot(initial_coords_rescaled[:, 1], initial_coords_rescaled[:, 0], label='Initial Actual Path', marker='o')
        axes[i].plot(actual_coords_rescaled[seq_length:, 1], actual_coords_rescaled[seq_length:, 0], label='Actual Path', marker='o')
        axes[i].plot(predicted_coords_rescaled[:, 1], predicted_coords_rescaled[:, 0], label='Predicted Path', linestyle='--', marker='x')
        axes[i].set_title(f'R2: {r2} | Max Difference: {max_difference} | RMSE: {rmse}')
        axes[i].set_xlabel('Longitude')
        axes[i].set_ylabel('Latitude')
        axes[i].legend()
        axes[i].grid(True)

    with ThreadPoolExecutor(max_workers=12) as executor:
        da = list(enumerate(dataset))
        predict(da[0][0], da[0][1])
        executor.map(lambda d: predict(*d), enumerate(dataset))
        #executor.map(lambda d: predict(*d), enumerate(dataset))
        #executor.map(lambda index, value: predict(index, value), *enumerate(dataset))

    plt.tight_layout()
    plt.show()


plot_paths(test_set, knn_models, scaler, limit_num_flights = 100, seq_length=seq_len)
