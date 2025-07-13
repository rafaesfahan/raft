
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


from iterate_flights import get_implied_origin_and_destination, itterate_flights, helper_determine_coordinate_in_box
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

# dir_flights = "/Users/aleksandranikevich/Desktop/AircraftTrajectory/data/Individual_Flights/"
# dir_flights = '/home/trumoo/Downloads/slock/flights/cleaned/cleaned/cleaned/fdir/'
dir_flights = '/home/trumoo/Downloads/slock/flights/USA/fdir/'
flights_dict = create_csv_dict(dir_flights, break_after=None)


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
iter_flights = itterate_flights(flights_dict, filter_flight_keys = filter_flight_keys)
for flightseries in iter_flights:
    if flightseries is None:
        break
    (msn, flight_id), flight_df = flightseries

    b1, b2 = get_implied_origin_and_destination(flight_df, airport2, airport4)
    if b1:
        l1.append(flight_df)
    if b2:
        l2.append(flight_df)

    if b1 and b2:
        target_dest_list.append(flight_df)

dummy = 1
len(l1), len(l2), len(target_dest_list)


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
    
    bool_origin_airport1 = helper_determine_coordinate_in_box(coor1, airport1)
    bool_origin_airport2 = helper_determine_coordinate_in_box(coor1, airport2)
    # print("0: ", bool_origin_airport1, bool_origin_airport2)
    
    bool_destination_airport1 = helper_determine_coordinate_in_box(coor2, airport1)
    bool_destination_airport2 = helper_determine_coordinate_in_box(coor2, airport2)
    # print("1: ", bool_destination_airport1, bool_destination_airport2)
    # print("coor2: ", coor2)
    # print("airport1: ", airport1)
    # print("airport2: ", airport2)
    
    bool_flight_from_airport1_to_airport2 = bool_origin_airport1 and bool_destination_airport2
    bool_flight_from_airport2_to_airport1 = bool_origin_airport2 and bool_destination_airport1
    
    #return bool_origin_airport1, bool_origin_airport2, bool_destination_airport1, bool_destination_airport2
    return bool_flight_from_airport1_to_airport2, bool_flight_from_airport2_to_airport1


plt.figure()
idx = 10
df = l1[idx]
lat = df["Latitude"]
long = df["Longitude"]


# print(get_implied_origin_and_destination(df, airport2, airport4))

# m = simple_folium_map(df)
# m


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
    for coords in dataframe_list:

        flightpath_length = len(coords)
        if flightpath_length <= seq_length:  # If fewer coordinates then desired seq length, 
            continue

        for i in range(flightpath_length - seq_length):
            input_seq = coords[i: i + seq_length]
            output_seq = coords[i + seq_length]
            sequences.append(input_seq)
            targets.append(output_seq)

    sequences = np.array(sequences)
    targets = np.array(targets)
    return sequences, targets


        
X_train_knn, y_train_knn = create_knn_sequence(train_set, seq_len)
X_train_knn.shape, y_train_knn.shape


knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_knn.reshape(X_train_knn.shape[0], -1), y_train_knn)


def predict_flightpath(actual_coords, knn_model, scaler, num_predictions):
    """
    Takes a dataframe or numpy array of (scaled) Lat/Long coordinates and feeds them through the KNN model to get the number 

    ** ASSUMES THE CORRECT seq_length of the numpy array of dataframe being fed it
    """
    predicted_coords = []
    current_sequence = actual_coords.copy() if isinstance(actual_coords, np.ndarray) else actual_coords.to_numpy().copy()

    for j in range(num_predictions):
        pred = knn_model.predict(current_sequence.reshape(1, -1))
        predicted_coords.append(pred[0])
        current_sequence = np.vstack([current_sequence[1:], pred])

    predicted_coords_rescaled = scaler.inverse_transform(predicted_coords)
    return predicted_coords_rescaled

def plot_paths(test_dataframes, knn_model, scaler, limit_num_flights = None, seq_length=100):
    """
    For every test dataframe, predict the flightpath given the first N timesteps (seq_length), 
    then plot the flightpaht trajectory for each one (ground truth vs predicted)
    """
    if limit_num_flights is None:
        limit_num_flights = len(test_dataframes)
    fig, axes = plt.subplots(limit_num_flights, 1, figsize=(10, 5 * limit_num_flights))

    for i, test_dataframe in enumerate(test_dataframes[:limit_num_flights]):
        num_predictions = len(test_dataframe) - seq_length
        ###print("0 debug: ", test_dataframe.shape)
        actual_coords_rescaled =  scaler.inverse_transform(test_dataframe.copy())
        predicted_coords_rescaled = predict_flightpath(test_dataframe[:seq_length], knn_model, scaler, num_predictions)
        initial_coords_rescaled = actual_coords_rescaled[:seq_length]

        print("debug")
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

        
        #print(f"Flight {fltKey} - MSE: {mse:.4f}, Max Difference: {max_difference:.4f}, R^2: {r2:.4f}")


        
    plt.tight_layout()
    plt.show()


plot_paths(test_set, knn_model, scaler, limit_num_flights = 2, seq_length=seq_len)


    # test_flights = test_data['fltKey'].unique()[:num_flights]
    # for i, fltKey in enumerate(test_flights):
    #     flight_data = test_data[test_data['fltKey'] == fltKey]
    #     actual_coords = flight_data[['Latitude', 'Longitude']].values
    #     initial_coords = actual_coords[:seq_length]

    #     # Predict subsequent points based on the initial 400 coordinates
    #     predicted_coords = []
    #     current_sequence = initial_coords.copy()
    
    #     for j in range(seq_length, len(actual_coords)):
    #         pred = knn_model.predict(current_sequence.reshape(1, -1))
    #         predicted_coords.append(pred[0])
    #         current_sequence = np.vstack([current_sequence[1:], pred])
    
    #     actual_coords_rescaled = scaler.inverse_transform(actual_coords)
    #     initial_coords_rescaled = actual_coords_rescaled[:seq_length]
    #     predicted_coords_rescaled = scaler.inverse_transform(predicted_coords)


# num_flights is the number of flights printed
def plot_paths(test_data, linear_model, scaler, num_flights=3, seq_length=1000):
    fig, axes = plt.subplots(num_flights, 1, figsize=(10, 5 * num_flights))
    
    test_flights = test_data['fltKey'].unique()[:num_flights]
    for i, fltKey in enumerate(test_flights):
        flight_data = test_data[test_data['fltKey'] == fltKey]
        actual_coords = flight_data[['Latitude', 'Longitude']].values
        initial_coords = actual_coords[:seq_length]

        # Predict subsequent points based on the initial 1000 coordinates
        predicted_coords = []
        current_sequence = initial_coords.copy()
        
        for j in range(seq_length, len(actual_coords)):
            pred = linear_model.predict(current_sequence.reshape(1, -1))
            predicted_coords.append(pred[0])
            current_sequence = np.vstack([current_sequence[1:], pred])
        
        actual_coords_rescaled = scaler.inverse_transform(actual_coords)
        initial_coords_rescaled = actual_coords_rescaled[:seq_length]
        predicted_coords_rescaled = scaler.inverse_transform(predicted_coords)
        
        axes[i].plot(initial_coords_rescaled[:, 1], initial_coords_rescaled[:, 0], label='Initial Actual Path', marker='o')
        axes[i].plot(actual_coords_rescaled[seq_length:, 1], actual_coords_rescaled[seq_length:, 0], label='Actual Path', marker='o')
        axes[i].plot(predicted_coords_rescaled[:, 1], predicted_coords_rescaled[:, 0], label='Predicted Path', linestyle='--', marker='x')
        axes[i].set_title(f'Flight {fltKey}')
        axes[i].set_xlabel('Longitude')
        axes[i].set_ylabel('Latitude')
        axes[i].legend()
        axes[i].grid(True)

        mse = mean_squared_error(actual_coords_rescaled[seq_length:], predicted_coords_rescaled)
        max_difference = np.max(np.linalg.norm(actual_coords_rescaled[seq_length:] - predicted_coords_rescaled, axis=1))
        r2 = r2_score(actual_coords_rescaled[seq_length:], predicted_coords_rescaled)
        
        print(f"Flight {fltKey} - MSE: {mse:.4f}, Max Difference: {max_difference:.4f}, R^2: {r2:.4f}")
    
    plt.tight_layout()
    plt.show()

plot_paths(test_data, linear_model, scaler, num_flights=3, seq_length=1000)

seq_length = 1000

def create_knn_sequences(data, seq_length):
    sequences = []
    targets = []
    for fltKey in data['fltKey'].unique():
        flight_data = data[data['fltKey'] == fltKey]
        coords = flight_data[['Latitude', 'Longitude']].values
        if len(coords) > seq_length:
            for i in range(len(coords) - seq_length):
                sequences.append(coords[i:i+seq_length])
                targets.append(coords[i+seq_length])
    return np.array(sequences), np.array(targets)

X_train_knn, y_train_knn = create_knn_sequences(train_data, seq_length)

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_knn.reshape(X_train_knn.shape[0], -1), y_train_knn)


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Example list of DataFrames
df1 = pd.DataFrame({
    'Latitude': [40.7128, 34.0522],
    'Longitude': [-74.0060, -118.2437]
})

df2 = pd.DataFrame({
    'Latitude': [37.7749, 39.7392],
    'Longitude': [-122.4194, -104.9903]
})

dataframes = [df1, df2]

# Step 1: Combine all DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)

# Step 2: Fit the scaler on the combined DataFrame
scaler = MinMaxScaler()
scaler.fit(combined_df[['Latitude', 'Longitude']])

# Step 3: Scale each individual DataFrame using the fitted scaler
scaled_dataframes = [df.copy() for df in dataframes]
for df in scaled_dataframes:
    df[['Latitude', 'Longitude']] = scaler.transform(df[['Latitude', 'Longitude']])

# Step 4: Optional - Unscale each individual DataFrame if needed
unscaled_dataframes = [df.copy() for df in scaled_dataframes]
for df in unscaled_dataframes:
    df[['Latitude', 'Longitude']] = scaler.inverse_transform(df[['Latitude', 'Longitude']])

# Print the scaled DataFrames
print("Scaled DataFrames:")
for i, df in enumerate(scaled_dataframes):
    print(f"DataFrame {i+1}:\n", df, "\n")

# Print the unscaled DataFrames
print("Unscaled DataFrames:")
for i, df in enumerate(unscaled_dataframes):
    print(f"DataFrame {i+1}:\n", df, "\n")

