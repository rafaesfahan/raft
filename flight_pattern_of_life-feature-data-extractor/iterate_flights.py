import numpy as np
import torch
import random
import pandas as pd 

from coordinate_transform import *
from collections import defaultdict


import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from enum import Enum
from copy import deepcopy

from torch_utils import min_max_normalize, min_max_unnormalize

class EnumInterpolate(Enum):
    LINEAR = 'linear'
    CUBIC = 'cubic'



# def itterate_flights(flights_dict, flight_dictionary_pre_loaded = False, shuffle = True):
#     """
#     Itterator for the flight data as a seperate modular component
#     loop over items in the pre-processed flight dictionary
#     """
#     list_key_pairs = []
#     for key_flight, dict_flight in flights_dict.items():
#         for key_id, dict_id in dict_flight.items():
#             list_key_pairs.append((key_flight, key_id))

#     if shuffle:
#         random.shuffle(list_key_pairs)

#     for key_pair in list_key_pairs:
#         flight_key = key_pair[0]
#         id_key = key_pair[1]

#         flight_df = flights_dict[flight_key][id_key]
#         if not flight_dictionary_pre_loaded: # if flight dictionary is not pre-loaded, it points to the individual csv of the flight
#             flight_df = pd.read_csv(flight_df)

#         yield key_pair, flight_df

#     yield None


def itterate_flights(flights_dict, shuffle = True, filter_flight_keys = None):
    """
    Itterator for the flight data as a seperate modular component
    loop over items in the pre-processed flight dictionary

    Args:
        filter_flight_keys: tuple keys to use to filteter broader dataset (use these flights only if present)
    """

    if isinstance(flights_dict, dict) or isinstance(flights_dict, defaultdict):
        list_key_pairs = []
        for key_flight, dict_flight in flights_dict.items():
            for key_id, dict_id in dict_flight.items():
                list_key_pairs.append((key_flight, key_id))

        if filter_flight_keys is not None:
            list_key_pairs = [key_pair for key_pair in filter_flight_keys if key_pair in list_key_pairs] # in case any key gets deleted 

        if shuffle:
            random.shuffle(list_key_pairs)

        for key_pair in list_key_pairs:
            flight_key = key_pair[0]
            id_key = key_pair[1]

            flight_df = flights_dict[flight_key][id_key]
            # if not flight_dictionary_pre_loaded: # if flight dictionary is not pre-loaded, it points to the individual csv of the flight
            #     flight_df = pd.read_csv(flight_df)
            if isinstance(flight_df, str):
                if flight_df.endswith(".csv"):
                    flight_df = pd.read_csv(flight_df)
                elif flight_df.endswith(".parquet"):
                    flight_df = pd.read_parquet(flight_df)
                else:
                    raise NotImplementedError(f"{flight_df} is not implemented")

            yield key_pair, flight_df
    
    elif isinstance(flights_dict, list):
        # assume list of dataframes:
        for file in flights_dict:
            yield None, file

    yield None
    

def convert_longitude(longitude):
    """
    Longitude values jump from negative to positive near the Alaska Air defense identification zone (ADIZ)
    Use: 
        flight_df['longitude'] = flight_df['longitude'].apply(convert_longitude)
    """
    if longitude > 0.0:
        return longitude - 360
    else:
        return longitude

def helper_create_necessary_superset_columns(flight_df):
    """
    Create different representations of the coordinate dataframe
    Args:
        flight_df, Pandas Dataframe: Flight dataframe with Lat / Long coordinates columns 
    """
    
    time_col = "Time" if "Time" in flight_df.columns else "datetime" # Nasa Sherlock dataset has "Time" column | C1 data has 'datetime' column
    if time_col == "datetime":
        # Convert datetime to seconds from the lowest datetime
        flight_df[time_col] = (flight_df[time_col] - flight_df[time_col].min()).dt.total_seconds().astype(float)

    flight_df.columns = [col.capitalize() if col.lower() in ['latitude', 'longitude'] else col for col in flight_df.columns]

    # Longitude values jump from negative values to positive values when crossing the line near Alaska
    #########################flight_df['Longitude'] = flight_df['Longitude'].apply(convert_longitude)      # TODO TODO TODO 


    time_df = flight_df[time_col].diff(-1).fillna(0).replace([np.inf, -np.inf], 0)
    # Latitude to complex number
    lat_complex_x, lat_complex_y = angle_to_complex_encoding(flight_df["Latitude"])
    # Longitude to complex number
    long_complex_x, long_complex_y = angle_to_complex_encoding(flight_df["Longitude" ])

    # New column where flight course -99 (unknown course) is replaced with zeros
    # and another column is added to indicate 1 or zero if replacement took place
    if 'FlightCourse' in flight_df.columns:
        flight_course_corrected = flight_df['FlightCourse'].replace([-99, -99.0], 0) / 360.0
        flight_course_unknown   = flight_df['FlightCourse'].apply(lambda x: 1 if x in [-99, -99.0] else 0)

    # Add these new feature columns to dataframe
    flight_df["diff_time"] = time_df
    flight_df["lat_complex_x"] = lat_complex_x
    flight_df["lat_complex_y"] = lat_complex_y
    flight_df["long_complex_x"] = long_complex_x
    flight_df["long_complex_y"] = long_complex_y
    if 'FlightCourse' in flight_df.columns:
        flight_df["flight_course_corrected"] = flight_course_corrected
        flight_df["flight_course_unknown"] = flight_course_unknown

    return flight_df



def build_features(flight_df, desired_input_features, desired_output_features):
    """
    Build the desired input features of the dataframe, (ex: convert lat/long into 'complex-number' representations)

    Args:
        fligh_df, Pandas dataframe: datafame of flight data
        desired_features_input_and_output, List[str]: list of desired input features (columns present in processed dataframe)
    """
    flight_df = helper_create_necessary_superset_columns(flight_df)

    desired_features_input_and_output = list(set(desired_input_features + desired_output_features))
    # convert relevent columns into tensor
    flight_df_desired_features_input_and_output = flight_df[desired_features_input_and_output]
    flight_df_input_features = flight_df_desired_features_input_and_output[desired_input_features]
    flight_df_output_features = flight_df_desired_features_input_and_output[desired_output_features]

    return flight_df_input_features, flight_df_output_features


# import pandas as pd
# import numpy as np
# from math import radians, sin, cos, sqrt, atan2

# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371.0  # Radius of the Earth in kilometers
#     lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
#     c = 2 * atan2(sqrt(a), sqrt(1 - a))
#     distance = R * c
#     return distance

# def filter_df_by_aircraft_speed(df, speed_threshold = None):
#     df['Altitude_m'] = df['Altitude'] * 30.48  # Convert altitude from 100s of feet to meters
#     df['Latitude_shifted'] = df['Latitude'].shift()
#     df['Longitude_shifted'] = df['Longitude'].shift()
#     df['Altitude_m_shifted'] = df['Altitude_m'].shift()
    
#     df['Distance_km'] = df.apply(lambda row: haversine(row['Latitude'], row['Longitude'], 
#                                                        row['Latitude_shifted'], row['Longitude_shifted']), axis=1)
#     df['Altitude_diff'] = (df['Altitude_m'] - df['Altitude_m_shifted']) / 1000  # Convert altitude difference to kilometers
#     df['3D_Distance_km'] = np.sqrt(df['Distance_km']**2 + df['Altitude_diff']**2)  # 3D distance in kilometers
#     df['Time_diff'] = df['Time'].diff() / 3600  # Convert time difference to hours
#     df['Speed'] = df['3D_Distance_km'] / df['Time_diff']  # Speed in km/h
    
#     if speed_threshold is not None:
#         df = df[df['Speed'] >= speed_threshold]  # Filter rows based on speed threshold
#     return df





# def filter_df_by_aircraft_speed(df, speed_threshold=None):  # WORKS BUT LOTS OF WARNINGS ! 
#     if 'Altitude' in df.columns:
#         df['Altitude_m'] = df['Altitude'] * 30.48  # Convert altitude from 100s of feet to meters
#         df['Altitude_m_shifted'] = df['Altitude_m'].shift()
#         df['Altitude_diff'] = (df['Altitude_m'] - df['Altitude_m_shifted']) / 1000  # Convert altitude difference to kilometers
#         df['Altitude_present'] = True
#     else:
#         df['Altitude_m'] = 0  # Assume constant altitude if not present
#         df['Altitude_m_shifted'] = 0
#         df['Altitude_diff'] = 0
#         df['Altitude_present'] = False

#     df['Latitude_shifted'] = df['Latitude'].shift()
#     df['Longitude_shifted'] = df['Longitude'].shift()

#     df['Distance_km'] = df.apply(lambda row: haversine(row['Latitude'], row['Longitude'], 
#                                                        row['Latitude_shifted'], row['Longitude_shifted']), axis=1)
#     df['3D_Distance_km'] = np.sqrt(df['Distance_km']**2 + df['Altitude_diff']**2)  # 3D distance in kilometers

#     # Check the type of the 'Time' column and convert if necessary
#     if np.issubdtype(df['Time'].dtype, np.datetime64):
#         df['Time_diff'] = df['Time'].diff().dt.total_seconds() / 3600  # Convert time difference to hours
#     else:
#         df['Time_diff'] = df['Time'].diff() / 3600  # Assume time is in seconds and convert to hours

#     df['Speed'] = df['3D_Distance_km'] / df['Time_diff']  # Speed in km/h
    
#     if speed_threshold is not None:
#         df = df[df['Speed'] >= speed_threshold]  # Filter rows based on speed threshold
    
#     if not df['Altitude_present'].all():  # If Altitude column was not initially present
#         df.drop(['Altitude_m', 'Altitude_m_shifted', 'Altitude_diff', 'Altitude_present'], axis=1, inplace=True)
    
#     return df



def filter_df_by_aircraft_speed(df, speed_threshold=None):
    if 'Altitude' in df.columns:
        df['Altitude_m'] = df['Altitude'] * 30.48  # Convert altitude from 100s of feet to meters
        df['Altitude_m_shifted'] = df['Altitude_m'].shift()
        df['Altitude_diff'] = (df['Altitude_m'] - df['Altitude_m_shifted']) / 1000  # Convert altitude difference to kilometers
        df['Altitude_present'] = True
    else:
        df['Altitude_m'] = 0  # Assume constant altitude if not present
        df['Altitude_m_shifted'] = 0
        df['Altitude_diff'] = 0
        df['Altitude_present'] = False

    df['Latitude_shifted'] = df['Latitude'].shift()
    df['Longitude_shifted'] = df['Longitude'].shift()

    df['Distance_km'] = df.apply(lambda row: haversine(row['Latitude'], row['Longitude'], 
                                                       row['Latitude_shifted'], row['Longitude_shifted']), axis=1)
    df['3D_Distance_km'] = np.sqrt(df['Distance_km']**2 + df['Altitude_diff']**2)  # 3D distance in kilometers

    # Check the type of the 'Time' column and convert if necessary
    if np.issubdtype(df['Time'].dtype, np.datetime64):
        df['Time_diff'] = df['Time'].diff().dt.total_seconds() / 3600  # Convert time difference to hours
    else:
        df['Time_diff'] = df['Time'].diff() / 3600  # Assume time is in seconds and convert to hours

    df['Speed'] = df['3D_Distance_km'] / df['Time_diff']  # Speed in km/h
    
    if speed_threshold is not None:
        df = df[df['Speed'] >= speed_threshold].copy()  # Filter rows based on speed threshold and create a copy
    
    if not df['Altitude_present'].all():  # If Altitude column was not initially present
        df = df.drop(['Altitude_m', 'Altitude_m_shifted', 'Altitude_diff', 'Altitude_present'], axis=1)
    
    return df













# def interpolate_flight_data(flight_df, method=EnumInterpolate.LINEAR):
#     flight_df = flight_df.sort_values(by='Time')
#     # Ensure 'Time' is the index
#     flight_df.set_index('Time', inplace=True)
    
#     # Create a new time range with 1-second intervals
#     new_time_index = pd.RangeIndex(start=flight_df.index.min(), stop=flight_df.index.max() + 1, step=1)
#     print("Degub reindex: ")
#     print(flight_df)
#     print(flight_df.columns)
#     new_flight_df = flight_df.reindex(new_time_index)
    
#     # Interpolate Latitude, Longitude, and Altitude
#     if method == EnumInterpolate.LINEAR:
#         new_flight_df[['Latitude', 'Longitude', 'Altitude']] = new_flight_df[['Latitude', 'Longitude', 'Altitude']].interpolate(method='linear')
#     elif method == EnumInterpolate.CUBIC:
#         for col in ['Latitude', 'Longitude', 'Altitude']:
#             cs = CubicSpline(flight_df.index, flight_df[col])
#             new_flight_df[col] = cs(new_time_index)
    
#     # Forward fill auxiliary columns
#     aux_columns = flight_df.columns.difference(['Latitude', 'Longitude', 'Altitude'])
#     new_flight_df[aux_columns] = new_flight_df[aux_columns].fillna(method='ffill')
    
#     # Reset index to make Time a column again
#     new_flight_df.reset_index(inplace=True)
#     new_flight_df.rename(columns={'index': 'Time'}, inplace=True)
    
#     return new_flight_df


###################### Prev code does not take into account differnt formats of data in Sherlock vs C1

# def interpolate_flight_data(flight_df, method=EnumInterpolate.LINEAR):
#     # Rename 'datetime' column to 'Time' if it exists
#     if 'datetime' in flight_df.columns:
#         flight_df.rename(columns={'datetime': 'Time'}, inplace=True)
    
#     flight_df = flight_df.sort_values(by='Time')
    
#     # Drop duplicates to avoid reindexing issues
#     flight_df = flight_df.drop_duplicates(subset='Time')
    
#     # Detect the type of 'Time' column
#     if pd.api.types.is_datetime64_any_dtype(flight_df['Time']):
#         # Ensure 'Time' is the index
#         flight_df.set_index('Time', inplace=True)
        
#         # Resample the data to a 1-second interval, filling in missing values
#         new_flight_df = flight_df.resample('1S').mean()

#         # Interpolate Latitude, Longitude, and Altitude
#         if method == EnumInterpolate.LINEAR:
#             new_flight_df[['Latitude', 'Longitude', 'Altitude']] = new_flight_df[['Latitude', 'Longitude', 'Altitude']].interpolate(method='linear')
#         elif method == EnumInterpolate.CUBIC:
#             for col in ['Latitude', 'Longitude', 'Altitude']:
#                 cs = CubicSpline(flight_df.index.astype(float), flight_df[col])
#                 new_flight_df[col] = cs(new_flight_df.index.astype(float))
    
#     else:
#         # Ensure 'Time' is the index
#         flight_df.set_index('Time', inplace=True)
        
#         # Create a new time range with 1-second intervals
#         new_time_index = pd.RangeIndex(start=flight_df.index.min(), stop=flight_df.index.max() + 1, step=1)
        
#         # Reindex and fill missing values
#         new_flight_df = flight_df.reindex(new_time_index)
        
#         # Interpolate Latitude, Longitude, and Altitude
#         if method == EnumInterpolate.LINEAR:
#             new_flight_df[['Latitude', 'Longitude', 'Altitude']] = new_flight_df[['Latitude', 'Longitude', 'Altitude']].interpolate(method='linear')
#         elif method == EnumInterpolate.CUBIC:
#             for col in ['Latitude', 'Longitude', 'Altitude']:
#                 cs = CubicSpline(flight_df.index, flight_df[col])
#                 new_flight_df[col] = cs(new_time_index)
    
#     # Forward fill auxiliary columns
#     aux_columns = flight_df.columns.difference(['Latitude', 'Longitude', 'Altitude'])
#     new_flight_df[aux_columns] = new_flight_df[aux_columns].fillna(method='ffill')
    
#     # Reset index to make Time a column again
#     new_flight_df.reset_index(inplace=True)
#     new_flight_df.rename(columns={'index': 'Time'}, inplace=True)
    
#     return new_flight_df











def interpolate_flight_data(flight_df, method=EnumInterpolate.LINEAR):

    # capitalize column names if not done so already
    flight_df.columns = [col.capitalize() if col.lower() in ['latitude', 'longitude', 'altitude'] else col for col in flight_df.columns]


    # Rename 'datetime' column to 'Time' if it exists
    if 'datetime' in flight_df.columns:
        flight_df.rename(columns={'datetime': 'Time'}, inplace=True)
    
    flight_df = flight_df.sort_values(by='Time')
    
    # Drop duplicates to avoid reindexing issues
    flight_df = flight_df.drop_duplicates(subset='Time')
    
    # Columns to interpolate if they exist
    columns_to_interpolate = ['Latitude', 'Longitude', 'Altitude']
    existing_columns = [col for col in columns_to_interpolate if col in flight_df.columns]
    
    # Detect the type of 'Time' column
    if pd.api.types.is_datetime64_any_dtype(flight_df['Time']):
        # Ensure 'Time' is the index
        flight_df.set_index('Time', inplace=True)
        
        # # Resample the data to a 1-second interval, filling in missing values
        # new_flight_df = flight_df.resample('1S').mean()


        # Separate numerical and non-numerical columns
        numerical_cols = flight_df.select_dtypes(include='number')
        non_numerical_cols = flight_df.select_dtypes(exclude='number')

        # Resample numerical columns
        resampled_numerical = numerical_cols.resample('1S').mean()

        # Resample non-numerical columns using forward fill
        resampled_non_numerical = non_numerical_cols.resample('1S').ffill()

        # Combine the resampled numerical and non-numerical DataFrames
        new_flight_df = pd.concat([resampled_numerical, resampled_non_numerical], axis=1)





        # Interpolate existing columns
        if method == EnumInterpolate.LINEAR:
            new_flight_df[existing_columns] = new_flight_df[existing_columns].interpolate(method='linear')
        elif method == EnumInterpolate.CUBIC:
            for col in existing_columns:
                cs = CubicSpline(flight_df.index.astype(float), flight_df[col])
                new_flight_df[col] = cs(new_flight_df.index.astype(float))
    
    else:
        # Ensure 'Time' is the index
        flight_df.set_index('Time', inplace=True)
        
        # Create a new time range with 1-second intervals
        new_time_index = pd.RangeIndex(start=flight_df.index.min(), stop=flight_df.index.max() + 1, step=1)
        
        # Reindex and fill missing values
        new_flight_df = flight_df.reindex(new_time_index)
        
        # Interpolate existing columns
        if method == EnumInterpolate.LINEAR:
            new_flight_df[existing_columns] = new_flight_df[existing_columns].interpolate(method='linear')
        elif method == EnumInterpolate.CUBIC:
            for col in existing_columns:
                cs = CubicSpline(flight_df.index, flight_df[col])
                new_flight_df[col] = cs(new_time_index)
    
    # Forward fill auxiliary columns
    aux_columns = flight_df.columns.difference(columns_to_interpolate)

    new_flight_df[aux_columns] = new_flight_df[aux_columns].ffill()

    
    # Reset index to make Time a column again
    new_flight_df.reset_index(inplace=True)
    new_flight_df.rename(columns={'index': 'Time'}, inplace=True)
    
    return new_flight_df







def flight_tensor_chunk_itterator(input_features_tensor_full, 
                                  output_features_tensor_full, 
                                  min_rows_input, 
                                  num_input_rows_total, 
                                  num_output_rows, 
                                  register_buffer_input = None, 
                                  register_buffer_output = None,
                                  shuffle_chunks = True):
    """
    Take the tensors (both input and output and yield the next X-y input/output tensor chunks pair 
    Helpfull to have this iterator as a modular component to re-use in eval scripts

    input_features_tensor_full, Pytorch.Tensor: Full tensor of input featurures, will be use to feed N rows at a time
    output_features_tensor_full, Pytorch.Tensor: Full tensor of output features, will take the next N: N+num_output_rows rows as the output feature
    min_rows_input, int: minimum number of input rows to feed in
    num_input_rows_total, int: total chunk size
    num_output_rows, int: number of rows to output
    register_buffer, predefined tensor of zeros in pytorch-lightning, in this case used to be able to feed in tensor chunks that are 
                     smaller than num_input_rows_total (in this case tensor is min_rows_input x features zero padded) 

    shuffle_chunks, [Bool or numpy array]: if this is a numpy array then we are passing in specific indecies as a np array
    """

    assert input_features_tensor_full.shape[0] == output_features_tensor_full.shape[0], f"input and output tensors need to have the same number of rows, got {input_features_tensor_full.shape} for the input tensor and and {output_features_tensor_full.shape} for the output tensor"
    num_rows, num_features = input_features_tensor_full.shape
    _, num_output_features = output_features_tensor_full.shape

    num_tensor_chunks = num_rows - min_rows_input
    if num_tensor_chunks <= 0:
        yield None

    assert min_rows_input <= num_input_rows_total, f"(min_rows_input {min_rows_input}) minimum number of input rows (zero pad case) must be less than or equal to num_input_rows_total (got {num_input_rows_total})"
    zero_pad_rows = 0
    if min_rows_input < num_input_rows_total:
        zero_pad_rows = num_input_rows_total - min_rows_input # number of rows to zero pad if minimum number of input rows is less than the total number of input rows

        if register_buffer_input is None:
            #register_buffer_input = torch.zeros(zero_pad_rows, num_features)
            last_row_tensor = input_features_tensor_full[0].clone()
            register_buffer_input = last_row_tensor.unsqueeze(0).repeat(zero_pad_rows, 1)
        if register_buffer_output is None:
            register_buffer_output = torch.zeros(zero_pad_rows, num_output_features)
        
        zero_pad_input_tensor = register_buffer_input[:zero_pad_rows]
        zero_pad_output_tensor = register_buffer_output[:zero_pad_rows]

        input_features_tensor_full = torch.concat([zero_pad_input_tensor, input_features_tensor_full], dim=0)
        output_features_tensor_full = torch.concat([zero_pad_output_tensor, output_features_tensor_full], dim=0)

    # update the num_rows variable based on the zero-padded tensor
    num_rows, _ = input_features_tensor_full.shape

    num_tensor_chunks = num_rows - num_input_rows_total - num_output_rows
    indecies_tensor_chunks_shuffled = np.arange(num_tensor_chunks)
    
    if isinstance(shuffle_chunks, np.ndarray):
        # Special case if we want only specific indecies
        indecies_tensor_chunks_shuffled= shuffle_chunks
    elif isinstance(shuffle_chunks, bool):
        if shuffle_chunks:
            np.random.shuffle(indecies_tensor_chunks_shuffled)
    else:
        raise NotImplementedError(f"shuffle_chunks has to be bool or np.ndarray not {type(shuffle_chunks)}")

    for i in indecies_tensor_chunks_shuffled: #range(num_tensor_chunks):
        input_tensor_chunk = input_features_tensor_full[i: i + num_input_rows_total]
        output_tensor_chunk = output_features_tensor_full[i + num_input_rows_total: i + num_input_rows_total + num_output_rows] # future n rows

        dict_tensors = {
            "input_tensor": input_tensor_chunk, 
            "output_tensor": output_tensor_chunk, 
            "zero_pad_rows": zero_pad_rows, 
            "chunk_index": i + num_input_rows_total,  # location of current position of the aircraft 
        }

        yield dict_tensors

    yield None




def flightpath_iterator(flights_dict, 
                    flight_dictionary_pre_loaded, 
                    desired_input_features, 
                    desired_output_features, 
                    min_rows_input, 
                    num_input_rows_total, 
                    num_output_rows, 
                    len_coordinate_system, 
                    shuffle_flights, 
                    shuffle_chunks,
                    bool_yield_meta_flightpath = True, 
                    force_new_flightpath_every_val_step = False, 
                    interpolation_method = EnumInterpolate.LINEAR, 
                    speed_threshold = 20, 
                    dataset_wide_normalization_dict = None
                    ):
    
    
    iter_flights = itterate_flights(flights_dict, shuffle = shuffle_flights)
    for flightseries in iter_flights:
        if flightseries is None:
            break

        (msn, flight_id), flight_df = flightseries 

        # filter flight df by plane speed
        ###flight_df = filter_df_by_aircraft_speed(flight_df, speed_threshold = speed_threshold) # filter out any rows in the dataframe speed below 20.0 km/h
        # Interpolate to every second for consistent timesteps
        ###flight_df = interpolate_flight_data(flight_df, method=interpolation_method)
        flight_df_copy_for_return = deepcopy(flight_df)


        # Normalize dataframe by global min/max values if the option to do so is specified
        if dataset_wide_normalization_dict is not None:
            flight_df = min_max_normalize(flight_df, dataset_wide_normalization_dict)

        # Build the features
        flight_df_input_features, flight_df_output_features= build_features(flight_df, desired_input_features, desired_output_features)

        full_input_tensor = torch.as_tensor(flight_df_input_features.to_numpy(), dtype=torch.float32)
        full_output_tensor = torch.as_tensor(flight_df_output_features.to_numpy(), dtype=torch.float32)

        # now create iterator to get individual "chunks" of this timeseries data
        iter_tensor_chunks = flight_tensor_chunk_itterator( input_features_tensor_full = full_input_tensor, 
                                                            output_features_tensor_full = full_output_tensor, 
                                                            min_rows_input = min_rows_input, 
                                                            num_input_rows_total = num_input_rows_total, 
                                                            num_output_rows = num_output_rows, 
                                                            register_buffer_input = None, 
                                                            register_buffer_output = None,
                                                            shuffle_chunks=shuffle_chunks)

        for input_output_tensors_dict in iter_tensor_chunks:
            if input_output_tensors_dict is None:
                break


            # # Get last coordinate (coordinate at last timestep) | Normalization
            last_coor = input_output_tensors_dict["input_tensor"][-1, :len_coordinate_system].clone()
            input_output_tensors_dict["normalization_tensor"] = torch.unsqueeze(last_coor, dim=0)  # torch.unsqueeze(t, dim=0)


            # re-arrange tensor to be Batch x Channels x Timesteps (pytorch lightning will take care of stacking the batch dim)
            excempt_from_permute_keys = set(["zero_pad_rows", "chunk_index"])
            #keys_permute = set(input_output_tensors_dict.keys()) - excempt_from_permute_keys
            input_output_tensors_dict_dims_rearranged = {key: (input_output_tensors_dict[key].permute(1, 0) if key not in excempt_from_permute_keys else input_output_tensors_dict[key]) for key in input_output_tensors_dict.keys()}


            # Add meta information
            if bool_yield_meta_flightpath:
                input_output_tensors_dict_dims_rearranged["meta_flightpath"] = flight_df_copy_for_return
                # input_output_tensors_dict_dims_rearranged["meta_flightpath"] = flights_dict[msn][flight_id]
                # if not flight_dictionary_pre_loaded:
                #     input_output_tensors_dict_dims_rearranged["meta_flightpath"] = pd.read_csv(flights_dict[msn][flight_id])
            input_output_tensors_dict_dims_rearranged["meta_msn"] = msn
            input_output_tensors_dict_dims_rearranged["meta_flight_id"] = flight_id

            yield input_output_tensors_dict_dims_rearranged

            # If Validation dataloader, get new flight every val step
            if force_new_flightpath_every_val_step:
                break

    yield None






def helper_airport_dit_to_min_max_lat_long(airport_dict):
    lat1 = airport_dict["lat1"]
    lat2 = airport_dict["lat2"]
    long1 = airport_dict["long1"]
    long2 = airport_dict["long2"]
    
    min_lat = min(lat1, lat2)
    max_lat = max(lat1, lat2)
    min_long = min(long1, long2)
    max_long = max(long1, long2)
    
    return min_lat, max_lat, min_long, max_long


def helper_determine_coordinate_in_box(coor, airport_dict):
    min_lat, max_lat, min_long, max_long = helper_airport_dit_to_min_max_lat_long(airport_dict)
    coor_lat, coor_long = coor
    
    #print(coor_lat, min_lat, "|", coor_lat, max_lat, "|", coor_long, min_long, "|", coor_long, max_long)
    if coor_lat >= min_lat and coor_lat <= max_lat and coor_long >= min_long and coor_long <= max_long:
        return True
    return False


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
    
    bool_destination_airport1 = helper_determine_coordinate_in_box(coor2, airport1)
    bool_destination_airport2 = helper_determine_coordinate_in_box(coor2, airport2)
    
    bool_flight_from_airport1_to_airport2 = bool_origin_airport1 and bool_destination_airport2
    bool_flight_from_airport2_to_airport1 = bool_origin_airport2 and bool_destination_airport1
    
    #return bool_origin_airport1, bool_origin_airport2, bool_destination_airport1, bool_destination_airport2
    return bool_flight_from_airport1_to_airport2, bool_flight_from_airport2_to_airport1






def helper_get_min_max_quantiles_from_dataset(flights_dict, quantiles, limit_samples = None):
    """
    Helper function to 
    """
    all_latitudes = []
    all_longitudes = []

    iter_flights = itterate_flights(flights_dict, filter_flight_keys = None, shuffle= True)
    for idx_flight, flightseries in enumerate(iter_flights):
        if flightseries is None:
            break

        if limit_samples is not None:
            if idx_flight > limit_samples:
                break

        (msn, flight_id), flight_df = flightseries # mission number and dataframe describing path of the flight

        # Capitalize column names if they are not so already
        flight_df.columns = [col.capitalize() if col.lower() in ['latitude', 'longitude', 'altitude'] else col for col in flight_df.columns]

        all_latitudes.extend(flight_df['Latitude'].values)
        all_longitudes.extend(flight_df['Longitude'].values)


    all_latitudes = pd.Series(all_latitudes)
    all_longitudes = pd.Series(all_longitudes)

    # Calculate the quantiles for latitude and longitude
    latitude_quantiles = all_latitudes.quantile(quantiles)
    longitude_quantiles = all_longitudes.quantile(quantiles)

    return latitude_quantiles, longitude_quantiles

