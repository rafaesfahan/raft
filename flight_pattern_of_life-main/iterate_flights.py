import numpy as np
import torch
import random
import pandas as pd 

from coordinate_transform import *


import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from enum import Enum
from copy import deepcopy

class EnumInterpolate(Enum):
    LINEAR = 'linear'
    CUBIC = 'cubic'



def itterate_flights(flights_dict, flight_dictionary_pre_loaded = False, shuffle = True):
    """
    Itterator for the flight data as a seperate modular component
    loop over items in the pre-processed flight dictionary
    """
    list_key_pairs = []
    for key_flight, dict_flight in flights_dict.items():
        for key_id, dict_id in dict_flight.items():
            list_key_pairs.append((key_flight, key_id))

    if shuffle:
        random.shuffle(list_key_pairs)

    for key_pair in list_key_pairs:
        flight_key = key_pair[0]
        id_key = key_pair[1]

        flight_df = flights_dict[flight_key][id_key]
        if not flight_dictionary_pre_loaded: # if flight dictionary is not pre-loaded, it points to the individual csv of the flight
            flight_df = pd.read_csv(flight_df)

        yield key_pair, flight_df

    yield None
    


def build_features(flight_df, desired_input_features, desired_output_features):
    """
    Build the desired input features of the dataframe, (ex: convert lat/long into 'complex-number' representations)

    Args:
        fligh_df, Pandas dataframe: datafame of flight data
        desired_features_input_and_output, List[str]: list of desired input features (columns present in processed dataframe)
    """


    desired_features_input_and_output = list(set(desired_input_features + desired_output_features))

    time_df = flight_df["Time"].diff(-1)
    # Latitude to complex number
    lat_complex_x, lat_complex_y = angle_to_complex_encoding(flight_df["Latitude"])
    # Longitude to complex number
    long_complex_x, long_complex_y = angle_to_complex_encoding(flight_df["Longitude"])

    # New column where flight course -99 (unknown course) is replaced with zeros
    # and another column is added to indicate 1 or zero if replacement took place
    flight_course_corrected = flight_df['FlightCourse'].replace([-99, -99.0], 0) / 360.0
    flight_course_unknown   = flight_df['FlightCourse'].apply(lambda x: 1 if x in [-99, -99.0] else 0)

    # Add these new feature columns to dataframe
    flight_df["diff_time"] = time_df
    flight_df["lat_complex_x"] = lat_complex_x
    flight_df["lat_complex_y"] = lat_complex_y
    flight_df["long_complex_x"] = long_complex_x
    flight_df["long_complex_y"] = long_complex_y
    flight_df["flight_course_corrected"] = flight_course_corrected
    flight_df["flight_course_unknown"] = flight_course_unknown

    # convert relevent columns into tensor
    flight_df_desired_features_input_and_output = flight_df[desired_features_input_and_output]
    flight_df_input_features = flight_df_desired_features_input_and_output[desired_input_features]
    flight_df_output_features = flight_df_desired_features_input_and_output[desired_output_features]

    return flight_df_input_features, flight_df_output_features



def interpolate_flight_data(flight_df, method=EnumInterpolate.LINEAR):
    flight_df = flight_df.sort_values(by='Time')
    # Ensure 'Time' is the index
    flight_df.set_index('Time', inplace=True)
    
    # Create a new time range with 1-second intervals
    new_time_index = pd.RangeIndex(start=flight_df.index.min(), stop=flight_df.index.max() + 1, step=1)
    new_flight_df = flight_df.reindex(new_time_index)
    
    # Interpolate Latitude, Longitude, and Altitude
    if method == EnumInterpolate.LINEAR:
        new_flight_df[['Latitude', 'Longitude', 'Altitude']] = new_flight_df[['Latitude', 'Longitude', 'Altitude']].interpolate(method='linear')
    elif method == EnumInterpolate.CUBIC:
        for col in ['Latitude', 'Longitude', 'Altitude']:
            cs = CubicSpline(flight_df.index, flight_df[col])
            new_flight_df[col] = cs(new_time_index)
    
    # Forward fill auxiliary columns
    aux_columns = flight_df.columns.difference(['Latitude', 'Longitude', 'Altitude'])
    new_flight_df[aux_columns] = new_flight_df[aux_columns].fillna(method='ffill')
    
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
    
    if shuffle_chunks:
        np.random.shuffle(indecies_tensor_chunks_shuffled)

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
                    force_new_flightpath_every_val_step = False
                    ):
    
    iter_flights = itterate_flights(flights_dict, flight_dictionary_pre_loaded = flight_dictionary_pre_loaded, shuffle = shuffle_flights)
    for flightseries in iter_flights:
        if flightseries is None:
            break

        (msn, flight_id), flight_df = flightseries 
        # Interpolate to every second for consistent timesteps
        flight_df = interpolate_flight_data(flight_df, method=EnumInterpolate.LINEAR)
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
                input_output_tensors_dict_dims_rearranged["meta_flightpath"] = flights_dict[msn][flight_id]
                if not flight_dictionary_pre_loaded:
                    input_output_tensors_dict_dims_rearranged["meta_flightpath"] = pd.read_csv(flights_dict[msn][flight_id])
            input_output_tensors_dict_dims_rearranged["meta_msn"] = msn
            input_output_tensors_dict_dims_rearranged["meta_flight_id"] = flight_id

            
            yield input_output_tensors_dict_dims_rearranged

            # If Validation dataloader, get new flight every val step
            if force_new_flightpath_every_val_step:
                break

    yield None
    

    