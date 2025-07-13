import torch
import torch.utils.data as data
import copy
import pandas as pd

from iterate_flights import filter_df_by_aircraft_speed, itterate_flights, flight_tensor_chunk_itterator, build_features, helper_create_necessary_superset_columns, interpolate_flight_data, EnumInterpolate
from coordinate_transform import CoordinateEnum, helper_get_coordinate_system_based_on_enum

from torch_utils import min_max_normalize, min_max_unnormalize


class FlightSeriesDataset(data.IterableDataset):
    def __init__(self, 
                 flights_dict, 
                 num_input_rows_total = 1000, 
                 min_rows_input = 10, 
                 num_output_rows = 1, 
                 coordinate_system_enum = None,
                 auxiliary_input_channels = None,
                 auxiliary_output_channels = None,
                 force_new_flightpath_every_val_step = False, 
                 flight_dictionary_pre_loaded = False, 
                 bool_yield_meta_flightpath = True, 
                 filter_flight_keys = None,
                 dataset_wide_normalization_dict = None):

        super(FlightSeriesDataset, self).__init__()
        
        self.flights_dict = flights_dict

        self.num_input_rows_total = num_input_rows_total
        self.min_rows_input = min_rows_input
        self.num_output_rows = num_output_rows


        self.coordinate_system_enum = coordinate_system_enum
        self.coordinate_system = helper_get_coordinate_system_based_on_enum(coordinate_system_enum)
        self.len_coordinate_system = len(self.coordinate_system )
        self.auxiliary_input_channels = auxiliary_input_channels
        self.auxiliary_output_channels = auxiliary_output_channels

        

        self.desired_input_features = self.coordinate_system  + self.auxiliary_input_channels
        self.desired_output_features = self.coordinate_system  + auxiliary_output_channels
        self.desired_features_input_and_output = self.coordinate_system  + self.auxiliary_input_channels + auxiliary_output_channels

        self.num_input_features = len(self.desired_input_features)
        self.num_output_features = len(self.desired_output_features)

        self.force_new_flightpath_every_val_step = force_new_flightpath_every_val_step
        self.flight_dictionary_pre_loaded = flight_dictionary_pre_loaded
        
        self.filter_flight_keys = filter_flight_keys
        self.bool_yield_meta_flightpath = bool_yield_meta_flightpath


        # option to normalize each dataframe of lat/longs using the dataset-wide (global) min-max lat-long values to keep eveyrthing consistent
        self.dataset_wide_normalization_dict = dataset_wide_normalization_dict
        self.speed_threshold = 20



    def __iter__(self):
        """
        Indefinite iterator over dataset, trining loop of pytorch lightning model

        Workflow goes like this: given dictionary of flight dataframes (with keys being mission numbers) we itterate over individual flights
        within the dictionary. Then we feed in a chunk of the flight data (as the total flight timeseries may have a large number of rows). 
        This is done using the self.num_input_rows_total and the self.min_rows_input parameters. self.num_input_rows_total x Features is 
        the input shape of the tensor to the model, while the self.min_rows_input parameter determines the minimum number of rows needed to 
        train on or make predictions (all other input rows will be zero-padded) 


        Note, Typical columns of interest for model input
        (diff) Time , Latitude	Longitude	Altitude   GroundSpeed   FlightCourse (-99 if unknown)
        """

        while True:
            iter_flights = itterate_flights(self.flights_dict, filter_flight_keys = self.filter_flight_keys)
            for flightseries in iter_flights:
                if flightseries is None:
                    break


                (msn, flight_id), flight_df = flightseries # mission number and dataframe describing path of the flight

                # Interpolate dataset to have consistent time till next sample
                flight_df = interpolate_flight_data(flight_df, method=EnumInterpolate.LINEAR)

                # filter flight df by plane speed
                flight_df = filter_df_by_aircraft_speed(flight_df, speed_threshold = self.speed_threshold) # filter out any rows in the dataframe speed below 20.0 km/h

                # Normalize dataframe by global min/max values if the option to do so is specified
                if self.dataset_wide_normalization_dict is not None:
                    flight_df = min_max_normalize(flight_df, self.dataset_wide_normalization_dict)
                
                # Build any extra features we would like to use here
                flight_df_input_features, flight_df_output_features= build_features(flight_df, 
                                                                                    self.desired_input_features, 
                                                                                    self.desired_output_features)

                # TODO ADD AUGMENTATIONS! 
                
                full_input_tensor = torch.as_tensor(flight_df_input_features.to_numpy(), dtype=torch.float32)
                full_output_tensor = torch.as_tensor(flight_df_output_features.to_numpy(), dtype=torch.float32)

                # now create iterator to get individual "chunks" of this timeseries data
                iter_tensor_chunks = flight_tensor_chunk_itterator( input_features_tensor_full = full_input_tensor, 
                                                                    output_features_tensor_full = full_output_tensor, 
                                                                    min_rows_input = self.min_rows_input, 
                                                                    num_input_rows_total = self.num_input_rows_total, 
                                                                    num_output_rows = self.num_output_rows, 
                                                                    register_buffer_input = None, 
                                                                    register_buffer_output = None,
                                                                  )

                for input_output_tensors_dict in iter_tensor_chunks:
                    if input_output_tensors_dict is None:
                        break

                    
                    # # Get last coordinate (coordinate at last timestep) | Normalization
                    last_coor = copy.deepcopy(input_output_tensors_dict["input_tensor"][-1, :self.len_coordinate_system])
                    input_output_tensors_dict["normalization_tensor"] = torch.unsqueeze(last_coor, dim=0)  # torch.unsqueeze(t, dim=0)


                    # re-arrange tensor to be Batch x Channels x Timesteps (pytorch lightning will take care of stacking the batch dim)
                    excempt_from_permute_keys = set(["zero_pad_rows", "chunk_index"])
                    #keys_permute = set(input_output_tensors_dict.keys()) - excempt_from_permute_keys
                    input_output_tensors_dict_dims_rearranged = {key: (input_output_tensors_dict[key].permute(1, 0) if key not in excempt_from_permute_keys else input_output_tensors_dict[key]) for key in input_output_tensors_dict.keys()}


                    # Add meta information
                    if self.bool_yield_meta_flightpath:
                        input_output_tensors_dict_dims_rearranged["meta_flightpath"] = self.flights_dict[msn][flight_id]
                        if not self.flight_dictionary_pre_loaded:
                            file = self.flights_dict[msn][flight_id]
                            if ".csv" in file:
                                input_output_tensors_dict_dims_rearranged["meta_flightpath"] = pd.read_csv(file)
                            elif ".parquet" in file:
                                input_output_tensors_dict_dims_rearranged["meta_flightpath"] = pd.read_parquet(file)
                            else:
                                raise NotImplementedError(f"reader for file type {file} not implemented")
                            
                            # Built the necessary features for plotting / any other analysis
                            input_output_tensors_dict_dims_rearranged["meta_flightpath"] = helper_create_necessary_superset_columns(input_output_tensors_dict_dims_rearranged["meta_flightpath"])

                    input_output_tensors_dict_dims_rearranged["meta_msn"] = msn

                    
                    yield input_output_tensors_dict_dims_rearranged

                    # If Validation dataloader, get new flight every val step
                    if self.force_new_flightpath_every_val_step:
                        break




