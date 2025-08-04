#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from eval_metrics import *

# In[ ]:


# from enum import Enum

# class CoordinateEnum(Enum):
#     LatLongCoordinates = 1
#     ComplexCoordinates = 2
#     EmbeddingCoordinates = 3
#     PolarCoordinates = 4

#     def __str__(self):
#         return self.name

#     @classmethod
#     def convert(cls, item):
#         print(f"Attempting to convert: {item} (type: {type(item)})")
        
#         if isinstance(item, cls):
#             print("It's already a CoordinateEnum.")
#             return item
#         if isinstance(item, str) and item in cls.__members__:
#             print(f"Found '{item}' in __members__.")
#             return cls.__members__[item]
#         if isinstance(item, int):
#             for member in cls:
#                 if member.value == item:
#                     print(f"Integer {item} matches {member}.")
#                     return member
#         raise KeyError(f"'{item}' is not a valid {cls.__name__}, item is of type: {type(item)}")

#     @classmethod
#     def convert_to_str(cls, item):
#         return str(cls.convert(item))

# # Usage examples
# print(CoordinateEnum.convert_to_str("LatLongCoordinates"))  # Should print 'LatLongCoordinates'
# print(CoordinateEnum.convert_to_str(CoordinateEnum.LatLongCoordinates))  # Should print 'LatLongCoordinates'
# print(CoordinateEnum.convert_to_str(1))  # Should print 'LatLongCoordinates'


# In[ ]:


coordinate_system_enum = CoordinateEnum.LatLongCoordinates

experiment_name_cnn = "test60"


num_output_rows = 1
num_res_blocks = 4
intermediate_channels = 64
dilation = [1 if i % 2 == 0 else 2 for i in range(num_res_blocks + 2)]
kernel_size = 27
use_bn_norm=False
stride=1
bias=True


map_location = 'cpu'
loss_fn = torch.nn.functional.mse_loss
optimizer = None 
max_num_val_maps = 8
n_future_timesteps = 10
mean = None
std = None


coordinate_dictionary = {}
coordinate_dictionary["CoordinateEnum"] = coordinate_system_enum
coordinate_dictionary["auxiliary_input_channels"] =  [
                                                        "diff_time", 
                                                        "flight_course_corrected", 
                                                        "flight_course_unknown", 
                                                    ]
coordinate_dictionary["auxiliary_output_channels"] = []


net_parameters_LSTM = {}
net_parameters_LSTM["intermediate_channels"] = intermediate_channels
net_parameters_LSTM["num_res_blocks"] = num_res_blocks
net_parameters_LSTM["num_output_rows"] = num_output_rows
net_parameters_LSTM["dilation"] = dilation
net_parameters_LSTM["kernel_size"] = kernel_size
net_parameters_LSTM["use_bn_norm"] = use_bn_norm
net_parameters_LSTM["stride"] = stride
net_parameters_LSTM["bias"] = bias


model_params = {}
model_params["map_location"] = map_location
model_params["loss_fn"] = loss_fn
model_params["optimizer"] = optimizer
model_params["max_num_val_maps"] = max_num_val_maps
model_params["n_future_timesteps"] = n_future_timesteps
model_params["mean"] = mean
model_params["std"] = std


model_dictionary = {}
model_dictionary["coordinate_dictionary"] = coordinate_dictionary
model_dictionary["NetEnum"] = NetEnum.SimpleCNN
model_dictionary["net_parameters"] = net_parameters_LSTM
model_dictionary["model_params"] = model_params

model = get_specific_eval_model(model_dictionary, experiment_name=experiment_name_cnn, models_dir=None, use_model_at_index=-2)



# In[ ]:


coordinate_dictionary = model_dictionary['coordinate_dictionary']

coordinate_system_enum = coordinate_dictionary["CoordinateEnum"]     #CoordinateEnum.LatLongCoordinates
coordinate_system = helper_get_coordinate_system_based_on_enum(coordinate_enum=coordinate_system_enum)
auxiliary_input_channels = coordinate_dictionary["auxiliary_input_channels"]
auxiliary_output_channels = coordinate_dictionary["auxiliary_output_channels"]

num_workers = 1
num_input_rows_total = 100
min_rows_input = 100
num_output_rows = 1


individual_flights_dir = "/Users/aleksandranikevich/Desktop/AircraftTrajectory/data/Individual_Flights/"
flight_dfs = create_csv_dict(individual_flights_dir)

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





#desired_keys = ['150171', '164344', '123483', '90107', '134434', '146014', '180338', '68524', '89816']
desired_keys = ['146014', '180338']
eval_flights_dfs = {key: flight_dfs[key] for key in desired_keys if key in flight_dfs}

flight_dictionary_pre_loaded = False

num_predict_steps = 10
len_coordinate_system  = len(coordinate_system)
desired_input_features = coordinate_system  + auxiliary_input_channels
desired_output_features = coordinate_system  + auxiliary_output_channels
flightpath_iter = flightpath_iterator(flights_dict=eval_flights_dfs, 
                        flight_dictionary_pre_loaded=flight_dictionary_pre_loaded, 
                        desired_input_features=desired_input_features, 
                        desired_output_features=desired_output_features, 
                        min_rows_input=min_rows_input, 
                        num_input_rows_total=num_input_rows_total, 
                        num_output_rows=num_predict_steps, 
                        len_coordinate_system=len_coordinate_system, 
                        shuffle_flights = False, 
                        shuffle_chunks = False, 
                        bool_yield_meta_flightpath = True, 
                        force_new_flightpath_every_val_step = False
                        )

# In[ ]:


eval_dict, flightpaths = eval_over_flightpaths(model, flightpath_iter, num_predict_steps, break_after_index = 3)

# In[ ]:


len(eval_dict[list(eval_dict.keys())[0]]['prediction_model_0'])

# In[ ]:


dict_overall_eval, dict_overall_eval_arrays = get_final_eval_metrics_all_models(eval_dict, flightpaths_dict=flightpaths)

# In[ ]:


dict_overall_eval

# In[ ]:


final_metrics_dict, final_array_metrics_dict = get_overall_error_metrics(dict_overall_eval, dict_overall_eval_arrays)

# In[ ]:


fig = plot_metrics_from_dict(final_array_metrics_dict)

# In[ ]:


final_array_metrics_dict

# In[ ]:


final_metrics_dict
