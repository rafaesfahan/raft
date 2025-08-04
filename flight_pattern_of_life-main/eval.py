
def get_model_paths(experiment_name, models_dir=None, use_model_at_index = None):
    if models_dir is None:
        models_dir = '/Users/aleksandranikevich/Desktop/AircraftTrajectory/flight_pattern_of_life-main/models/'

    experiment_dir = models_dir + experiment_name + "/"
    models_dir = experiment_dir + "models/"
    paths = glob(models_dir + "*.ckpt")

    def extract_step_number(path):
        match = re.search(r"step=(\d+)", path)
        return int(match.group(1)) if match else float('inf')  # Use 'inf' for 'last.ckpt' to sort it last

    sorted_paths = sorted(paths, key=extract_step_number)

    # return specific desired model at a particular index if this parameter is specified
    if use_model_at_index is not None:
        return sorted_paths[use_model_at_index]
    return sorted_paths



from enum import Enum
import inspect

class NetEnum(Enum):
    SimpleCNN = 1
    SimpleLSTM = 2
    SimpleTimeSeriesTransformer = 3

    def __str__(self):
        return self.name


def filter_params(func, param_dict):
    sig = inspect.signature(func)
    return {k: v for k, v in param_dict.items() if k in sig.parameters}


def get_specific_eval_model(model_dictionary, experiment_name, models_dir=None, use_model_at_index=-2):
    dir_model = get_model_paths(experiment_name, models_dir=models_dir, use_model_at_index = use_model_at_index)
    

    # Coordinate System Initialization
    coordinate_dictionary = model_dictionary['coordinate_dictionary']

    coordinate_system_enum = coordinate_dictionary["CoordinateEnum"]     #CoordinateEnum.LatLongCoordinates
    coordinate_system = helper_get_coordinate_system_based_on_enum(coordinate_enum=coordinate_system_enum)
    auxiliary_input_channels = coordinate_dictionary["auxiliary_input_channels"]
    auxiliary_output_channels = coordinate_dictionary["auxiliary_output_channels"]

    len_coordinate_system  = len(coordinate_system)
    desired_input_features = coordinate_system  + auxiliary_input_channels
    desired_output_features = coordinate_system  + auxiliary_output_channels
    desired_features_input_and_output = coordinate_system  + auxiliary_input_channels + auxiliary_output_channels

    num_input_features = len(desired_input_features)
    num_output_features = len(desired_output_features)

    in_channels = len(coordinate_system) + len(auxiliary_input_channels)
    out_channels = len(coordinate_system)


    # Net Initialization
    net_enum = model_dictionary["NetEnum"]
    net_parameters = model_dictionary["net_parameters"]
    net_parameters["in_channels"] = in_channels
    net_parameters["out_channels"] = out_channels
    if net_enum == NetEnum.SimpleCNN:
        from net import SimpleNet

        params_for_init = filter_params(SimpleNet.__init__, net_parameters)
        net = SimpleNet(**params_for_init)
    
    elif net_enum == NetEnum.SimpleLSTM:
        from net_lstm import SimpleLSTM

        params_for_init = filter_params(SimpleLSTM.__init__, net_parameters)
        net = SimpleLSTM(**params_for_init)

    elif net_enum == NetEnum.SimpleTimeSeriesTransformer:
        from net_transformer import SimpleTimeSeriesTransformer

        params_for_init = filter_params(SimpleTimeSeriesTransformer.__init__, net_parameters)
        net = SimpleTimeSeriesTransformer(**params_for_init)

    else:
        raise NotImplementedError


    # Model Initialization
    model_params = model_dictionary["model_params"]
    model_params["checkpoint_path"] = dir_model
    model_params["model"] = net
    model_params["coordinate_system_enum"] = coordinate_system_enum
    model_simple_cnn = FlightModel.load_from_checkpoint(**model_params)

    return model_simple_cnn


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


models_dir = '/Users/aleksandranikevich/Desktop/AircraftTrajectory/flight_pattern_of_life-main/models/'


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


desired_keys = ['150171', '164344', '123483', '90107', '134434', '146014', '180338', '68524', '89816']
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



def eval_over_flightpaths(some_model_list, flightpath_iter, num_predict_steps, break_after_index = None, models_names_list = None):

    if not isinstance(some_model_list, list):
        some_model_list = [some_model_list]

    results_dict = defaultdict(lambda: defaultdict(list))
    flightpaths = {}

    idx_flight = 0
    for idx_iter, d in enumerate(flightpath_iter):
        if d is None:
            break

        msn = d['meta_msn']
        flight_id = d['meta_flight_id']
        if (msn, flight_id) not in flightpaths:

            idx_flight += 1
            if break_after_index is not None:
                if idx_flight >= break_after_index:
                    break

            flightpaths[(msn, flight_id)] = copy.deepcopy(d['meta_flightpath'])

        # dict_keys(['input_tensor', 'output_tensor', 'zero_pad_rows', 'chunk_index', 'normalization_tensor', 'meta_flightpath', 'meta_msn', 'meta_flight_id'])
        input_tensor = d["input_tensor"]
        output_tensor = d["output_tensor"]

        input_tensor = torch.unsqueeze(input_tensor, dim=0)
        d["input_tensor"] = input_tensor

        # Iterate over all models that were passed in
        for index_model, some_model in enumerate(some_model_list):
            pass_in_dict = {"input_tensor": d["input_tensor"].clone(), "normalization_tensor": d["normalization_tensor"].clone()}
            iterative_predictions_tensor_np1 = iterative_path_predict(batch_dict=pass_in_dict, 
                                                                    model=some_model, 
                                                                    coordinate_system_enum=coordinate_system_enum, 
                                                                    num_predict_steps=num_predict_steps, 
                                                                    bool_normalize_center = True, 
                                                                    bool_with_eval_model = True
                                                                    )

            prediction_model_str = "prediction_model_" + str(index_model)
            if models_names_list is not None:
                prediction_model_str = models_names_list[index_model]
                
            results_dict[(msn, flight_id)][prediction_model_str].append(iterative_predictions_tensor_np1)

        results_dict[(msn, flight_id)]['chunk_index'].append(d['chunk_index'])
        results_dict[(msn, flight_id)]['ground_truth'].append(output_tensor.detach().cpu().numpy())

    print("idx iter: ", idx_iter)
    return results_dict, flightpaths



def get_final_eval_metrics_all_models(eval_dict, flightpaths_dict):
    """
    *Currently assumes Lat-Long coordinates
    """


    paths_we_evaled_keys_list = list(eval_dict.keys())
    models_and_other = list(eval_dict[paths_we_evaled_keys_list[0]].keys()) # Ex: dict_keys(['prediction_model_0', 'chunk_index', 'ground_truth'])
    all_model_keys = [model_name for model_name in models_and_other if "prediction_model_" in model_name]

    dict_overall_eval = {}
    dict_overall_eval_arrays = {}
    for flightpath_key in paths_we_evaled_keys_list:

        flightpath_dataframe = flightpaths_dict[flightpath_key]
        flightpath_np = flightpath_dataframe[["Latitude", "Longitude"]].to_numpy()
        flightpath_np = np.squeeze(flightpath_np)

        ground_truth_list = eval_dict[flightpath_key]["ground_truth"]
        num_samples = len(ground_truth_list)

        #dict_disance_overall = {model_key: 0.0 for model_key in all_model_keys}
        dict_errors = {model_key: defaultdict(float) for model_key in all_model_keys}
        dict_errors_arrs = {model_key: defaultdict(list) for model_key in all_model_keys}

        for sample_index in range(num_samples):
            for model_key in all_model_keys:
                ground_truth_array = eval_dict[flightpath_key]["ground_truth"][sample_index]
                model_prediction_array = eval_dict[flightpath_key][model_key][sample_index]
                ground_truth_array = np.squeeze(ground_truth_array)
                model_prediction_array = np.squeeze(model_prediction_array)

                # Sometimes there are no more samples in the ground truth array, we must then ignore the corresponding predictions
                if ground_truth_array.shape != model_prediction_array.shape:
                    continue

                error_distance_arr = haversine(ground_truth_array[0], ground_truth_array[1], model_prediction_array[0], model_prediction_array[1])

                error_distance_total = np.sum(error_distance_arr)
                dict_errors[model_key]["error_distance_overall"] += error_distance_total
                dict_errors_arrs[model_key]["error_distance_arr"].append(error_distance_arr)


                # Normalized errors
                chunk_idx = eval_dict[flightpath_key]["chunk_index"][sample_index] - 1
                current_position = flightpath_np[chunk_idx]
                current_lat = current_position[0]
                current_long = current_position[1]
                distance_traveled_arr = haversine(current_lat, current_long, ground_truth_array[0], ground_truth_array[1])
                
                normalized_error_arr = error_distance_arr / distance_traveled_arr
                # sometimes the distance traveled to the next timestep is insignificant, we want to filter NaNs and Infs that are caused by this
                normalized_error_arr = np.nan_to_num(normalized_error_arr, nan=0.0, posinf=0.0, neginf=0.0)

                if np.isinf(normalized_error_arr).any() or np.isnan(normalized_error_arr).any():
                    print('\n\n')
                    print("current_lat, current_long, ground_truth_array[0], ground_truth_array[1]: ", current_lat, current_long, ground_truth_array[0], ground_truth_array[1])
                    print("distance_traveled_arr: ", distance_traveled_arr)
                    print("normalized_error_arr: ", normalized_error_arr)

                error_distance_overall_normalized = np.sum(normalized_error_arr)
                dict_errors[model_key]["error_distance_overall_normalized"] += error_distance_overall_normalized
                dict_errors_arrs[model_key]["normalized_error_arr"].append(normalized_error_arr)

                _, num_predictions = ground_truth_array.shape
                dict_errors[model_key]["num_predictions_total"] += num_predictions

                mse_error_arr = 0.5*((ground_truth_array[0] - current_lat)**2.0 + (ground_truth_array[1] - current_long)**2.0)
                dict_errors[model_key]["mse_error_arr"] += np.sum(mse_error_arr)
                dict_errors_arrs[model_key]["mse_error_arr"].append(mse_error_arr)

                

        ###dict_errors_arrs = {key: np.stack(arr_list, axis=0) for key, arr_list in dict_errors_arrs.items()}

        dict_overall_eval[flightpath_key] = dict_errors
        dict_overall_eval_arrays[flightpath_key] = dict_errors_arrs

    
    return dict_overall_eval, dict_overall_eval_arrays



dict_overall_eval, dict_overall_eval_arrays = get_final_eval_metrics_all_models(eval_dict, flightpaths_dict=flightpaths)
dict_overall_eval_arrays
                
dict_overall_eval_arrays[('150171', 'SKW5664')]['prediction_model_0'].keys() # dict_keys(['error_distance_arr', 'normalized_error_arr', 'mse_error_arr'])
dict_overall_eval_arrays[('150171', 'SKW5664')]['prediction_model_0']['normalized_error_arr'][0].shape

# In[84]:


def get_overall_error_metrics(dict_overall_eval, dict_overall_eval_arrays):
    # first get overall metrics averaged out over all evaluations
    path_keys = list(dict_overall_eval.keys())
    model_keys = list(dict_overall_eval[path_keys[0]].keys())
    eval_metric_keys = list(dict_overall_eval[path_keys[0]][model_keys[0]])

    final_metrics_dict = defaultdict(lambda: defaultdict(float)) # averaged over all paths METRICS 
    num_paths = len(path_keys)
    for path_key in path_keys:
        for model_key in model_keys:
            for eval_metric_key in eval_metric_keys:
                if eval_metric_key == 'num_predictions_total':
                    continue
                num_predictions_total = dict_overall_eval[path_key][model_key]['num_predictions_total']
                inverse_num_paths = (1.0/num_paths)
                inverse_num_preditions_total = (1.0/num_predictions_total)
                final_metrics_dict[model_key][eval_metric_key] += inverse_num_paths *inverse_num_preditions_total * dict_overall_eval[path_key][model_key][eval_metric_key]

    # final result of this is nested dict of [model][eval_metric] with final metrics
    
    path_keys = list(dict_overall_eval_arrays.keys())
    model_keys = list(dict_overall_eval_arrays[path_keys[0]].keys())
    eval_metric_keys = list(dict_overall_eval_arrays[path_keys[0]][model_keys[0]])
    num_future_steps = dict_overall_eval_arrays[path_keys[0]][model_keys[0]][eval_metric_keys[0]][0].shape[0] # shape of the first array stored in this arrays nested dict
    final_array_metrics_dict = defaultdict(lambda: defaultdict(lambda: np.zeros(num_future_steps)))
    final_array_counts_dict = defaultdict(lambda: defaultdict(float))

    for path_key in path_keys:
        for model_key in model_keys:
            for eval_metric_key in eval_metric_keys:
                path_predictions_list_from_model = dict_overall_eval_arrays[path_key][model_key][eval_metric_key]
                for path in path_predictions_list_from_model:
                    # Each path is a numpy array
                    final_array_metrics_dict[model_key][eval_metric_key] += path
                    final_array_counts_dict[model_key][eval_metric_key] += 1.0

    for model_key in model_keys:
        for eval_metric_key in eval_metric_keys:
            sum_of_all_predictions_array = final_array_metrics_dict[model_key][eval_metric_key]
            num_predictions = final_array_counts_dict[model_key][eval_metric_key]
            final_array_metrics_dict[model_key][eval_metric_key] = sum_of_all_predictions_array / num_predictions

    return final_metrics_dict, final_array_metrics_dict
    

final_metrics_dict, final_array_metrics_dict = get_overall_error_metrics(dict_overall_eval, dict_overall_eval_arrays)


import matplotlib.pyplot as plt

def plot_metrics_from_dict(results_dict):
    model_keys = list(results_dict.keys())
    metric_keys = list(results_dict[model_keys[0]].keys())
    
    # Custom labels
    labels = {
        'error_distance_arr': "Absolute Error",
        'normalized_error_arr': "Absolute Error Normalized By Distance Flown",
        'mse_error_arr': "Mean Squared Error"
    }
    
    n_rows = len(model_keys)
    n_cols = len(metric_keys)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    for i, model_key in enumerate(model_keys):
        for j, metric_key in enumerate(metric_keys):
            ax = axes[i, j] if n_rows > 1 and n_cols > 1 else axes[max(i, j)]
            metric_array = results_dict[model_key][metric_key]
            ax.plot(metric_array)
            ax.set_xlabel("Timestep")
            ax.set_ylabel(labels.get(metric_key, metric_key))  # Use custom label or fallback to key
            ax.set_title(f"Timestep vs average {labels.get(metric_key, metric_key)}", fontsize=9)
    
    plt.tight_layout()
    #plt.show()
    return fig


fig = plot_metrics_from_dict(final_array_metrics_dict)

final_metrics_dict

final_array_metrics_dict

from collections import defaultdict

# Create a nested defaultdict
nested_default_dict = defaultdict(lambda: defaultdict(float))

# Example usage
nested_default_dict['outer_key1']['inner_key1'] += 1.5
nested_default_dict['outer_key2']['inner_key2'] += 2.5

import matplotlib.pyplot as plt
import numpy as np

def generate_colors(n):
    cmap = plt.cm.get_cmap('viridis', n)
    colors = [cmap(i) for i in range(cmap.N)]
    # Convert RGBA to hexadecimal for Folium
    hex_colors = [mcolors.to_hex(c[:3]) for c in colors]
    return hex_colors, colors

def eval_mapmaker(flightpath, eval_dict_at_flightpath = None, corresponding_color_list = None):
    flightpath_compleate = flightpath[["Latitude", "Longitude"]].to_numpy()
    min_lat, max_lat = np.min(flightpath_compleate[:, 0]), np.max(flightpath_compleate[:, 0])
    min_long, max_long = np.min(flightpath_compleate[:, 1]), np.max(flightpath_compleate[:, 1])

    m = create_folium_map(min_lat, min_long, max_lat, max_long, border_lat_prop=0.15, border_long_prop=0.03, tiles=None)
    folium.PolyLine(locations=flightpath_compleate, color='black', weight=2.5, opacity=1).add_to(m)

    if eval_dict_at_flightpath is None:
        return m
    
    model_keys = list(eval_dict_at_flightpath.keys())
    # Have to filter because thse keys will contain for example: dict_keys(['prediction_model_0', 'chunk_index', 'ground_truth'])
    model_keys = [model_name for model_name in model_keys if "prediction_model_" in model_name]

    if corresponding_color_list is None:
        # generate default color list
        N = len(model_keys)
        hex_colors, rgba_colors = generate_colors(N)
        corresponding_color_list = hex_colors
    
    for idx_model_key, model_predictions_key in enumerate(model_keys):
        model_predictions_list = eval_dict_at_flightpath[model_predictions_key]
        model_predictions_color = corresponding_color_list[idx_model_key]
        #print("debug model_predictions_key: ", model_predictions_key)
        for model_prediction in model_predictions_list:
            model_prediction = np.squeeze(model_prediction) # just in case we have [1 x ...] dimensionality
            #print("debug model prediction: ", model_prediction, type(model_prediction))
            #print(model_prediction.shape)
            folium.PolyLine(locations=model_prediction.T, color=model_predictions_color, weight=1.5, opacity=1).add_to(m)

    return m, corresponding_color_list


flightpath_key_example = ('150171', 'N9803H')
m, _ = eval_mapmaker(flightpath=flightpaths[flightpath_key_example], eval_dict_at_flightpath=eval_dict[flightpath_key_example])
m


eval_dict_at_fligthpath = eval_dict_at_flightpath=eval_dict[flightpath_key_example]
key = 'prediction_model_0'

predictions = eval_dict_at_flightpath[key]
num_correct = 0
for pred in predictions:
    if not isinstance(pred, np.int64):
        print(pred)
    else:
        num_correct += 1

print("num correct: ", num_correct)


tuple_key = list(results_dict.keys())[0]

color_cnn = "blue"
color_lstm = "orange"
color_transformer = 'green'



flightpath_compleate = flightpaths[tuple_key][["Latitude", "Longitude"]].to_numpy()
min_lat, max_lat = np.min(flightpath_compleate[:, 0]), np.max(flightpath_compleate[:, 0])
min_long, max_long = np.min(flightpath_compleate[:, 1]), np.max(flightpath_compleate[:, 1])

m = create_folium_map(min_lat, min_long, max_lat, max_long, border_lat_prop=0.15, border_long_prop=0.03, tiles=None) #"Cartodb dark_matter")
folium.PolyLine(locations=flightpath_compleate, color='black', weight=2.5, opacity=1).add_to(m)


num_predictions = len(results_dict[tuple_key]["prediction_cnn"])
for idx_pred in range(30, num_predictions, 10):
    cnn_prediction = results_dict[tuple_key]["prediction_cnn"][idx_pred][0]
    lstm_prediction = results_dict[tuple_key]["prediction_lstm"][idx_pred][0]
    transformer_prediction = results_dict[tuple_key]["prediction_trasformer"][idx_pred][0]

    folium.PolyLine(locations=cnn_prediction.T, color=color_cnn, weight=2.5, opacity=1).add_to(m)
    folium.PolyLine(locations=lstm_prediction.T, color=color_lstm, weight=2.5, opacity=1).add_to(m)
    folium.PolyLine(locations=transformer_prediction.T, color=color_transformer, weight=2.5, opacity=1).add_to(m)


import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Latitude is specified in degrees within the range [-90, 90]. 
# Longitude is specified in degrees within the range [-180, 180].

def degrees_to_radians(some_degrees_angle):
    radians_angle = some_degrees_angle * np.pi / 180.0
    return radians_angle

def radians_to_degrees(some_radians_angle):
    degrees_angle = some_radians_angle * 180.0 / np.pi
    return degrees_angle
    
def angle_to_complex_encoding(degrees):
    """
    Angle representation (lat/long) fed into models can have sudden sharp discontinuities, to aleviate this we can use a "complex number" 
    representation of each of the coordinate in the coordinate pair. Method transforms either Lattitude or Longgitude into "complex reperesentaion"
    """
    radians = degrees_to_radians(degrees)
    complex_x = np.cos(radians)
    complex_y = np.sin(radians)
    return complex_x, complex_y

def complex_number_to_degrees(complex_x, complex_y):
    radians = np.arctan2(complex_y, complex_x)
    degrees = radians_to_degrees(radians)
    ###degrees = (degrees + 360) % 360.0
    return degrees


import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the distance in kilometers from (lat1, lon1) to (lat2, lon2).
    Supports both scalar and numpy array inputs.
    """
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c

    return distance

#haversine(np.ones(10), np.ones(10)*1.1, np.ones(10)*1.2, np.ones(10)*1.3)
haversine(1.0, 1.1, np.ones(10)*1.2, np.ones(10)*1.3)


import inspect

d1 = {'a': 1, 'b': 2, 'c': 3}

def func1(a, b, d=4, e=5):
    return a + b + d + e

def func2(a, c):
    return a * c

# Function to filter dictionary based on function parameters
def filter_params(func, param_dict):
    sig = inspect.signature(func)
    return {k: v for k, v in param_dict.items() if k in sig.parameters}

# Using the filter_params function to get the correct parameters
params_func1 = filter_params(func1, d1)
params_func2 = filter_params(func2, d1)

result_of_func1 = func1(**params_func1)
result_of_func2 = func2(**params_func2)

print(result_of_func1)  # Output: 12 (1 + 2 + 4 + 5)
print(result_of_func2)  # Output: 3 (1 * 3)


def dummy_net(dummy_arg):
    return 0.01 * torch.ones((32, 2, 1))


def dummy_net2(dummy_arg):
    # Create a tensor with the desired values for timesteps
    timestep_values = torch.tensor([0.01, 0.02, 0.03])
    
    # Repeat the tensor for each batch and channel
    dummy_tensor = timestep_values.repeat(32, 2, 1)
    
    return dummy_tensor


import re
from glob import glob

def get_model_paths(experiment_name, models_dir=None, use_model_at_index = None):
    if models_dir is None:
        models_dir = '/Users/aleksandranikevich/Desktop/AircraftTrajectory/flight_pattern_of_life-main/models/'

    experiment_dir = models_dir + experiment_name + "/"
    models_dir = experiment_dir + "models/"
    paths = glob(models_dir + "*.ckpt")

    def extract_step_number(path):
        match = re.search(r"step=(\d+)", path)
        return int(match.group(1)) if match else float('inf')  # Use 'inf' for 'last.ckpt' to sort it last

    sorted_paths = sorted(paths, key=extract_step_number)

    # return specific desired model at a particular index if this parameter is specified
    if use_model_at_index is not None:
        return sorted_paths[use_model_at_index]
    return sorted_paths

# Example usage
sorted_paths = get_model_paths("test60")
sorted_paths[-2]


coordinate_system_enum = CoordinateEnum.LatLongCoordinates
coordinate_system = helper_get_coordinate_system_based_on_enum(coordinate_enum=coordinate_system_enum)

auxiliary_input_channels = [
                            "diff_time", 
                            "flight_course_corrected", 
                            "flight_course_unknown", 
                            ]

auxiliary_output_channels = []


len_coordinate_system  = len(coordinate_system)
desired_input_features = coordinate_system  + auxiliary_input_channels
desired_output_features = coordinate_system  + auxiliary_output_channels
desired_features_input_and_output = coordinate_system  + auxiliary_input_channels + auxiliary_output_channels

num_input_features = len(desired_input_features)
num_output_features = len(desired_output_features)


num_res_blocks = 4
intermediate_channels = 64
in_channels = len(coordinate_system) + len(auxiliary_input_channels)
out_channels = len(coordinate_system)


num_output_rows = 1
num_res_blocks = 4
intermediate_channels = 64
in_channels = len(coordinate_system) + len(auxiliary_input_channels)
out_channels = len(coordinate_system)

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




# In[5]:


max_num_val_maps = 8
loss_fn = torch.nn.functional.mse_loss
optimizer = None #                               TODO 
mean = None
std = None


model_simple_cnn = FlightModel.load_from_checkpoint(checkpoint_path=sorted_paths[-2], 
                    map_location='cpu', 
                    model=net, 
                    coordinate_system_enum=coordinate_system_enum, 
                    loss_fn=loss_fn, 
                    optimizer=optimizer, 
                    max_num_val_maps = max_num_val_maps, 
                    n_future_timesteps = 10, 
                    mean=mean, 
                    std=std)


num_lstm_layers = 12
hidden_size = 64
#net = SimpleLSTM(in_channels, num_lstm_layers, out_channels)
net_simple_lstm = SimpleLSTM(in_channels, num_lstm_layers, out_channels, hidden_size=hidden_size, out_timesteps=num_output_rows) 


num_transformer_blocks_stacked = 4
hidden_dim = 64
nhead = 8
num_input_rows_total = 100
net_simple_trasformer = SimpleTimeSeriesTransformer(in_channels, num_input_rows_total, num_transformer_blocks_stacked, out_channels, num_output_rows, hidden_dim, nhead)


model_path_lstm = get_model_paths("test62")[-2]
model_path_transformer = get_model_paths("test63")[-2]



model_simple_lstm = FlightModel.load_from_checkpoint(checkpoint_path=model_path_lstm, 
                    map_location='cpu', 
                    model=net_simple_lstm, 
                    coordinate_system_enum=coordinate_system_enum, 
                    loss_fn=loss_fn, 
                    optimizer=optimizer, 
                    max_num_val_maps = max_num_val_maps, 
                    n_future_timesteps = 10, 
                    mean=mean, 
                    std=std)


model_simple_trasformer = FlightModel.load_from_checkpoint(checkpoint_path=model_path_transformer, 
                    map_location='cpu', 
                    model=net_simple_trasformer, 
                    coordinate_system_enum=coordinate_system_enum, 
                    loss_fn=loss_fn, 
                    optimizer=optimizer, 
                    max_num_val_maps = max_num_val_maps, 
                    n_future_timesteps = 10, 
                    mean=mean, 
                    std=std)


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


desired_keys = ['150171', '164344', '123483', '90107', '134434', '146014', '180338', '68524', '89816']
eval_flights_dfs = {key: flight_dfs[key] for key in desired_keys if key in flight_dfs}


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
    
    iter_flights = itterate_flights(flights_dict, shuffle = shuffle_flights)
    for flightseries in iter_flights:
        if flightseries is None:
            break

        (msn, flight_id), flight_df = flightseries 
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


desired_keys = ['150171', '164344', '123483', '90107', '134434', '146014', '180338', '68524', '89816']
eval_flights_dfs = {key: flight_dfs[key] for key in desired_keys if key in flight_dfs}

flight_dictionary_pre_loaded = False

num_predict_steps = 10
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



model1 = model_simple_cnn
model2 = model_simple_lstm
model3 = model_simple_trasformer


results_dict = defaultdict(lambda: defaultdict(list))
###flightpaths = {(msn, flight_id): copy.deepcopy(d["meta_flightpath"])}
flightpaths = {}

idx_flight = 0
for idx_iter, d in enumerate(flightpath_iter):
    if d is None:
        break

    msn = d['meta_msn']
    flight_id = d['meta_flight_id']
    if (msn, flight_id) not in flightpaths:
        flightpaths[(msn, flight_id)] = copy.deepcopy(d['meta_flightpath'])


        idx_flight += 1
        if idx_flight >= 2:
            break


    # dict_keys(['input_tensor', 'output_tensor', 'zero_pad_rows', 'chunk_index', 'normalization_tensor', 'meta_flightpath', 'meta_msn', 'meta_flight_id'])
    input_tensor = d["input_tensor"]
    output_tensor = d["output_tensor"]



    input_tensor = torch.unsqueeze(input_tensor, dim=0)
    d["input_tensor"] = input_tensor



    pass_in_dict = {"input_tensor": d["input_tensor"].clone(), "normalization_tensor": d["normalization_tensor"].clone()}


    # Predictions from our CNN, LSTM, Transformer models
    iterative_predictions_tensor_np1 = iterative_path_predict(batch_dict=pass_in_dict, 
                                                             model=model1, 
                                                             coordinate_system_enum=coordinate_system_enum, 
                                                             num_predict_steps=num_predict_steps, 
                                                             bool_normalize_center = True, 
                                                             bool_with_eval_model = True
                                                             )
    
    pass_in_dict = {"input_tensor": d["input_tensor"].clone(), "normalization_tensor": d["normalization_tensor"].clone()}
    iterative_predictions_tensor_np2 = iterative_path_predict(batch_dict=pass_in_dict, 
                                                             model=model2, 
                                                             coordinate_system_enum=coordinate_system_enum, 
                                                             num_predict_steps=num_predict_steps, 
                                                             bool_normalize_center = True, 
                                                             bool_with_eval_model = True
                                                             )
    
    pass_in_dict = {"input_tensor": d["input_tensor"].clone(), "normalization_tensor": d["normalization_tensor"].clone()}
    iterative_predictions_tensor_np3 = iterative_path_predict(batch_dict=pass_in_dict, 
                                                             model=model3, 
                                                             coordinate_system_enum=coordinate_system_enum, 
                                                             num_predict_steps=num_predict_steps, 
                                                             bool_normalize_center = True, 
                                                             bool_with_eval_model = True)
    

    results_dict[(msn, flight_id)]["prediction_cnn"].append(iterative_predictions_tensor_np1)
    results_dict[(msn, flight_id)]["prediction_lstm"].append(iterative_predictions_tensor_np2)
    results_dict[(msn, flight_id)]["prediction_trasformer"].append(iterative_predictions_tensor_np3)
    results_dict[(msn, flight_id)]['chunk_index'].append(d['chunk_index'])
    results_dict[(msn, flight_id)]['ground_truth'].append(output_tensor.detach().cpu().numpy())


tuple_key = list(results_dict.keys())[0]
print("tuple key: ", tuple_key)
print("results_dict prd list:", len(results_dict[tuple_key]['prediction_cnn']))
print("flight df shape: ", flightpaths[tuple_key].shape)

tuple_key = list(results_dict.keys())[0]

color_cnn = "blue"
color_lstm = "orange"
color_transformer = 'green'

flightpath_compleate = flightpaths[tuple_key][["Latitude", "Longitude"]].to_numpy()
min_lat, max_lat = np.min(flightpath_compleate[:, 0]), np.max(flightpath_compleate[:, 0])
min_long, max_long = np.min(flightpath_compleate[:, 1]), np.max(flightpath_compleate[:, 1])

m = create_folium_map(min_lat, min_long, max_lat, max_long, border_lat_prop=0.15, border_long_prop=0.03, tiles=None) #"Cartodb dark_matter")
folium.PolyLine(locations=flightpath_compleate, color='black', weight=2.5, opacity=1).add_to(m)


num_predictions = len(results_dict[tuple_key]["prediction_cnn"])
for idx_pred in range(30, num_predictions, 10):
    cnn_prediction = results_dict[tuple_key]["prediction_cnn"][idx_pred][0]
    lstm_prediction = results_dict[tuple_key]["prediction_lstm"][idx_pred][0]
    transformer_prediction = results_dict[tuple_key]["prediction_trasformer"][idx_pred][0]

    folium.PolyLine(locations=cnn_prediction.T, color=color_cnn, weight=2.5, opacity=1).add_to(m)
    folium.PolyLine(locations=lstm_prediction.T, color=color_lstm, weight=2.5, opacity=1).add_to(m)
    folium.PolyLine(locations=transformer_prediction.T, color=color_transformer, weight=2.5, opacity=1).add_to(m)


dist_errors_cnn_list_of_lists = []
dist_errors_lstm_list_of_lists = []
dist_errors_transformer_list_of_lists = []
ground_truth_distance_list_of_lists = []

cnn_mse_error_list_of_lists = []
lstm_mse_error_list_of_lists = []
transformer_mse_error_list_of_lists = []

# plots of error
num_predictions = len(results_dict[tuple_key]["prediction_cnn"])
flightpath_compleate = flightpaths[tuple_key][["Latitude", "Longitude"]].to_numpy()
#for idx_pred in range(30, num_predictions-11, 10):
for idx_pred in [600]:
    cnn_prediction = results_dict[tuple_key]["prediction_cnn"][idx_pred][0]
    lstm_prediction = results_dict[tuple_key]["prediction_lstm"][idx_pred][0]
    transformer_prediction = results_dict[tuple_key]["prediction_trasformer"][idx_pred][0]

    ground_truth = results_dict[tuple_key]["ground_truth"][idx_pred]
    idx_current_position = results_dict[tuple_key]['chunk_index'][idx_pred]
    current_position = flightpath_compleate[idx_current_position]
    current_lat = current_position[0]
    current_long = current_position[1]

    # haversine(lat1, lon1, lat2, lon2)
    
    num_pred = cnn_prediction.shape[1]
    dist_errors_cnn_list = []
    dist_errors_lstm_list = []
    dist_errors_transformer_list = []
    ground_truth_distance_list = []

    cnn_mse_error_list = []
    lstm_mse_error_list = []
    transformer_mse_error_list = []

    for i in range(num_pred):

        next_ground_truth_coor = flightpath_compleate[idx_current_position + i+1] # + 1 because i starts with zero 
        next_ground_truth_coor_lat = next_ground_truth_coor[0]
        next_ground_truth_coor_long = next_ground_truth_coor[1]

        pred_lat_cnn = cnn_prediction[0][i]
        pred_long_cnn = cnn_prediction[1][i]

        pred_lat_lstm = lstm_prediction[0][i]
        pred_long_lstm = lstm_prediction[1][i]

        pred_lat_transformer = transformer_prediction[0][i]
        pred_long_transformer = transformer_prediction[1][i]

        dist_error_cnn = haversine(current_lat, current_long, pred_lat_cnn, pred_long_cnn)
        dist_error_lstm = haversine(current_lat, current_long, pred_lat_lstm, pred_long_lstm)
        dist_error_trasformer = haversine(current_lat, current_long, pred_lat_transformer, pred_long_transformer)
        dist_ground_truth = haversine(current_lat, current_long, next_ground_truth_coor_lat, next_ground_truth_coor_long)

        dist_errors_cnn_list.append(dist_error_cnn)
        dist_errors_lstm_list.append(dist_error_lstm)
        dist_errors_transformer_list.append(dist_error_trasformer)
        ground_truth_distance_list.append(dist_ground_truth)

        cnn_mse_error_list.append(0.5* (current_lat - pred_lat_cnn)**2 + (current_long - pred_long_cnn)**2)
        lstm_mse_error_list.append(0.5* (current_lat - pred_lat_lstm)**2 + (current_long - pred_long_lstm)**2)
        transformer_mse_error_list.append(0.5* (current_lat - pred_lat_transformer)**2 + (current_long - pred_long_transformer)**2)

    
    dist_errors_cnn_list_of_lists.append(dist_errors_cnn_list)
    dist_errors_lstm_list_of_lists.append(dist_errors_lstm_list)
    dist_errors_transformer_list_of_lists.append(dist_errors_transformer_list)
    ground_truth_distance_list_of_lists.append(ground_truth_distance_list)

    cnn_mse_error_list_of_lists.append(cnn_mse_error_list)
    lstm_mse_error_list_of_lists.append(lstm_mse_error_list)
    transformer_mse_error_list_of_lists.append(transformer_mse_error_list)


cnn_error_arr = np.array(dist_errors_cnn_list_of_lists)
lstm_error_arr = np.array(dist_errors_lstm_list_of_lists)
transformer_error_arr = np.array(dist_errors_transformer_list_of_lists)
gt_distance = np.array(ground_truth_distance_list_of_lists)

error_cnn_dist_average = np.mean(cnn_error_arr, axis=0)
error_lstm_dist_average = np.mean(lstm_error_arr, axis=0)
error_transformer_dist_average = np.mean(transformer_error_arr, axis=0)
gt_disance_avg = np.median(gt_distance, axis=0)
plt.figure()
#plt.plot(erro_dist_average)
plt.plot(gt_disance_avg, error_cnn_dist_average, label='cnn error')
plt.plot(gt_disance_avg, error_lstm_dist_average, label='lstm error')
plt.plot(gt_disance_avg, error_transformer_dist_average, label='transformer error')


plt.figure()
#plt.plot(erro_dist_average)
plt.plot(gt_disance_avg, error_cnn_dist_average/gt_disance_avg, label='cnn error')
plt.plot(gt_disance_avg, error_lstm_dist_average/gt_disance_avg, label='lstm error')
plt.plot(gt_disance_avg, error_transformer_dist_average/gt_disance_avg, label='transformer error')

cnn_error_arr = np.array(cnn_mse_error_list_of_lists)
lstm_error_arr = np.array(lstm_mse_error_list_of_lists)
transformer_error_arr = np.array(transformer_mse_error_list_of_lists)
gt_distance = np.array(ground_truth_distance_list_of_lists)

error_cnn_dist_average = np.mean(cnn_error_arr, axis=0)
error_lstm_dist_average = np.mean(lstm_error_arr, axis=0)
error_transformer_dist_average = np.mean(transformer_error_arr, axis=0)
gt_disance_avg = np.median(gt_distance, axis=0)
plt.figure()
#plt.plot(erro_dist_average)
plt.plot(gt_disance_avg, error_cnn_dist_average, label='cnn error')
plt.plot(gt_disance_avg, error_lstm_dist_average, label='lstm error')
plt.plot(gt_disance_avg, error_transformer_dist_average, label='transformer error')

def png_path_to_fig(image_path, main_title=None, num_rows=None, num_cols=None, prop_trim_x=None, prop_trim_y=None, fig_width_scale=1.0, fig_height_scale=1.0):
    # Read the image
    img = mpimg.imread(image_path)
    
    # Get the dimensions of the image
    height, width, _ = img.shape

    if prop_trim_x is not None and prop_trim_y is not None:
        trim_x = int(prop_trim_x * height)
        trim_y = int(prop_trim_y * width)

        img = img[trim_x: -trim_x, trim_y:-trim_y, :]

        height, width, _ = img.shape
    
    # Convert dimensions from pixels to inches (assuming 100 DPI)
    dpi = 100
    fig_width = (width / dpi) * fig_width_scale
    fig_height = (height / dpi) * fig_height_scale
    
    # Check if num_rows and num_cols are specified
    if num_rows is not None and num_cols is not None:
        # Create a figure with the specified number of subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), dpi=dpi)
        
        # Flatten the axs array if it's 2D for easier indexing
        if num_rows > 1 or num_cols > 1:
            axs = axs.flatten()
        
        # Display the image in the first subplot
        axs[0].imshow(img)
        axs[0].axis('off')
        
        # Add a main title if provided
        if main_title is not None:
            fig.suptitle(main_title)
        
        return fig, axs
    else:
        # Create a figure with the same size as the image
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        # Display the image
        ax.imshow(img)
        ax.axis('off')
        
        # Add a main title if provided
        if main_title is not None:
            fig.suptitle(main_title)
        
        return fig, ax


html_map_path, png_map_path = get_map_image(m, file_base = None, save_map_name = str(100), firefox_dir = None, firefox_binary = None)


fig, ax = png_path_to_fig(png_map_path, main_title=None, num_rows=2, num_cols=2, prop_trim_x = 0.09, prop_trim_y = 0.23, fig_width_scale=1.8, fig_height_scale=1.8)
#fig, ax = png_path_to_fig(png_map_path, main_title=None, num_rows=2, num_cols=2, prop_trim_x = 0.3, prop_trim_y = 0.3, fig_width_scale=2.8, fig_height_scale=2.8)


cnn_error_arr = np.array(dist_errors_cnn_list_of_lists)
lstm_error_arr = np.array(dist_errors_lstm_list_of_lists)
transformer_error_arr = np.array(dist_errors_transformer_list_of_lists)
gt_distance = np.array(ground_truth_distance_list_of_lists)

error_cnn_dist_average = np.mean(cnn_error_arr, axis=0)
error_lstm_dist_average = np.mean(lstm_error_arr, axis=0)
error_transformer_dist_average = np.mean(transformer_error_arr, axis=0)
gt_disance_avg = np.median(gt_distance, axis=0)


cnn_error_arr_mse = np.array(cnn_mse_error_list_of_lists)
lstm_error_arr_mse = np.array(lstm_mse_error_list_of_lists)
transformer_error_arr_mse = np.array(transformer_mse_error_list_of_lists)

error_cnn_dist_average_mse = np.mean(cnn_error_arr_mse, axis=0)
error_lstm_dist_average_mse = np.mean(lstm_error_arr_mse, axis=0)
error_transformer_dist_average_mse = np.mean(transformer_error_arr_mse, axis=0)


ax[0].set_title("Predicted Paths")

ax[1].plot(gt_disance_avg, error_cnn_dist_average, label='cnn error')
ax[1].plot(gt_disance_avg, error_lstm_dist_average, label='lstm error')
ax[1].plot(gt_disance_avg, error_transformer_dist_average, label='transformer error')
ax[1].set_title("Track Error")
ax[1].set_xlabel("Distance Flown (Km)")
ax[1].set_ylabel("Predicted Track Error (Km)")
ax[1].legend(loc='best')  # Add legend here

ax[2].plot(gt_disance_avg, error_cnn_dist_average/gt_disance_avg, label='cnn error')
ax[2].plot(gt_disance_avg, error_lstm_dist_average/gt_disance_avg, label='lstm error')
ax[2].plot(gt_disance_avg, error_transformer_dist_average/gt_disance_avg, label='transformer error')
ax[2].set_title("Track Error Normalized By Distance Flown")
ax[2].set_xlabel("Distance Flown (Km)")
ax[2].set_ylabel("Predicted Track Error Normalized ")
ax[2].legend(loc='best')  # Add legend here

ax[3].plot(gt_disance_avg, error_cnn_dist_average_mse, label='cnn error')
ax[3].plot(gt_disance_avg, error_lstm_dist_average_mse, label='lstm error')
ax[3].plot(gt_disance_avg, error_transformer_dist_average_mse, label='transformer error')
ax[3].set_title("Distance Traveled vs Mean Squared Error")
ax[3].set_xlabel("Distance Flown (Km)")
ax[3].set_ylabel("Mean Squared Error")
ax[3].legend(loc='best')  # Add legend here

fig.subplots_adjust(wspace=0.4, hspace=0.4)
# Display the figure
fig

import numpy as np
import matplotlib.pyplot as plt

# Example data
batch_size = 100
timesteps_future = 10

predictions_model1 = np.array(dist_errors_cnn_list_of_lists)
predictions_model2 = np.array(dist_errors_lstm_list_of_lists)
predictions_model3 = np.array(dist_errors_transformer_list_of_lists)

# Create subplots
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
bins = 30  # Adjusted number of bins for better visibility

for i in range(timesteps_future):
    ax = axes[i]
    mean_cnn_error = np.mean(predictions_model1[:, i])
    mean_lstm_error = np.mean(predictions_model2[:, i])
    mean_transformer_error = np.mean(predictions_model3[:, i])
    ax.hist(predictions_model1[:, i], bins=bins, alpha=0.4, label=f'CNN Error | mean error: {mean_cnn_error:.2f}')
    ax.hist(predictions_model2[:, i], bins=bins, alpha=0.4, label=f'LSTM Error | mean error: {mean_lstm_error:.2f}')
    ax.hist(predictions_model3[:, i], bins=bins, alpha=0.4, label=f'Transformer Error | mean error: {mean_transformer_error:.2f}')
    ax.set_title(f'Predicted Timestep {i+1}')
    ax.legend()

plt.tight_layout()
plt.show()


cnn_error_arr.shape, lstm_error_arr.shape, transformer_error_arr.shape, gt_distance.shape


cnn_error_arr = np.array(dist_errors_cnn_list_of_lists)
gt_distance = np.array(ground_truth_distance_list_of_lists)

tuple_key = ('150171', 'SKW5664')

color_cnn = "green"
color_lstm = "blue"
color_transformer = 'orange'


flightpath_compleate = flightpaths[tuple_key][["Latitude", "Longitude"]].to_numpy()
min_lat, max_lat = np.min(flightpath_compleate[:, 0]), np.max(flightpath_compleate[:, 0])
min_long, max_long = np.min(flightpath_compleate[:, 1]), np.max(flightpath_compleate[:, 1])

m = create_folium_map(min_lat, min_long, max_lat, max_long, border_lat_prop=0.15, border_long_prop=0.15, tiles=None) #"Cartodb dark_matter")
folium.PolyLine(locations=flightpath_compleate, color='black', weight=2.5, opacity=1).add_to(m)


num_predictions = len(results_dict[tuple_key]["prediction_cnn"])
for idx_pred in range(91, num_predictions, 10):
    cnn_prediction = results_dict[tuple_key]["prediction_cnn"][idx_pred][0]
    lstm_prediction = results_dict[tuple_key]["prediction_lstm"][idx_pred][0]
    transformer_prediction = results_dict[tuple_key]["prediction_trasformer"][idx_pred][0]

    folium.PolyLine(locations=cnn_prediction.T, color=color_cnn, weight=2.5, opacity=1).add_to(m)
    folium.PolyLine(locations=lstm_prediction.T, color=color_lstm, weight=2.5, opacity=1).add_to(m)
    folium.PolyLine(locations=transformer_prediction.T, color=color_transformer, weight=2.5, opacity=1).add_to(m)



flightpath_compleate = flightpath_compleate_pd[["Latitude", "Longitude"]].to_numpy()

timesteps = flightpath_compleate.shape[0]
flightpath_chunk = flightpath_compleate[int(0.5*timesteps): int(0.7*timesteps), :]


min_lat, max_lat = np.min(flightpath_chunk[:, 0]), np.max(flightpath_chunk[:, 0])
min_long, max_long = np.min(flightpath_chunk[:, 1]), np.max(flightpath_chunk[:, 1])

m = create_folium_map(min_lat, min_long, max_lat, max_long, border_lat_prop=0.15, border_long_prop=0.15, tiles=None) #"Cartodb dark_matter")
folium.PolyLine(locations=flightpath_compleate, color='black', weight=2.5, opacity=1).add_to(m)