import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
from pytorch_lightning import Trainer
import numpy as np
import pandas as pd

from net import SimpleNet
from net_lstm import SimpleLSTM
from net_transformer import SimpleTimeSeriesTransformer

from datamodule import Datamodule
from model import FlightModel

from file_parsing_utils import create_csv_dict
from coordinate_transform import CoordinateEnum, helper_get_coordinate_system_based_on_enum
from iterate_flights import itterate_flights, build_features, flight_tensor_chunk_itterator
from model import iterative_path_predict
from coordinate_transform import  haversine
from iterate_flights import flightpath_iterator

from glob import glob 
import copy
from collections import defaultdict

from folium_utils import create_folium_map
from folium_utils import get_map_image
import folium

from matplotlib import pyplot as plt
import matplotlib.image as mpimg


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from enum import Enum
import inspect
import os




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

    coordinate_system_enum = CoordinateEnum.convert_to_str(coordinate_dictionary["CoordinateEnum"] ) #coordinate_dictionary["CoordinateEnum"]     #CoordinateEnum.LatLongCoordinates
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

    
###########################



def eval_over_flightpaths(some_model_list, flightpath_iter, num_predict_steps, break_after_index = None, models_names_list = None, coordinate_system_enum=None):
    """
    
    """

    if not isinstance(some_model_list, list):
        some_model_list = [some_model_list]

    if coordinate_system_enum is None:
        coordinate_system_enum = CoordinateEnum.LatLongCoordinates

    results_dict = defaultdict(lambda: defaultdict(list))
    flightpaths = {}

    for idx_iter, d in enumerate(flightpath_iter):
        print(f"Working on flightpath index number: {idx_iter}, and flightpath is None: {d is None}")
        if d is None: # Break if no more flightpathts
            print("broken because dict is None")
            break

        if break_after_index is not None:  # break if we want to break early 
            if idx_iter > break_after_index:
                print("broken because flight chunk index  >= break after index")
                break

        msn = d['meta_msn']
        flight_id = d['meta_flight_id']
        # Add the Full path of the flight as reference 
        if (msn, flight_id) not in flightpaths:
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

    return results_dict, flightpaths






def get_final_eval_metrics_all_models(eval_dict, flightpaths_dict):
    """
    *Currently assumes Lat-Long coordinates

    # TODO: CURRENTLY CLIPPING THE MODEL OUTPUT TO MATCH THE GROUND TRUTH ARRAY 
    """


    paths_we_evaled_keys_list = list(eval_dict.keys())
    models_and_other = list(eval_dict[paths_we_evaled_keys_list[0]].keys()) # Ex: dict_keys(['prediction_model_0', 'chunk_index', 'ground_truth'])
    all_model_keys = [model_name for model_name in models_and_other if "prediction_model_" in model_name]

    dict_overall_eval = {}
    dict_overall_eval_arrays = {}
    for flightpath_key in paths_we_evaled_keys_list:

        flightpath_dataframe = flightpaths_dict[flightpath_key]
        flightpath_dataframe.columns = [col.capitalize() if col.lower() in ['latitude', 'longitude'] else col for col in flightpath_dataframe.columns]
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
                ground_truth_array_shape = ground_truth_array.shape
                model_prediction_array = eval_dict[flightpath_key][model_key][sample_index]
                ground_truth_array = np.squeeze(ground_truth_array)
                model_prediction_array = np.squeeze(model_prediction_array)[:, :ground_truth_array_shape[-1]] # TODO FORCE THE SAME NUMBER OF PREDICICTIONS AS GROUND TRUTH ARRAY

                # Sometimes there are no more samples in the ground truth array, we must then ignore the corresponding predictions
                if ground_truth_array.shape != model_prediction_array.shape:
                    print("CONTINUE")
                    print(ground_truth_array.shape, model_prediction_array.shape, "\n")
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


# def default_float():
#     return float

# def default_array():
#     return np.array([])

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
    return fig


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
        for model_prediction in model_predictions_list:
            model_prediction = np.squeeze(model_prediction) # just in case we have [1 x ...] dimensionality
            folium.PolyLine(locations=model_prediction.T, color=model_predictions_color, weight=1.5, opacity=1).add_to(m)

    return m, corresponding_color_list



import pickle
import os

import pickle
import os

def convert_defaultdict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict(v) for k, v in d.items()}
    return d

def save_as_pickle(obj, filename, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, filename)
    
    converted_obj = convert_defaultdict(obj)
    with open(file_path, 'wb') as f:
        pickle.dump(converted_obj, f)
    print(f"Object saved as {file_path}")



def load_pickle(filename, results_dir):
    file_path = os.path.join(results_dir, filename)
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    print(f"Object loaded from {file_path}")
    return obj



def get_eval_keys(dict_path):
    """
    Sillyness
    We have a bunch of datasets in different places (local   |   Kurrent   |   Kurrent in the .mil domain)
    Also multiple datasts in each of these places !! 

    Kinda painfull to manually specify which eval files to use every time for every dataset, 
    I'll just pick a few eval files manually for each of these datasets

    This function will give back the desired files if they exist in the current dictionary depending where the source folder of the dict was
    """

    eval_files = None
    # DC Flightpaths on .mil Kurrent    |    Example /home/jovyan/Dataset/DC/Flightpath/F9F1A577B1E24BC996687AD93EEF8727.parquet
    if "DC/Flightpath" in dict_path:
        eval_files = [
            ("C024948A6B8E4BC9B0871F38D32A9090", "filler_key"),
            # ('D288A381A0ED470C9F0981E8593DC275', "filler_key"), 
            # ('F9F1A577B1E24BC996687AD93EEF8727', "filler_key"), 
            # ('B7163018020E4A06B41C25FAABB3FC75', "filler_key"), 
            # ('3B97AC2B8DCE4909850CD667632AC2FA', "filler_key")
        ]

    elif "ADIZ" in dict_path:
        eval_files = [
            ('6D721D55D8304668BD94A1EB1A5199EE', "filler_key"), 
            ('9D7C3D125C6446D9BA7F812BC24CCADB', "filler_key"), 
            ('812493FD85E346E09A6CC507DA816755', "filler_key"), 
            ('8E562696B40546F6AD54A29225AA8F27', "filler_key")
        ]

    # Local Flightpaths   |    /Users/aleksandranikevich/Desktop/AircraftTrajectory/data/Individual_Flights/178224/N90K/178224_N90K.csv
    elif "Users/aleksandranikevich/Desktop/AircraftTrajectory/data/Individual_Flights" in dict_path:
        eval_files = [
            ('178224', 'N90K'), 
            # ('84444', 'NKS1441'), 
            # ('124758', 'AAL2060'), 
            # ('60307', 'JBU1145'), 
        ]

    else:
        raise NotImplementedError("None of the keys found, impossible to tell which eval files to use for eval")
    
    return eval_files






def eval_models(model, 
                individual_flights_dir, 
                coordinate_system_enum, 
                auxiliary_input_channels, 
                auxiliary_output_channels, 
                min_rows_input, 
                num_input_rows_total, 
                results_save_dir, 
                desired_keys = None, 
                num_predict_steps = 10, 
                break_after_index = 3, 
                dataset_wide_normalization_dict = None, 
                ):

    # Eval over only desired flightpath keys
    flight_dfs = create_csv_dict(individual_flights_dir)
    if desired_keys is None:
        desired_keys = ['146014', '180338']
        eval_flights_dfs = {key: flight_dfs[key] for key in desired_keys if key in flight_dfs}
    else:
        eval_flights_dfs = {key[0]: {key[1]: flight_dfs[key[0]][key[1]]} for key in desired_keys}

    flight_dictionary_pre_loaded = False

    coordinate_system = helper_get_coordinate_system_based_on_enum(coordinate_system_enum)
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
                            force_new_flightpath_every_val_step = False, 
                            dataset_wide_normalization_dict = dataset_wide_normalization_dict,
                            )
    
    # Evaluate the model on each of the desired flightpahts, N steps at a time (feeding the model M prior steps aircraft took)
    eval_dict, flightpaths = eval_over_flightpaths(model, flightpath_iter, num_predict_steps, break_after_index = break_after_index)
    # Get metrics 
    dict_overall_eval, dict_overall_eval_arrays = get_final_eval_metrics_all_models(eval_dict, flightpaths_dict=flightpaths)
    # Get averaged metrics for overall performance measure
    final_metrics_dict, final_array_metrics_dict = get_overall_error_metrics(dict_overall_eval, dict_overall_eval_arrays)
    # create figure of the average errors based on the model by different metric
    fig = plot_metrics_from_dict(final_array_metrics_dict)


    # save the figure
    os.makedirs(results_save_dir, exist_ok=True)
    save_path = os.path.join(results_save_dir, "metrics_plot.png")
    fig.savefig(save_path, format='png') 

    # Save the eval metrics (both arrays and final scores)
    save_as_pickle(final_metrics_dict, "final_metrics_dict.pkl", results_save_dir)
    save_as_pickle(final_array_metrics_dict, "final_array_metrics_dict.pkl", results_save_dir)










# broken because dict is None
# Error executing job with overrides: ['experiment=experiment_local_1']
# Traceback (most recent call last):
#   File "/Users/aleksandranikevich/Desktop/AircraftTrajectory/REPO/flight_pattern_of_life/main.py", line 287, in <module>
#     main()
#   File "/opt/anaconda3/envs/trajectory/lib/python3.12/site-packages/hydra/main.py", line 94, in decorated_main
#     _run_hydra(
#   File "/opt/anaconda3/envs/trajectory/lib/python3.12/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
#     _run_app(
#   File "/opt/anaconda3/envs/trajectory/lib/python3.12/site-packages/hydra/_internal/utils.py", line 457, in _run_app
#     run_and_report(
#   File "/opt/anaconda3/envs/trajectory/lib/python3.12/site-packages/hydra/_internal/utils.py", line 223, in run_and_report
#     raise ex
#   File "/opt/anaconda3/envs/trajectory/lib/python3.12/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
#     return func()
#            ^^^^^^
#   File "/opt/anaconda3/envs/trajectory/lib/python3.12/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
#     lambda: hydra.run(
#             ^^^^^^^^^^
#   File "/opt/anaconda3/envs/trajectory/lib/python3.12/site-packages/hydra/_internal/hydra.py", line 132, in run
#     _ = ret.return_value
#         ^^^^^^^^^^^^^^^^
#   File "/opt/anaconda3/envs/trajectory/lib/python3.12/site-packages/hydra/core/utils.py", line 260, in return_value
#     raise self._return_value
#   File "/opt/anaconda3/envs/trajectory/lib/python3.12/site-packages/hydra/core/utils.py", line 186, in run_job
#     ret.return_value = task_function(task_cfg)
#                        ^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/aleksandranikevich/Desktop/AircraftTrajectory/REPO/flight_pattern_of_life/main.py", line 272, in main
#     eval_models(model, 
#     ^^^^^^^^^^^^^^^^^^
#   File "/Users/aleksandranikevich/Desktop/AircraftTrajectory/REPO/flight_pattern_of_life/eval_metrics.py", line 547, in eval_models
#     final_metrics_dict, final_array_metrics_dict = get_overall_error_metrics(dict_overall_eval, dict_overall_eval_arrays)
#                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/aleksandranikevich/Desktop/AircraftTrajectory/REPO/flight_pattern_of_life/eval_metrics.py", line 324, in get_overall_error_metrics
#     num_future_steps = dict_overall_eval_arrays[path_keys[0]][model_keys[0]][eval_metric_keys[0]][0].shape[0] # shape of the first array stored in this arrays nested dict
#                                                                              ~~~~~~~~~~~~~~~~^^^
# IndexError: list index out of range