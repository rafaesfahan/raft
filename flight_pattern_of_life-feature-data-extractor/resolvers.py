import hydra
from omegaconf import DictConfig
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf import OmegaConf


# Define your resolver function
def calculate_channels(coordinate_system_enum, auxiliary_input_channels, auxiliary_output_channels):
    from coordinate_transform import helper_get_coordinate_system_based_on_enum, CoordinateEnum

    coordinate_system = helper_get_coordinate_system_based_on_enum(
        coordinate_enum=CoordinateEnum[coordinate_system_enum]
    )
    in_channels = len(coordinate_system) + len(auxiliary_input_channels)
    out_channels = len(coordinate_system) + len(auxiliary_output_channels)
    return in_channels, out_channels


def calculate_in_channels(coordinate_system_enum, auxiliary_input_channels, auxiliary_output_channels):
    in_channels, _ = calculate_channels(coordinate_system_enum, auxiliary_input_channels, auxiliary_output_channels)
    return in_channels

def calculate_out_channels(coordinate_system_enum, auxiliary_input_channels, auxiliary_output_channels):
    _, out_channels = calculate_channels(coordinate_system_enum, auxiliary_input_channels, auxiliary_output_channels)
    return out_channels


def get_experiment_name_base_on_existing_experiments(dir_model):  # dir_model: cfg.model_dir
    """
    Resolver to create directory for this specific experiment, give back experiment name based on the number of experiments already present in dir_model
    """
    os.makedirs(dir_model, exist_ok=True)
    num_experiments_present_already = len(os.listdir(dir_model))
    experiment_name = f"Experiment_{num_experiments_present_already}/"
    return experiment_name




# def register_resolvers():
#     OmegaConf.register_new_resolver("calculate_in_channels", calculate_in_channels)
#     OmegaConf.register_new_resolver("calculate_out_channels", calculate_out_channels)
#     OmegaConf.register_new_resolver("get_experiment_name_base_on_existing_experiments", get_experiment_name_base_on_existing_experiments)




# def is_resolver_registered(name):
#     try:
#         # Try to use the resolver; if it doesn't exist, it will raise a KeyError
#         OmegaConf.resolve({"key": f"${{{name}:dummy}}"})  # dummy value to test
#         return True
#     except KeyError:
#         return False

# def register_resolvers():
#     if not is_resolver_registered("calculate_in_channels"):
#         OmegaConf.register_new_resolver("calculate_in_channels", calculate_in_channels)
#     if not is_resolver_registered("calculate_out_channels"):
#         OmegaConf.register_new_resolver("calculate_out_channels", calculate_out_channels)
#     if not is_resolver_registered("get_experiment_name_base_on_existing_experiments"):
#         OmegaConf.register_new_resolver("get_experiment_name_base_on_existing_experiments", get_experiment_name_base_on_existing_experiments)



def is_resolver_registered(name):
    try:
        # Use a dummy resolver key to check for existence
        OmegaConf.register_new_resolver(name, lambda x: x)  # Temporary dummy resolver
        OmegaConf.clear_resolver(name)  # Clean up if registration was successful
        return False
    except ValueError:
        return True

def register_resolvers():
    if not is_resolver_registered("calculate_in_channels"):
        OmegaConf.register_new_resolver("calculate_in_channels", calculate_in_channels)
    if not is_resolver_registered("calculate_out_channels"):
        OmegaConf.register_new_resolver("calculate_out_channels", calculate_out_channels)
    if not is_resolver_registered("get_experiment_name_base_on_existing_experiments"):
        OmegaConf.register_new_resolver("get_experiment_name_base_on_existing_experiments", get_experiment_name_base_on_existing_experiments)





