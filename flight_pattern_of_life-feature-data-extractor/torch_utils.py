import pandas as pd
import numpy as np
import torch



def normalize_tensor(tensor):
    min_val = tensor.min(dim=-1, keepdim=True)[0]
    max_val = tensor.max(dim=-1, keepdim=True)[0]
    normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-8)  # Adding a small epsilon to avoid division by zero
    return normalized_tensor, min_val, max_val

def unnormalize_tensor(tensor, min_val, max_val):
    unnormalized_tensor = tensor * (max_val - min_val + 1e-8) + min_val
    return unnormalized_tensor








def min_max_normalize(data, min_max_dict):
    """
    Normalize a flight dataframe by the min/max values specified in dict
    Args: 
        data: [Pandas Dataframe, np array , pytorch tensor], data containing flightpath information (lat/long coordinates)
        min_max_dict: Dict[float], values used to do min-max un-normalization
    Returns: 
        Normalized data in the same dataformat (Dataframe / numpy array / tensor) as was given
    """

    if isinstance(data, pd.DataFrame):
        # Normalize DataFrame
        data['Latitude'] = (data['Latitude'] - min_max_dict['lat_min']) / (min_max_dict['lat_max'] - min_max_dict['lat_min'])
        data['Longitude'] = (data['Longitude'] - min_max_dict['long_min']) / (min_max_dict['long_max'] - min_max_dict['long_min'])
    elif isinstance(data, np.ndarray):
        # Normalize numpy array
        data[:, 0, :] = (data[:, 0, :] - min_max_dict['lat_min']) / (min_max_dict['lat_max'] - min_max_dict['lat_min'])
        data[:, 1, :] = (data[:, 1, :] - min_max_dict['long_min']) / (min_max_dict['long_max'] - min_max_dict['long_min'])
    elif torch.is_tensor(data):
        # Normalize PyTorch tensor
        data[:, 0, :] = (data[:, 0, :] - min_max_dict['lat_min']) / (min_max_dict['lat_max'] - min_max_dict['lat_min'])
        data[:, 1, :] = (data[:, 1, :] - min_max_dict['long_min']) / (min_max_dict['long_max'] - min_max_dict['long_min'])
    else:
        raise TypeError("Input data must be a DataFrame, numpy array, or PyTorch tensor.")
    

    
    return data


def min_max_unnormalize(data, min_max_dict):
    """
    Un-Normalize a flight dataframe by the min/max values specified in dict
    Args: 
        data: [Pandas Dataframe, np array , pytorch tensor], data containing flightpath information (lat/long coordinates)
        min_max_dict: Dict[float], values used to do min-max un-normalization
    Returns: 
        Un-Normalized data in the same dataformat (Dataframe / numpy array / tensor) as was given
    """
    if isinstance(data, pd.DataFrame):
        # Un-normalize DataFrame
        data['Latitude'] = data['Latitude'] * (min_max_dict['lat_max'] - min_max_dict['lat_min']) + min_max_dict['lat_min']
        data['Longitude'] = data['Longitude'] * (min_max_dict['long_max'] - min_max_dict['long_min']) + min_max_dict['long_min']
    elif isinstance(data, np.ndarray):
        # Un-normalize numpy array
        data[:, 0, :] = data[:, 0, :] * (min_max_dict['lat_max'] - min_max_dict['lat_min']) + min_max_dict['lat_min']
        data[:, 1, :] = data[:, 1, :] * (min_max_dict['long_max'] - min_max_dict['long_min']) + min_max_dict['long_min']
    elif torch.is_tensor(data):
        # Un-normalize PyTorch tensor
        data[:, 0, :] = data[:, 0, :] * (min_max_dict['lat_max'] - min_max_dict['lat_min']) + min_max_dict['lat_min']
        data[:, 1, :] = data[:, 1, :] * (min_max_dict['long_max'] - min_max_dict['long_min']) + min_max_dict['long_min']
    else:
        raise TypeError("Input data must be a DataFrame, numpy array, or PyTorch tensor.")
    
    return data
