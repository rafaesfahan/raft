import numpy as np
from matplotlib import pyplot as plt

import torch

import pytorch_lightning as pl
import folium

from coordinate_transform import *
from net import init_weights
from folium_utils import create_folium_map, get_map_image, png_path_to_fig

from coordinate_transform import CoordinateEnum, helper_get_coordinate_system_based_on_enum


class FlightModel(pl.LightningModule):
    def __init__(self, model, coordinate_system_enum, loss_fn, optimizer, max_num_val_maps = 8, n_future_timesteps = 10, mean=0.0, std=0.0001):
        super().__init__()
        self.model = model
        self.coordinate_system_enum =coordinate_system_enum
        self.coordinate_system = helper_get_coordinate_system_based_on_enum(coordinate_system_enum)
        self.len_coordinate_system = len(self.coordinate_system )
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.max_num_val_maps = max_num_val_maps
        self.n_future_timesteps = n_future_timesteps

        if mean is None or std is None:
            print("Using Default model weights initialization")
        else:
            self.model.apply(lambda m: init_weights(m, mean, std))
            print(f"Initializing model with weights distribution {mean} and standard deviation {std}")


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        batch = normalize_center(batch, coordinate_system_enum=self.coordinate_system_enum)
        
        input_tensor = batch["input_tensor"]
        output_tensor = batch["output_tensor"]
        pred_tensor = self.model(input_tensor)

        loss = self.loss_fn(pred_tensor, output_tensor)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):

        # Model prediction
        input_tensor = batch["input_tensor"]
        input_coordinates_np = input_tensor[:, :self.len_coordinate_system , :].detach().cpu().numpy()

        batch = normalize_center(batch, coordinate_system_enum=self.coordinate_system_enum)
        input_tensor = batch["input_tensor"]
        output_tensor = batch["output_tensor"]
        list_full_flightpaths = batch["meta_flightpath"]
        list_of_msn = batch["meta_msn"]
        pred_tensor = self.model(input_tensor) # torch.Size([32, 4, 1])
        b = input_tensor.shape[0]

        zero_pad_rows_np = batch['zero_pad_rows'].detach().cpu().numpy()
        
        iterative_predictions_tensor_np = iterative_path_predict(batch, self.model, self.coordinate_system_enum, num_predict_steps=self.n_future_timesteps, bool_normalize_center = False)
        

        loss = self.loss_fn(pred_tensor, output_tensor)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=False)

        batch = un_normalize_center(batch, coordinate_system_enum=self.coordinate_system_enum)
        pred_tensor = un_normalize_center_predictions(pred_tensor, batch["normalization_tensor"])


        for idx_batch in range(b):
            if idx_batch > self.max_num_val_maps:
                break

            coors_zero_padded = input_coordinates_np[idx_batch]
            zero_pad = zero_pad_rows_np[idx_batch]
            iterative_predictions_tensor_np_current_sample = iterative_predictions_tensor_np[idx_batch]
            flightpath_compleate = list_full_flightpaths[idx_batch][["Latitude", "Longitude"]].to_numpy()
            current_msn = list_of_msn[idx_batch]

            try:
                m = get_folium_map_per_perdiction(coors_zero_padded, 
                                                    zero_pad, 
                                                    iterative_predictions_tensor_np_current_sample, 
                                                    flightpath_compleate, 
                                                    self.coordinate_system_enum)
                

                html_map_path, png_map_path = get_map_image(m, file_base = None, save_map_name = str(idx_batch), firefox_dir = None, firefox_binary = None)
                fig = png_path_to_fig(png_map_path, main_title=current_msn)
                self.logger.experiment.add_figure(f'Flight {idx_batch}', fig, self.global_step)
                plt.close('all')
            except Exception as e:
                print("Exception in Validation step: ", e)


        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1.0e-4)



def normalize_center(batch_dict, coordinate_system_enum, coef_multiply = 1.0e4):
    """
    Normalize Coordinates for given coordinate system enum
    """


    with torch.no_grad():
        input_tensor = batch_dict["input_tensor"]
        normalization_tensor = batch_dict["normalization_tensor"]

        if coordinate_system_enum == CoordinateEnum.ComplexCoordinates:
            len_coordinates = 4

        elif coordinate_system_enum == CoordinateEnum.LatLongCoordinates:
            len_coordinates = 2

        else:
            raise NotImplementedError(f"Coordinate system: {coordinate_system_enum} Not implemented")
        

        input_tensor[:, :len_coordinates, :] = input_tensor[:, :len_coordinates, :] - normalization_tensor
        batch_dict["input_tensor"] = input_tensor
    
        if "output_tensor" in batch_dict:
            output_tensor = batch_dict["output_tensor"]
            output_tensor = output_tensor - normalization_tensor
            batch_dict["output_tensor"] = output_tensor

    return batch_dict

def un_normalize_center(batch_dict, coordinate_system_enum, coef_multiply = 1.0e-4):
    """
    Un-normalize Coordinates for given coordinate system enum
    """


    with torch.no_grad():
        input_tensor = batch_dict["input_tensor"]
        normalization_tensor = batch_dict["normalization_tensor"]

        if coordinate_system_enum == CoordinateEnum.ComplexCoordinates:
            len_coordinates = 4

        elif coordinate_system_enum == CoordinateEnum.LatLongCoordinates:
            len_coordinates = 2

        else:
            raise NotImplementedError(f"Coordinate system: {coordinate_system_enum} Not implemented")
        

        input_tensor[:, :len_coordinates, :] = input_tensor[:, :len_coordinates, :] + normalization_tensor
        

        batch_dict["input_tensor"] = input_tensor
        if "output_tensor" in batch_dict:
            output_tensor = batch_dict["output_tensor"]
            output_tensor = output_tensor + normalization_tensor
            batch_dict["output_tensor"] = output_tensor

    return batch_dict

def un_normalize_center_predictions(pred, normalization_tensor):
    un_normalized_pred = pred + normalization_tensor
    return un_normalized_pred


# PREDICT


def iterative_path_predict_step(input_tensor, prediction_tensor, coordinate_system_enum, len_coordinate_system):
    with torch.no_grad():
        _, _, num_input_rows = input_tensor.shape
        _, _, num_output_rows = prediction_tensor.shape
        input_tensor_final_n_rows = input_tensor[:, :, -num_output_rows:]  
        stacked_orig_pred = torch.concat([input_tensor, input_tensor_final_n_rows], dim=-1) # keep auxiliary info, pred tensor will then replace the coordinate values
        stacked_orig_pred[:, :len_coordinate_system, -num_output_rows:] = prediction_tensor 
        new_normalization_tensor = stacked_orig_pred[:, :len_coordinate_system, -1:].clone()  # Clone to avoid modifying the original tensor

        stacked_orig_pred[:, :len_coordinate_system, :] = stacked_orig_pred[:, :len_coordinate_system, :] - new_normalization_tensor # new input tensor

    return torch.clone(stacked_orig_pred[:, :, -num_input_rows:]), torch.clone(new_normalization_tensor)



def iterative_path_predict(batch_dict, model, coordinate_system_enum, num_predict_steps, bool_normalize_center = False, bool_with_eval_model = False):
    """
    iteratively predict N steps using the model
    """
    
    coordinate_system = helper_get_coordinate_system_based_on_enum(coordinate_system_enum)
    len_coordinate_system = len(coordinate_system )

    if bool_normalize_center:
        batch_dict = normalize_center(batch_dict, coordinate_system_enum=coordinate_system_enum)

    input_tensor = batch_dict["input_tensor"]
    normalization_tensor_cumulative = torch.clone(batch_dict["normalization_tensor"])


    pred_tensors_list = []

    for i in range(num_predict_steps):
        if bool_with_eval_model:
            model.eval()
            with torch.no_grad():
                prediction_tensor = model(input_tensor)
        else:
            prediction_tensor = model(input_tensor)
        input_tensor, normalization_tensor = iterative_path_predict_step(input_tensor.clone(), prediction_tensor.clone(), coordinate_system_enum, len_coordinate_system)
        #######normalization_tensor_cumulative = normalization_tensor_cumulative + normalization_tensor.clone()
        pred_tensors_list.append(prediction_tensor.clone())

    
    iterative_predictions_tensor = torch.concat(pred_tensors_list, dim=-1) 
    iterative_predictions_tensor = torch.cumsum(iterative_predictions_tensor, dim=-1) 
    iterative_predictions_tensor = un_normalize_center_predictions(iterative_predictions_tensor, normalization_tensor_cumulative) 
    iterative_predictions_tensor_np = iterative_predictions_tensor.detach().cpu().numpy()
    return iterative_predictions_tensor_np





def helper_complex_coors_to_lat_long(complex_coors_zero_padded, zero_pad = None):
    if zero_pad is None:
        complex_coors = complex_coors_zero_padded[:4, : ]
    else:
        complex_coors = complex_coors_zero_padded[:4, zero_pad: ]

    complex_lat_x = complex_coors[0, :]
    complex_lat_y = complex_coors[1, :]
    complex_long_x = complex_coors[2, :]
    complex_long_y = complex_coors[3, :]

    degrees_lat = complex_number_to_degrees(complex_lat_x, complex_lat_y)
    degrees_long = complex_number_to_degrees(complex_long_x, complex_long_y)
    return degrees_lat, degrees_long



def get_folium_map_per_perdiction(coors_zero_padded, zero_pad, iterative_predictions_tensor_np_current_sample, flightpath_compleate, coordinate_system_enum):
    """
    Assumes no Batch dim for coors_zero_padded
    """
    coordinate_system = helper_get_coordinate_system_based_on_enum(coordinate_system_enum)
    len_coordinate_system = len(coordinate_system )

    if coordinate_system_enum == CoordinateEnum.ComplexCoordinates:
        degrees_lat, degrees_long = helper_complex_coors_to_lat_long(coors_zero_padded, zero_pad)
        degrees_lat_iterative_pred, degrees_long_iterative_pred = helper_complex_coors_to_lat_long(iterative_predictions_tensor_np_current_sample, zero_pad)
        flightpath_chunk_given_to_model = np.stack([degrees_lat, degrees_long], axis=1)
        flightpath_iterative_pred = np.stack([degrees_lat_iterative_pred, degrees_long_iterative_pred], axis=1)
    elif coordinate_system_enum == CoordinateEnum.LatLongCoordinates:
        flightpath_chunk_given_to_model = coors_zero_padded[:2, zero_pad: ]
        flightpath_iterative_pred = iterative_predictions_tensor_np_current_sample[:2, : ]
    else:
        raise NotImplementedError(f"Coordinate System {coordinate_system_enum} not implemented")


    min_lat, max_lat = np.min(flightpath_compleate[:, 0]), np.max(flightpath_compleate[:, 0])
    min_long, max_long = np.min(flightpath_compleate[:, 1]), np.max(flightpath_compleate[:, 1])


    print("debug flightpath_chunk_given_to_model.shape: ", flightpath_chunk_given_to_model.shape)

    m = create_folium_map(min_lat, min_long, max_lat, max_long, border_lat_prop=0.15, border_long_prop=0.15, tiles=None)
    folium.PolyLine(locations=flightpath_chunk_given_to_model.T, color='blue', weight=3.4, opacity=1).add_to(m)
    folium.PolyLine(locations=flightpath_compleate, color='black', weight=1.5, opacity=1).add_to(m)
    folium.PolyLine(locations=flightpath_iterative_pred.T, color='red', weight=3.5, opacity=1).add_to(m)
    return m