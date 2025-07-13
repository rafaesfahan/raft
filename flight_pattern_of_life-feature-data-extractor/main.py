
import torch #
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
from pytorch_lightning import Trainer
import numpy as np
import os

from net import SimpleNet
from net_lstm import SimpleLSTM
from net_transformer import SimpleTimeSeriesTransformer

from datamodule import Datamodule
from model import FlightModel
from iterate_flights import helper_get_min_max_quantiles_from_dataset

from file_parsing_utils import create_csv_dict
from coordinate_transform import CoordinateEnum, helper_get_coordinate_system_based_on_enum

from enum import Enum



import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
from hydra.utils import instantiate
from hydra import initialize, compose


import hydra
from omegaconf import DictConfig

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf import OmegaConf

from resolvers import register_resolvers
from eval_metrics import eval_models, get_eval_keys
import mlflow
from mlflow.tracking import MlflowClient


class ModelEnum(Enum):
    SimpleCNN = 0
    SimpleLSTM = 1
    SimpleTransformer = 2 
    SimpleMamba = 3

    def __str__(self):
        return self.name
    




# class LoggingCallback(pl.Callback):
#     def on_validation_end(self, trainer, pl_module):
#         val_loss = trainer.callback_metrics.get("val_loss")
#         if val_loss is not None:
#             mlflow.log_metric("val_loss", val_loss.item())
        
#         # Assuming `pl_module` is your model
#         # Log the model
#         mlflow.pytorch.log_model(pl_module, "model")

#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
#         train_loss = trainer.callback_metrics.get("train_loss")
#         if train_loss is not None:
#             mlflow.log_metric("train_loss", train_loss.item())



# def register_model(model_name):
#     client = MlflowClient()
    
#     # Get the latest run ID
#     run_id = mlflow.active_run().info.run_id

#     # Register the model
#     result = client.create_registered_model(model_name)

#     # Create a new version of the registered model
#     model_uri = f"runs:/{run_id}/model"
#     model_version = client.create_model_version(model_name, model_uri, run_id)
    
#     print(f"Model registered with name: {model_name}, version: {model_version.version}")



import mlflow
from mlflow.tracking import MlflowClient


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            mlflow.log_metric("val_loss", val_loss.item())

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = None):
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            mlflow.log_metric("train_loss", train_loss.item())




def register_model(model_name):
    client = MlflowClient()
    
    # Get the latest run ID
    run_id = mlflow.active_run().info.run_id

    # Register the model
    result = client.create_registered_model(model_name)

    # Create a new version of the registered model
    model_uri = f"runs:/{run_id}/model"
    model_version = client.create_model_version(model_name, model_uri, run_id)
    
    print(f"Model registered with name: {model_name}, version: {model_version.version}")

# def main(cfg):
#     # Initialize the trainer, model, and dataloaders
#     trainer = pl.Trainer(callbacks=[LoggingCallback()])

#     # Train the model
#     trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=ckpt_path)
    
#     # Log and register the model after training
#     run_id = mlflow.active_run().info.run_id

#     # Log the model
#     mlflow.pytorch.log_model(model, "model")

#     # Register the model
#     register_model(model_name="my_model_name")




















# mlflow server --host 127.0.0.1 --port 5000
# @hydra.main(version_base=None, config_path="conf", config_name="config")
# def main(cfg: DictConfig):
#     # Hydra Stuff
#     register_resolvers()
#     cfg = OmegaConf.to_container(cfg, resolve=True)
#     cfg = OmegaConf.create(cfg)

#     # Create model_save_dir where the (numbered) experiment will live, update Hydra checkpoint config to reflect this
#     model_save_dir = cfg['model_dir']
#     os.makedirs(model_save_dir, exist_ok=True)

#     # Save the config that was used to generate this run
#     config_save_dir = os.path.join(model_save_dir, "config")
#     os.makedirs(config_save_dir, exist_ok=True)
#     with open(os.path.join(config_save_dir, "config.yaml"), "w") as f:
#         OmegaConf.save(cfg, f)

#     # Set the experiment name for MLflow
#     model_name_long = str(cfg['model_dir'].replace("/", "_"))
#     mlflow.set_tracking_uri(cfg['mlflow']['mlflow']['set_tracking_uri'])  # mlflow server --host 127.0.0.1 --port 5000
#     mlflow.set_experiment(model_name_long)

#     # Rest of your code...






















# HYDRA_FULL_ERROR=1 python3 main.py -m experiment=mil_simple_cnn,mil_simple_lstm,mil_simple_transformer
# HYDRA_FULL_ERROR=1 python3 main.py -m experiment=local_simple_cnn,local_simple_lstm,local_simple_transformer
# mlflow server --host 127.0.0.1 --port 5000
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Hydra Stuff
    register_resolvers()
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)
    

    # Create model_save_dir where the (numbered) experiment will live, update hydar checkpoint config to reflect this
    model_save_dir = cfg.model_dir
    os.makedirs(model_save_dir, exist_ok=True)
    print(cfg['callbacks']['callbacks'])

    # Save the config that was used to generate this run
    config_save_dir = model_save_dir + "/config/"
    os.makedirs(config_save_dir, exist_ok=True)
    print(OmegaConf.to_yaml(cfg))
    with open(config_save_dir + "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    
    model_name_long = str(cfg.model_dir.replace("/", "_"))
    
    # Set Ml Flow tracking
    mlflow.set_tracking_uri(cfg.mlflow.mlflow.set_tracking_uri)   # mlflow server --host 127.0.0.1 --port 5000   
    ###mlflow.set_experiment(str(cfg.experiment_name))
    mlflow.set_experiment(model_name_long)




    # Get flightpath keys we will be using to evaluate the model / make sure those keys are not trained on
    if cfg.eval_config.eval_config.desired_keys is None:
        eval_keys = get_eval_keys(cfg.datamodule.datamodule.all_flight_dataframes_dict)
    else:
        eval_keys = cfg.eval_config.eval_config.desired_keys
    print("Eval keys: ", eval_keys)

    datamodule = instantiate(cfg.datamodule.datamodule)
    datamodule.guarantee_eval_keys_in_test_set(eval_keys = eval_keys) # Make sure model is not tained on the same flightpaths we will use to eval


    # Normalization values for dataset-wide normalization (min/max of both lat/long values)
    dataset_wide_normalization_dict = None
    if cfg.bool_normalize_dataset_wide:
        quantiles = [cfg.quantile_normalization_min, cfg.quantile_normalization_max]
        latitude_quantiles, longitude_quantiles = helper_get_min_max_quantiles_from_dataset(datamodule.train_dict, quantiles, limit_samples = cfg.limit_samples)

        lat_min = latitude_quantiles[cfg.quantile_normalization_min]
        lat_max = latitude_quantiles[cfg.quantile_normalization_max]
        long_min = longitude_quantiles[cfg.quantile_normalization_min]
        long_max = longitude_quantiles[cfg.quantile_normalization_max]

        dataset_wide_normalization_dict = {
            "lat_min": lat_min, 
            "lat_max": lat_max,
            "long_min": long_min,
            "long_max": long_max,
        }

        datamodule.train_loader.dataset_wide_normalization_dict = dataset_wide_normalization_dict
        datamodule.test_loader.dataset_wide_normalization_dict = dataset_wide_normalization_dict


    # Train the model / pass dummy tensor to instantiate things (necessary step for torch.jit I believe among other things)
    train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.test_dataloader()
    model = instantiate(cfg.model.model)
    model.dataset_wide_normalization_dict = dataset_wide_normalization_dict # to give validation step access to the un-normalizing code 
    test_dataloader_iterator = test_dataloader.__iter__()
    tensors_dict = next(test_dataloader_iterator)
    dummy_input_tensor = tensors_dict["input_tensor"]
    dummy_output = model(dummy_input_tensor)
    trainer = instantiate(cfg.trainer.trainer)
    trainer.callbacks.append(LoggingCallback())
    ckpt_path = None
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader, ckpt_path=ckpt_path)


    # Log and register the model after training
    run_id = mlflow.active_run().info.run_id
    # Log the model
    mlflow.pytorch.log_model(model, "model")
    # Register the model
    try:
        register_model(model_name=model_name_long)        # TODO LONG-TERM SOLUTINON IS TO CHECK IF THIS NAME EXISTS AND DO V2, V2 AND SO ON DEPENDING ON WHICH NAMES ARE ALREADY IN USE
    except Exception as e:
        print("Ml Flow register model not working: ", e)


    # Save Config with added number of training samples parameter
    cfg["num_train_limited"] =  datamodule.idx_train_limited          # Number of training flights used in this run
    with open(config_save_dir + "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)



    # Jit the model
    print("torch.jit.trace")
    scripted_model = torch.jit.trace(model, dummy_input_tensor)

    # Save the scripted model
    torch_jit_save_dir = model_save_dir + "model_jit/"
    os.makedirs(torch_jit_save_dir, exist_ok=True)
    torch.jit.save(scripted_model, torch_jit_save_dir + "scripted_model.pt")
    print("torch jit saved !")


    # Log the jit model to MLflow
    try:
        mlflow.pytorch.log_model(scripted_model, "scripted_model_jit")
        print("Jit model logged to MLflow!")
    except Exception as e:
        print("Failed to log JIT model to mlflow !", e)



    # Run eval over desired keys specified earlier
    print("Running Eval")
    # try:
    results_dir = model_save_dir + "eval_results/"
    print("results dir: ", results_dir)
    eval_models(model, 
                individual_flights_dir=cfg.datamodule.datamodule.all_flight_dataframes_dict,
                coordinate_system_enum=cfg.coordinate_system.coordinate_system.coordinate_system_enum,
                auxiliary_input_channels=cfg.coordinate_system.coordinate_system.auxiliary_input_channels,
                auxiliary_output_channels=cfg.coordinate_system.coordinate_system.auxiliary_output_channels,
                min_rows_input=cfg.datamodule.datamodule.min_rows_input,
                num_input_rows_total=cfg.datamodule.datamodule.num_input_rows_total,
                results_save_dir=results_dir,
                desired_keys=eval_keys,
                num_predict_steps=cfg.eval_config.eval_config.num_predict_steps,
                break_after_index=cfg.eval_config.eval_config.break_after_index, 
                dataset_wide_normalization_dict=dataset_wide_normalization_dict, 
            )
    
    # except Exception as e:
    #     print("SOMETHING BROKE IN EVAL: ", e)

    print("Finished! \n")

if __name__ == "__main__": #
    main()


