from flight_dataset import FlightSeriesDataset
import random
import collections

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from file_parsing_utils import create_csv_dict
from coordinate_transform import CoordinateEnum


class Datamodule(pl.LightningDataModule):
    def __init__(self, 
                 all_flight_dataframes_dict, 
                 num_input_rows_total = 1000, 
                 min_rows_input = 10, 
                 num_output_rows = 1, 
                 coordinate_system_enum = None,
                 auxiliary_input_channels = None,
                 auxiliary_output_channels = None,
                 train_prop = 0.8, 
                 batch_size = 4, 
                 num_workers = -1, 
                 pin_memory = True,
                 break_after = None, 
                 limit_dataset = None, ):
        
        """
        Create the train and test dataloaders
        """

        self.num_input_rows_total = num_input_rows_total
        self.min_rows_input = min_rows_input
        self.num_output_rows = num_output_rows
        self.coordinate_system_enum = CoordinateEnum.convert(coordinate_system_enum)
        self.auxiliary_input_channels = auxiliary_input_channels
        self.auxiliary_output_channels = auxiliary_output_channels
        self.train_prop = train_prop 
        self.batch_size = batch_size
        self.num_workers = num_workers     
        self.pin_memory = pin_memory

        self.break_after = break_after
        self.limit_dataset = limit_dataset

        if isinstance(all_flight_dataframes_dict, str):
            all_flight_dataframes_dict = create_csv_dict(all_flight_dataframes_dict, break_after=self.break_after)

        
        self.all_mission_keys = list(all_flight_dataframes_dict.keys())
        random.shuffle(self.all_mission_keys)
        idx_train = int(train_prop * len(self.all_mission_keys))
        self.idx_train_limited = idx_train if self.limit_dataset is None else int(self.limit_dataset * idx_train)
        self.train_dict = {key: all_flight_dataframes_dict[key] for key in self.all_mission_keys[:self.idx_train_limited]}
        self.test_dict = {key: all_flight_dataframes_dict[key] for key in self.all_mission_keys[idx_train:]}


        self.train_loader = FlightSeriesDataset(flights_dict = self.train_dict, 
                                                 num_input_rows_total = self.num_input_rows_total, 
                                                 min_rows_input = self.min_rows_input, 
                                                 num_output_rows = self.num_output_rows, 
                                                 coordinate_system_enum = self.coordinate_system_enum,
                                                 auxiliary_input_channels = self.auxiliary_input_channels,
                                                 auxiliary_output_channels = self.auxiliary_output_channels,
                                                 bool_yield_meta_flightpath= False)
        
        self.test_loader = FlightSeriesDataset(flights_dict = self.test_dict, 
                                                 num_input_rows_total = self.num_input_rows_total, 
                                                 min_rows_input = self.min_rows_input, 
                                                 num_output_rows = self.num_output_rows, 
                                                 coordinate_system_enum = self.coordinate_system_enum,
                                                 auxiliary_input_channels = self.auxiliary_input_channels,
                                                 auxiliary_output_channels = self.auxiliary_output_channels,
                                                 force_new_flightpath_every_val_step = True, 
                                                 bool_yield_meta_flightpath = True)

    # DataLoader(dataset, batch_size=2, collate_fn=custom_collate)

    def guarantee_eval_keys_in_test_set(self, eval_keys = None):
        """
        Default instantiation of train and test flight datasets will asign flights randomly, 
        we want to guarantee the flights we will use to eval are not included in the train set so model cannot memorize them 
        """
        if eval_keys is None:
            return 
        
        for outer_key, inner_key in eval_keys:
            if outer_key in self.train_dict:
                if inner_key in self.train_dict[outer_key]:
                    val = self.train_dict[outer_key][inner_key]
                    #self.test_dict[outer_key][inner_key] = val          ### TODO TODO OUTER KEY DOES NOT EXIST SOMETIMES ! 
                    del(self.train_dict[outer_key])


    def train_dataloader(self):
        return DataLoader(dataset = self.train_loader, batch_size=self.batch_size, collate_fn=custom_collate, num_workers=self.num_workers, pin_memory = self.pin_memory)

    def test_dataloader(self):
        return DataLoader(dataset = self.test_loader, batch_size=self.batch_size, collate_fn=custom_collate, num_workers=self.num_workers, pin_memory = self.pin_memory)




def custom_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        return torch.stack([torch.as_tensor(b) for b in batch], 0)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: custom_collate([d[key] for d in batch]) if not key.startswith("meta_") else [d[key] for d in batch] for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        return [custom_collate(samples) for samples in zip(*batch)]
    raise TypeError(f"batch must contain tensors, numpy arrays, numbers, dicts or lists; found {elem_type}")