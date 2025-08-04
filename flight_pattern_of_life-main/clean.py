from pathlib import Path
import pandas as pd
import glob
from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm


def create_dictionary_of_flight_dataframes(csv_paths_list):
    csv_paths_list = csv_paths_list
    dataframes = []
    for idx, csv_path in enumerate(csv_paths_list):
        print(f"working on reading in {csv_path} | {idx + 1} / {len(csv_paths_list)}")
        dataframes.append(pd.read_csv(csv_path))

    dataframes = pd.concat(dataframes, axis=0, ignore_index=True)
    print(f"combined dataframe has shape: {dataframes.shape}")

    grouped = dataframes.groupby('fltKey')
    #flight_dfs = {fltKey: group.sort_values(by=['Time'], ascending=True) for fltKey, group in grouped}
    flight_dfs = {key: {id: group for id, group in key_group.groupby('fltKey')} for key, key_group in grouped}
    # flight_dfs = {key: {id: group for id, group in key_group} for key, key_group in grouped}
    dummy = 1

    return flight_dfs

dir_clean_iff = "/raid/mo0dy/cleaned/"
iff_csvs_path_list = glob.glob(dir_clean_iff + "*.csv")
flight_dfs = create_dictionary_of_flight_dataframes(iff_csvs_path_list)

for key_flight, dict_flight in flight_dfs.items():
    for key_id, dict_id in dict_flight.items():
        print(key_flight, key_id)


individual_flights_dir = "/raid/mo0dy/F/flights/"

for key_flight, dict_flight in tqdm(flight_dfs.items()):
    key_flight = str(key_flight)
    dir_flight = individual_flights_dir + key_flight + "/"
    Path(dir_flight).mkdir(parents=True, exist_ok=True)
    for key_id, dataframe in dict_flight.items():
        key_id = str(key_id)
        key_dir = dir_flight + key_id + "/"
        Path(key_dir).mkdir(parents=True, exist_ok=True)
        
        csv_name = key_flight + "_" + key_id + ".csv"
        dataframe.to_csv(key_dir + csv_name, )
