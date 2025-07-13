
from pathlib import Path
import os
import pandas as pd
import gzip
import shutil
import glob
from collections import defaultdict
from tqdm import tqdm


def create_csv_dict(dir_individual_flights):
    """
    Create nested dictionary based on dir_individual_flights
    """
    d = defaultdict(lambda: {})
    flight_folders = os.listdir(dir_individual_flights)
    for flight_folder in flight_folders:
        flight_folder_dir = dir_individual_flights + flight_folder + "/"
        if not os.path.isdir(flight_folder_dir):
            continue
    
        dict_ids = {}
        id_keys = os.listdir(flight_folder_dir) 
        for id_key in id_keys:
            id_key_dir = flight_folder_dir + id_key + "/"
            if not os.path.isdir(id_key_dir):
                continue
    
            csv_list = glob.glob(id_key_dir + "*.csv")
            if len(csv_list) > 0:
                dict_ids[id_key] = csv_list[0]
            
        if len(dict_ids.keys()) > 0:
            d[flight_folder] = dict_ids

    return d



def unzip_gzip_util(gz_file_or_folder_path, unzip_here_path = None):
    """
    unzip the gzip file or files in directory
    if directory, go over all gz files and unzip all of them

    args:
        gz_file_or_folder_path, str: string path of gz file or folder with gz files
        unzip_here_path, str: string path of desired location to unzip gz files
    returns: 
        list_filenames, List[str]: list of string paths to each unzipped gz file
    """

    def unzip_single_gzip_file(gz_file_path, unzip_here_path):
        """
        Helper function to unzip a single gz file and save it in the unzip path
        """
        filename = os.path.basename(gz_file_path.replace('.gz', ''))
        with gzip.open(gz_file_path, 'rb') as f_in:
            with open(unzip_here_path + filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        return filename

    assert os.path.exists(gz_file_or_folder_path), f"file or folder does not exist: {gz_file_or_folder_path}"
    is_folder = os.path.isdir(gz_file_or_folder_path)
    folder_path = gz_file_or_folder_path if is_folder else str(Path(gz_file_or_folder_path).parent)
    
    if unzip_here_path is None:
        unzip_here_path = folder_path

    list_filenames = []
    file_paths = [gz_file_or_folder_path] if not is_folder else glob.glob(gz_file_or_folder_path + '*.gz')
    for file_path in file_paths:
        unzipped_filename = unzip_single_gzip_file(file_path, unzip_here_path)
        list_filenames.append(unzipped_filename)

    return list_filenames



def group_files_by_type(csv_folder_path):
    """
    A file might have a prefix IFF or EV or RD, this function creates a defaultdict to first group by common name 
    (shared by the 3 files in the group, ex: USA_20240518_050003_86395_LADDfiltered.csv)

    Example output: 
    default_dict with keys such as: "USA_20240518_050003_86395_LADDfiltered" this key will have value which is a regular dictionary: 
        {'IFF': '/Users/.../IFF_USA_20240518_050003_86395_LADDfiltered.csv',
         'EV': '/Users/.../EV_USA_20240518_050003_86395_LADDfiltered.csv',
         'RD': '/Users/.../RD_USA_20240518_050003_86395_LADDfiltered.csv'}

    This way we are able to group relevent files by the different prefixes
    
    """
    d = defaultdict(lambda: {})

    IFF_files = glob.glob(csv_folder_path + "IFF*.csv")
    all_files = glob.glob(csv_folder_path + "*.csv")
    for filepath_IFF in IFF_files:
        folder_path = str(Path(filepath_IFF).parent) + "/"
        filename_without_extension = os.path.basename(filepath_IFF).split(".")[0]
        filename_no_prefix = filename_without_extension.replace("IFF_", "")

        filename_EV = "EV_" + filename_no_prefix + ".csv"
        filename_RD = "RD_" + filename_no_prefix + ".csv"

        filepath_EV = folder_path + filename_EV
        filepath_RD = folder_path + filename_RD

        d[filename_no_prefix]["IFF"] = filepath_IFF
        if filepath_EV in all_files:
            d[filename_no_prefix]["EV"] = filepath_EV
        if filepath_RD in all_files:
            d[filename_no_prefix]["RD"] = filepath_RD

    return d


def read_iff(filepath_IFF, num_lines_to_read_at_a_time = 50_000):
    """
    Modified from the original NASA code to get the entire dataframe, 
    no need to specify origin, est origin / dest, est dest... this will be taken care of in the itterator that yields filtered data
    """


    columns=['fltKey', 'CID', 'UAID', 'Time', 'Latitude', 'Longitude',
                                                                'Altitude', 'PointSource', 'RecTypeCat', 'Significance',
                                                                'GroundSpeed', 'FlightCourse']

    dictToTurnIntoDataFrame = {key: [] for key in columns}
    num_lines = sum(1 for _ in open(filepath_IFF))
    tqdm_count = (num_lines // num_lines_to_read_at_a_time) + 1

    with tqdm(total=num_lines) as pbar:
        with open(filepath_IFF, 'r+') as f:
            for idx, line in enumerate(f):
                iffWrite = line.split(",")
    
                if iffWrite[0] == "3" :  # record type 3 in iff documentation 
                    dictToTurnIntoDataFrame['fltKey'].append(str(iffWrite[2])) # msn number 
                    dictToTurnIntoDataFrame['CID'].append(str(iffWrite[4]))
                    dictToTurnIntoDataFrame['UAID'].append(str(iffWrite[7]))
                    dictToTurnIntoDataFrame['PointSource'].append(str(iffWrite[6]))
                    dictToTurnIntoDataFrame['Time'].append(int(float(iffWrite[1])))
                    dictToTurnIntoDataFrame['Latitude'].append(float(iffWrite[9]))
                    dictToTurnIntoDataFrame['Longitude'].append(float(iffWrite[10]))
                    dictToTurnIntoDataFrame['Altitude'].append(float(iffWrite[11]))
                    dictToTurnIntoDataFrame['RecTypeCat'].append(str(iffWrite[8]))
                    dictToTurnIntoDataFrame['Significance'].append(str(iffWrite[12]))
                    dictToTurnIntoDataFrame['GroundSpeed'].append(str(iffWrite[16]))
                    dictToTurnIntoDataFrame['FlightCourse'].append(str(iffWrite[17]))
              

                pbar.update(1)
    print("converting to dataframe...")
    df = pd.DataFrame(dictToTurnIntoDataFrame, 
                      columns=columns)

    # Rabit hole of Pandas datatype conversion, unfortunatly specifying a dictionary of datatypes in the pd.DataFrame breaks ):
    for string_col in ['fltKey', 'CID', 'UAID', 'PointSource', 'RecTypeCat', 'Significance', 'GroundSpeed', 'FlightCourse']:
        df[string_col] = df[string_col].astype(str)
    print("converted")
    
    return df


#####################


def helper_update_csv_storage(flight_dfs, individual_flights_dir):
    """
    Helper function to update a folder filled with folders of flight data 
    Goes over nested dictionary of flight dataframes, saves each one in designated folder, updates csv if two filights map to the same folder
    
    args: 
        flight_dfs: Type, Dict{fltKey: Dict{UAID: Dataframe}}: Nested dictionary of Dataframes of individual flightpaths
        individual_flights_dir: Type, str: Day-long clean flight dataframe filepath (big dataframe of many flights over the course of one day)
    """

    # Iterate over flightnumbers
    for key_flight, dict_flight in tqdm(flight_dfs.items()):
        key_flight = str(key_flight)
        dir_flight = individual_flights_dir + key_flight + "/"
        Path(dir_flight).mkdir(parents=True, exist_ok=True)     # create folder if none yet exists

        for key_id, dataframe in dict_flight.items():
            key_id = str(key_id)
            key_dir = dir_flight + key_id + "/"
            Path(key_dir).mkdir(parents=True, exist_ok=True)

            csv_name = key_flight + "_" + key_id + ".csv"
            csv_path = key_dir + csv_name

            # save new csv if it does not exist: 
            if not os.path.isfile(csv_path):
                dataframe.to_csv(csv_path)
            else: # otherwise append to what already exists
                old_dataframe = pd.read_csv(csv_path)
                new_dataframe = pd.concat([old_dataframe, dataframe], axis=0, ignore_index=True)
                new_dataframe = new_dataframe.sort_values(by=['Time'], ascending=True)
                new_dataframe.to_csv(csv_path)

def dynamically_create_dictionary_of_flight_dataframes(csv_paths_list, individual_flights_dir):
    """
    Go over every (clean) day-long flights csv one by one, split by flight, save every individual flight to individual_flights_dir
    # Group by 'key' and then by 'id'
    grouped_by_key = df.groupby('key')
    result = {key: {id: group for id, group in key_group.groupby('id')} for key, key_group in grouped_by_key}

    UAID | CID

    params: 
        csv_paths_list: Type, List[str]: List of paths to csv files
         individual_flights_dir: Type, str: output where individual flights will be saved
    """

    for idx, csv_path in enumerate(csv_paths_list):
        print(f"working on reading in {csv_path} | {idx + 1} / {len(csv_paths_list)}")
        current_day_of_flight_data_dataframe = pd.read_csv(csv_path)
        print("large csv read in")
        grouped = current_day_of_flight_data_dataframe.groupby('fltKey')
        flight_dfs = {key: {id: group.sort_values(by=['Time'], ascending=True) for id, group in key_group.groupby('UAID')} for key, key_group in grouped}
        print("grouped by individual flights")
        # Save the individual csv files for every flight
        helper_update_csv_storage(flight_dfs, individual_flights_dir)











