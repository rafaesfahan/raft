import pandas as pd
import gzip
import shutil
import os
import csv

# directory_files = "/raid/mo0dy/tryouts/"
# iffFileName = 'IFF_USA_20240705_050000_86395_LADDfiltered.csv'
# rdFileName = 'RD_USA_20240705_050000_86395_LADDfiltered.csv' 
# outputFileName = "/raid/mo0dy/cleaned/20240705.csv"

directory_files = "/raid/mo0dy/tryouts/"
iffFileName = 'IFF_USA_20240622_050000_86398_LADDfiltered.csv'
rdFileName = 'RD_USA_20240622_050000_86398_LADDfiltered.csv' 
outputFileName = "/raid/mo0dy/cleaned/20240622.csv"

iffFileName_dir = os.path.join(directory_files, iffFileName)
rdFileName_dir = os.path.join(directory_files, rdFileName)

def decompress_and_replace_file(file_path):
    if file_path.endswith('.csv.gz'):
        decompressed_file_path = file_path.replace('.gz', '')
        with gzip.open(file_path, 'rb') as f_in, open(decompressed_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(file_path)
        return decompressed_file_path
    return file_path

def read_rd_file(rd_path):
    df = pd.read_csv(rd_path, usecols=["Msn", "Orig", "EstOrig", "Dest", "EstDest"])
    return set(df['Msn'].tolist())  # use set for O(1) lookup

def get_merged_csv_file(flight_keys):
    output_data = {
        'fltKey': [], 'Time': [], 'Latitude': [], 'Longitude': [], 'Altitude': [],
        'RecTypeCat': [], 'Significance': [], 'GroundSpeed': [], 'FlightCourse': []
    }

    with open(iffFileName_dir, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0] != "3":
                continue
            try:
                key = int(row[2])
                if key in flight_keys:
                    output_data['fltKey'].append(row[2])
                    output_data['Time'].append(int(float(row[1])))
                    output_data['Latitude'].append(float(row[9]))
                    output_data['Longitude'].append(float(row[10]))
                    output_data['Altitude'].append(float(row[11]))
                    output_data['RecTypeCat'].append(row[8])
                    output_data['Significance'].append(row[12])
                    output_data['GroundSpeed'].append(row[16])
                    output_data['FlightCourse'].append(row[17])
            except (IndexError, ValueError):
                continue  # ignore malformed lines

    df = pd.DataFrame(output_data)
    df.to_csv(outputFileName, index=False)
    print(f'{len(set(df["fltKey"]))} traces exported')
    return df.head(15)

# Run the full pipeline
flight_keys = read_rd_file(rdFileName_dir)
get_merged_csv_file(flight_keys)
