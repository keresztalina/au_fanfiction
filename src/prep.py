import os
import pandas as pd
import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import re
import numpy as np
import spacy
import textdescriptives as td
from tqdm import tqdm
import seaborn as sns

def load_data(path_to_folder):
    all_files = os.listdir(path_to_folder) 
    valid_files = [file for file in all_files if 'errors' not in file] # remove error logs
    dfs = []
    for file in valid_files:
        file_path = os.path.join(path_to_folder, file)
        filename = str(file.split('.')[0])
        fandom = str(filename.split('_')[0])
        au = str(filename.split('_')[1])
        try:
            print(f"Loading file: {file}")
            df = pd.read_csv(file_path)
            df['Main fandom'] = fandom
            df['Main AU'] = au
            dfs.append(df) # load files into list of dfs
            print(f"Finished loading file: {file}")
        except Exception as e:
            print(f"Could not read {file}: {e}") # in case sth is missed
    big_df = pd.concat(dfs, ignore_index=True) # pull into df
    return big_df

def str_to_list(value):
    if pd.isna(value): 
        value = []
        return value
    elif isinstance(value, str):
        value = value.strip("[]").replace("'", "").split(", ")
        return value
    else:
        pass
    
def str_cols_to_list(data, cols):
    for col in cols:
        data[col] = data[col].apply(str_to_list)
    return data

def str_to_int(value):
    if pd.isna(value): 
        value = 0
        value = int(value)
        return value
    elif isinstance(value, int):
        pass
    elif isinstance(value, float):
        value = int(value)
        return(value)
    elif isinstance(value, str):
        value = value.replace(",", "").replace(".0", "")
        value = int(value)
        return value
    else:
        print(f"Trouble with data: {value}")

def str_cols_to_int(data, cols):
    for col in cols:
        data[col] = data[col].apply(str_to_int)
    return data

def str_to_date(data, cols):
    for col in cols:
        data[col] = pd.to_datetime(data[col], errors='coerce')
    return data

def main():

    #paths
    data_path = os.path.join("..", "texts")

    #load data
    df = load_data(data_path)

    # preprocessing 