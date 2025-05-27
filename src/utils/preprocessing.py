import os
import pandas as pd
import re
import numpy as np
import spacy
import textdescriptives as td
from tqdm import tqdm
from itertools import islice

spacy.load("en_core_web_lg")

def load_data(path_to_folder):
    """
    Loads and combines CSV files from a specified folder into a single DataFrame.

    This function reads all `.csv` files in the given folder, excluding those that contain 
    "errors" or "cross" in their filenames (case-insensitive). It extracts fandom and AU labels 
    from each filename (assumed to be in the format "Fandom_AU.csv"), adds them as columns, and 
    returns a combined DataFrame.

    Parameters:
        path_to_folder (str): Path to the folder containing the CSV files.

    Returns:
        pd.DataFrame: Combined DataFrame with all loaded data, including 'Fandom' and 'AU' columns.
                      Returns an empty DataFrame if no valid files are found.
    """

    # list files and exclude error logs and crossover file
    all_files = os.listdir(path_to_folder) 
    valid_files = [file for file in all_files if 'errors' not in file.lower() and 'cross' not in file.lower()]
    dfs = []

    # loop through files
    for file in valid_files:
        file_path = os.path.join(path_to_folder, file)

        # retrieve fandom and AU id from filename
        filename = str(file.split('.')[0])
        parts = filename.split('_')
        if len(parts) < 2:
            print(f"Skipping file with unexpected name format: {file}")
            continue
        fandom = parts[0].capitalize()
        au = parts[1].capitalize()

        # load file and append to list of dfs
        try:
            print(f"Loading file: {file}")
            df = pd.read_csv(file_path)
            df['Fandom'] = fandom
            df['AU'] = au
            dfs.append(df)
            print(f"Finished loading file: {file}")
        except Exception as e:
            print(f"Could not read {file}: {e}")
    
    # combine dfs
    if dfs:
        big_df = pd.concat(dfs, ignore_index=True)
    else:
        big_df = pd.DataFrame()
    return big_df


def str_to_list(value):
    """
    Converts a string representation of a list to a Python list.

    If the value is a string like "['item1', 'item2']", it will be converted to 
    ['item1', 'item2']. If the value is NaN, an empty list is returned. If the value 
    is already a list or another type, it is returned as-is.

    Parameters:
        value (any): The value to convert.

    Returns:
        list or original value: A list if input was a string or NaN, otherwise the original value.
    """
    if pd.isna(value): 
        return []
    elif isinstance(value, str):
        return value.strip("[]").replace("'", "").split(", ")
    return value


def str_cols_to_list(data, cols):
    """
    Applies `str_to_list` to multiple columns in a DataFrame.

    Useful for converting columns that contain string representations of lists into actual lists.

    Parameters:
        data (pd.DataFrame): The DataFrame to process.
        cols (list of str): List of column names to apply the transformation to.

    Returns:
        pd.DataFrame: The updated DataFrame with specified columns converted.
    """
    for col in cols:
        data[col] = data[col].apply(str_to_list)
    return data


def str_to_int(value):
    """
    Converts a value to an integer.

    Handles strings with commas and decimal points (e.g., "1,000.0"), floats, and NaN values.
    If conversion fails, returns 0 and prints the problematic value.

    Parameters:
        value (any): The value to convert.

    Returns:
        int: The converted integer or 0 if conversion fails.
    """
    if pd.isna(value): 
        return 0
    elif isinstance(value, int):
        return value
    elif isinstance(value, float):
        return int(value)
    elif isinstance(value, str):
        value = value.replace(",", "").replace(".0", "")
        return int(value)
    else:
        print(f"Trouble with data: {value}")
        return 0


def str_cols_to_int(data, cols):
    """
    Applies `str_to_int` to multiple columns in a DataFrame.

    Useful for converting columns with string or float representations of numbers to integers.

    Parameters:
        data (pd.DataFrame): The DataFrame to process.
        cols (list of str): List of column names to apply the transformation to.

    Returns:
        pd.DataFrame: The updated DataFrame with specified columns converted.
    """
    for col in cols:
        data[col] = data[col].apply(str_to_int)
    return data


def str_to_date(data, cols):
    """
    Converts one or more DataFrame columns to datetime objects.

    Uses pandas' `to_datetime` with error coercion, so invalid formats become NaT.

    Parameters:
        data (pd.DataFrame): The DataFrame to process.
        cols (list of str): List of column names to convert to datetime.

    Returns:
        pd.DataFrame: The updated DataFrame with datetime-converted columns.
    """
    for col in cols:
        data[col] = pd.to_datetime(data[col], errors='coerce')
    return data

def get_quality_checked_snippets(df):
    """
    Extracts mid-section text snippets from fanfiction and performs a quality check.

    This function processes the 'body' column in the given DataFrame by:
    - Extracting tokens 100–600 from each text
    - Running a quality assessment using the `textdescriptives` package and spaCy
    - Adding a new column `passed_qual_check` to indicate whether each snippet passed the quality check

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'body' column with the fanfiction text.

    Returns:
        pd.DataFrame: The input DataFrame with an additional boolean column `passed_qual_check`.
    """

    print("Getting snippets for quality check...")
    snippets = []
    for text in tqdm(list(df["body"]), total=df.shape[0]):
        text = str(text).replace("\n", " ")
        tokens = [token for token in text.split(" ") if token != ""]
        snippets.append(" ".join(tokens[100:600]))

    print("Checking text quality. This will take a while...")
    metrics = td.extract_metrics(
        text=tqdm(snippets), 
        spacy_model='en_core_web_lg', 
        metrics=['quality']
    )
    df['passed_qual_check'] = list(metrics['passed_quality_check'])
    return df


def split_text_with_id(text, chunk_size=350):
    words = text.split()
    return ((i, " ".join(chunk)) for i, chunk in enumerate(
        (islice(words, i, i + chunk_size) for i in range(0, len(words), chunk_size)))) 