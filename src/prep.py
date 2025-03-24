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

spacy.load("en_core_web_lg")

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
    crossover_path = os.path.join("..", "cross.csv")
    prep_stats_path = os.path.join("out", "prep_stats.txt")
    au_fandom_path = os.path.join("out", "au_fandom_counts.csv")
    pickle_path = os.path.join("obj", "prepped_data.pkl")

    #load data
    df = load_data(data_path)
    cross_df = pd.read_csv(crossover_path, header=None)

    # preprocessing 
    print("Fixing data types...")
    # get existing and planned chapter counts
    df[['current_chapters', 'total_chapters']] = df['chapters'].str.split('/', expand=True) 
    df = df.drop(columns=['chapters'])
    # column types
    str_columns = ['author', 'category', 'fandom', 'relationship', 'character', 'additional tags', 'all_kudos']
    int_columns = ['words', 'comments', 'kudos', 'bookmarks', 'hits']
    date_cols = ['published', 'status date']
    # type conversions
    df = str_cols_to_list(df, str_columns)
    df = str_cols_to_int(df, int_columns)
    df = str_to_date(df, date_cols)
    # for stats
    len_orig = df.shape[0]

    # remove crossovers
    print("Removing crossovers...")
    cross_values = cross_df.iloc[:, 0].values
    df = df[~df['work_id'].isin(cross_values)]
    # for stats
    len_no_xover = df.shape[0]

    # remove duplicates
    print("Dropping duplicates...")
    df = df.drop_duplicates(subset='work_id')
    # for stats
    len_no_dupes = df.shape[0]

    # remove fics that are too short
    print("Removing fics that are too short...")
    df = df[df['words'] >= 600]
    # for stats
    len_length = df.shape[0]

    # remove fics that don't pass quality check
    # get snippets
    print("Getting snippets for quality check...")
    snippets = []
    for text in tqdm(list(df["body"]), total=df.shape[0]):
        text = str(text)
        clean_text = text.replace("\n", " ")
        tokenized_text = clean_text.split(" ")
        tokenized_text = [token for token in tokenized_text if token != ""]
        snippets.append(" ".join(tokenized_text[100:600]))
    # assess quality
    print("Checking text quality. This will take a while...")
    metrics = td.extract_metrics(
        text=tqdm(snippets), 
        spacy_model='en_core_web_lg', 
        metrics=['quality'])
    df['passed_qual_check'] = list(metrics['passed_quality_check'])
    # remove fics that failed check
    df = df[df['passed_qual_check'] == True]
    # for stats
    len_final = df.shape[0]

    # stats for how many fics got excluded
    print("Getting some final stats...")
    with open(prep_stats_path, "w") as file:
        file.write(f"Original number of fics: {len_orig}\n")
        file.write(f"After removing crossovers: {len_no_xover}\n")
        file.write(f"After removing removing duplicates: {len_no_dupes}\n")
        file.write(f"After removing fics of insufficient length: {len_length}\n")
        file.write(f"After removing fics of insufficient quality: {len_final}\n")

    # stats for how many fics remain
    au_fandom_counts = pd.crosstab(df['Main AU'], df['Main fandom'])
    au_fandom_counts.to_csv(au_fandom_path)

    # pickle df for easy use in next script
    df.to_pickle(pickle_path)

if __name__ == "__main__":
    main()