import os
import pandas as pd
from tqdm import tqdm

from utils.preprocessing import (
    load_data,
    str_cols_to_list,
    str_cols_to_int,
    str_to_date,
    get_quality_checked_snippets,
)

def main():
    # paths
    data_path = os.path.join("obj", "texts")
    crossover_path = os.path.join(data_path, "cross.csv")
    prep_stats_path = os.path.join("obj", "text_files", "prep_stats.txt")
    au_fandom_path = os.path.join("obj", "text_files", "au_fandom_counts.csv")
    pickle_path = os.path.join("obj", "data", "prepped_data.pkl")

    # load data
    df = load_data(data_path)
    cross_df = pd.read_csv(crossover_path, header=None)

    # preprocessing 
    print("Fixing data types...")
    df[['current_chapters', 'total_chapters']] = df['chapters'].str.split('/', expand=True)
    df = df.drop(columns=['chapters'])

    str_columns = ['author', 'category', 'fandom', 'relationship', 'character', 'additional tags', 'all_kudos']
    int_columns = ['words', 'comments', 'kudos', 'bookmarks', 'hits']
    date_cols = ['published', 'status date']

    df = str_cols_to_list(df, str_columns)
    df = str_cols_to_int(df, int_columns)
    df = str_to_date(df, date_cols)

    len_orig = df.shape[0]

    print("Removing crossovers...")
    cross_values = cross_df.iloc[:, 0].values
    df = df[~df['work_id'].isin(cross_values)]
    len_no_xover = df.shape[0]

    print("Dropping duplicates...")
    df = df.drop_duplicates(subset='work_id')
    len_no_dupes = df.shape[0]

    print("Removing fics that are too short...")
    df = df[df['words'] >= 600]
    len_length = df.shape[0]

    # quality check
    df = get_quality_checked_snippets(df)
    df = df[df['passed_qual_check'] == True]
    len_final = df.shape[0]

    print("Getting some final stats...")
    with open(prep_stats_path, "w") as file:
        file.write(f"Original number of fics: {len_orig}\n")
        file.write(f"After removing crossovers: {len_no_xover}\n")
        file.write(f"After removing removing duplicates: {len_no_dupes}\n")
        file.write(f"After removing fics of insufficient length: {len_length}\n")
        file.write(f"After removing fics of insufficient quality: {len_final}\n")

    au_fandom_counts = pd.crosstab(df['AU'], df['Fandom'])
    au_fandom_counts.to_csv(au_fandom_path)

    df.to_pickle(pickle_path)

if __name__ == "__main__":
    main()