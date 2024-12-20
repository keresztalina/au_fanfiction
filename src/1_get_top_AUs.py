import os
import pandas as pd
import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import re

# FUNCTIONS
def load_data(path_to_folder):
    all_files = os.listdir(path_to_folder) # get all files
    valid_files = [file for file in all_files if 'errors' not in file] # remove error logs
    dfs = [] # empty list
    for file in valid_files:
        file_path = os.path.join(path_to_folder, file)
        try:
            df = pd.read_csv(file_path)
            dfs.append(df) # load files into list of dfs
        except Exception as e:
            print(f"Could not read {file}: {e}") # in case sth is missed
    big_df = pd.concat(dfs, ignore_index=True) # pull into df
    return big_df

def string_to_list(string):
    # turn a string of tags/authors/categories etc. separated by commas into a list
    list = string.apply(lambda x: x.split(", ") if isinstance(x, str) else [])
    return list

def filter_list(list):
    # filter a list for items referencing AUs
    pattern = re.compile(r'\b(alternate universe|au)\b', re.IGNORECASE)
    return [item for item in list if pattern.search(item)]

def list_counter(list):

    # count how many times each item appears in a column of lists       
    all_items = [item for sublist in list for item in sublist]
    counts_coll = collections.Counter(all_items)
    counts = pd.DataFrame(
        counts_coll.items(), 
        columns=['Tag', 'Count']) # df for better data handling
    counts['Count'] = pd.to_numeric(counts['Count'], errors='coerce') # dtype fix
    return counts

def plot_most_common(counts: pd.DataFrame, # with item counts
                    number: int, # how many of the most common ones?
                    ylab: str, # name of the variable to be counted
                    title: str, # plot title
                    bar_color: str, 
                    xtick: int): # where to place ticks for readability

    # find n most common items
    most_common = counts.nlargest(number, 'Count') 

    plt.figure(figsize=(10, 6))
    plt.barh(most_common[ylab], most_common['Count'], color=bar_color) # horizontal bars for readability
    plt.xlabel('Count')
    plt.ylabel(ylab)
    plt.title(title)
    plt.gca().xaxis.set_major_locator(MultipleLocator(xtick)) # ticks for readability
    plt.gca().invert_yaxis()
    plt.savefig('plots/top_20_au.png', format='png', bbox_inches='tight') # save entire plot


# MAIN
def main():

    # load data
    folder_path = 'metadata_all/test_folder'
    meta = load_data(folder_path)

    # drop any duplicates from scraping error
    meta = meta.drop_duplicates(subset=['work_id'])

    # split tag field into list of tags, find only AU-related ones
    meta['additional tags'] = string_to_list(meta['additional tags'])
    meta['au tags'] = meta['additional tags'].apply(filter_list)

    # get AU counts and plot most common types
    au_counts = list_counter(meta['au tags'])
    plot_most_common(au_counts, 20, 'Tag', 'Most popular AUs', 'skyblue', 5)

    # save data in case it needs to be loaded later
    meta.to_pickle('obj/meta.pkl')
    au_counts.to_csv('obj/au_counts.csv', index=False)

if __name__ == "__main__":
    main()