import os
import pandas as pd
import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import re

# FUNCTIONS
def string_to_list(string):

    list = string.apply(lambda x: x.split(", ") if isinstance(x, str) else [])
    return list

def filter_list(list):
    # Convert tag list to lowercase and check if 'alternate universe' or 'au' is in the tag
    pattern = re.compile(r'\b(alternate universe|au)\b', re.IGNORECASE)
    return [item for item in list if pattern.search(item)]

def list_counter(list):

    # count how many times each item appears in a column of lists       
    all_items = [item for sublist in list for item in sublist]
    counts_coll = collections.Counter(all_items)
    counts = pd.DataFrame(counts_coll.items(), columns=['Tag', 'Count'])
    counts['Count'] = pd.to_numeric(counts['Count'], errors='coerce')
    return counts

def plot_most_common(counts: pd.DataFrame, 
                    number: int, 
                    ylab: str, 
                    title: str,
                    bar_color: str,
                    xtick: int):

    most_common = counts.nlargest(number, 'Count')

    plt.figure(figsize=(10, 6))
    plt.barh(most_common[ylab], most_common['Count'], color=bar_color)
    plt.xlabel('Count')
    plt.ylabel(ylab)
    plt.title(title)
    plt.gca().xaxis.set_major_locator(MultipleLocator(xtick))
    plt.gca().invert_yaxis()
    plt.savefig('plots/top_20_au.png', format='png', bbox_inches='tight')


# MAIN
def main():
    folder_path = 'metadata_all/test_folder'
    all_files = os.listdir(folder_path) # get all files
    valid_files = [file for file in all_files if 'errors' not in file] # remove error logs
    dfs = []
    for file in valid_files:
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"Could not read {file}: {e}")
    meta = pd.concat(dfs, ignore_index=True)
    meta['additional tags'] = string_to_list(meta['additional tags'])
    meta['au tags'] = meta['additional tags'].apply(filter_list)
    au_counts = list_counter(meta['au tags'])
    plot_most_common(au_counts, 20, 'Tag', 'Most popular AUs', 'skyblue', 5)
    meta.to_pickle('obj/meta.pkl')
    au_counts.to_csv('obj/au_counts.csv', index=False)

if __name__ == "__main__":
    main()