import pandas as pd
import os

def main():

    pickle_path = os.path.join("obj", "df_chunked.pkl")
    chunked_df = pd.read_pickle(pickle_path)
    
    chunk_size = 50000
    for i, start in enumerate(range(0, len(chunked_df), chunk_size), start=1):
        chunk = chunked_df.iloc[start:start+chunk_size]
        chunk.to_pickle(f'chunk_{i}.pkl')

if __name__ == "__main__":
    main()