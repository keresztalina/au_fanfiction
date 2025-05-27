import os
import pandas as pd 
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from utils.preprocessing import split_text_with_id


def main():

    # paths
    data_path = os.path.join("obj", "data", "prepped_data.pkl")
    pickle_path = os.path.join("obj", "data", "df_chunked.pkl")
    embeddings_path = os.path.join("obj", "embeddings", "embeddings.npy")

    # read data
    df = pd.read_pickle(data_path)

    # get chunks of length suitable for transformer processing
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking text"):
        chunks = split_text_with_id(row['body'])
        rows.extend({
            'work_id': row['work_id'], 
            'chunk_id': chunk_id, 
            'body': chunk} for chunk_id, chunk in chunks)
    
    # get chunked df and pickle
    df_chunked = pd.DataFrame(rows)
    df_chunked.to_pickle(pickle_path)

    # get corpus
    corpus = df_chunked['body'].tolist()

    # load embedding model
    model = SentenceTransformer("paraphrase-mpnet-base-v2")

    # generate embeddings
    embeddings = model.encode(
        corpus,
        show_progress_bar=True)

    # save embeddings
    np.save(embeddings_path, embeddings)
    
if __name__ == "__main__":
    main()