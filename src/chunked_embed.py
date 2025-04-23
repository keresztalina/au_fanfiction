import os
import pandas as pd 
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from itertools import islice

def split_text_with_id(text, chunk_size=350):
    words = text.split()
    return ((i, " ".join(chunk)) for i, chunk in enumerate(
        (islice(words, i, i + chunk_size) for i in range(0, len(words), chunk_size))))

def main():

    # paths
    data_path = os.path.join("obj", "prepped_data.pkl")
    pickle_path = os.path.join("obj", "df_chunked.pkl")
    embeddings_path = os.path.join("obj", "embeddings_2.npy")

    chunk_path = os.path.join("obj", "chunk_2.pkl")

    # read data
    #df = pd.read_pickle(data_path)

    # get chunks of length suitable for transformer processing
    #rows = []
    #for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking text"):
    #    chunks = split_text_with_id(row['body'])
    #    rows.extend({
    #        'work_id': row['work_id'], 
    #        'chunk_id': chunk_id, 
    #        'body': chunk} for chunk_id, chunk in chunks)
    
    # get chunked df and pickle
    #df_chunked = pd.DataFrame(rows)
    #df_chunked.to_pickle(pickle_path)
    df = pd.read_pickle(chunk_path)

    # get corpus
    corpus = df['body'].tolist()

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