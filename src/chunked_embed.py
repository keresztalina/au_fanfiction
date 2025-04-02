import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

def split_text_with_id(text, chunk_size=350):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return [(idx, chunk) for idx, chunk in enumerate(chunks)]

def main():

    # paths
    data_path = os.path.join("obj", "prepped_data.pkl")
    pickle_path = os.path.join("obj", "df_chunked.pkl")
    embeddings_path = os.path.join("obj", "embeddings.npy")

    # read data
    df = pd.read_pickle(data_path)

    # get chunks of length suitable for transformer processing
    rows = []
    for _, row in df.iterrows():
        chunks = split_text_with_id(row['body'])
        rows.extend({
            'work_id': row['work_id'], 
            'chunk_id': chunk_id, 
            'body': chunk} for chunk_id, chunk in chunks)
    
    # get chunked df and pickle
    df_chunked = pd.DataFrame(rows)
    df_chunked.to_pickle(pickle_path)

    # load embedding model
    model = SentenceTransformer("paraphrase-mpnet-base-v2")

    # set up parallelization
    pool = model.start_multi_process_pool()

    # generate embeddings
    embeddings = model.encode_multi_process(
        test_corpus, 
        pool=pool, 
        show_progress_bar=True)

    # save embeddings
    np.save(embeddings_path, embeddings)
    
if __name__ == "__main__":
    main()