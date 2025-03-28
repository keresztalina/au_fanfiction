import pandas as pd
import numpy as np
import os
import spacy
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def chunk_text(text, model, max_words=500):
    doc = model(text)
    sentences = [sent.text for sent in doc.sents]
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        if current_word_count + word_count > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def main():
    data_path = os.path.join("obj", "prepped_data.pkl")
    chunked_data_path = os.path.join("obj", "chunked_data.pkl")
    embeddings_path = os.path.join("obj", "chunked_embeddings.npy")
    
    df = pd.read_pickle(data_path)

    nlp = spacy.load("en_core_web_lg")
    new_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking text"):
        chunks = chunk_text(row['body'], nlp)
        for chunk in chunks:
            new_rows.append({'work_id': row['work_id'], 'body': chunk})
    chunked_df = pd.DataFrame(new_rows)
    chunked_corpus = chunked_df['body'].tolist()
    chunked_df.to_pickle(chunked_data_path)

    encoder = SentenceTransformer("paraphrase-mpnet-base-v2", backend="onnx")
    chunked_embeddings = np.asarray([encoder.encode(text) for text in tqdm(chunked_corpus, desc="Encoding texts")])
    np.save(embeddings_path, chunked_embeddings)
    
if __name__ == "__main__":
    main()