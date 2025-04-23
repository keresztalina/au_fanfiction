import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from turftopic import KeyNMF

def main(num_topics: int, top_n: int):
    # Load corpus
    with open(os.path.join("obj", "corpus_ner.pkl"), "rb") as f:
        corpus = pickle.load(f)

    # Load embeddings
    embeddings = np.load(os.path.join("obj", "embeddings.npy"))

    # Setup vectorizer and encoder
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
    encoder = SentenceTransformer("paraphrase-mpnet-base-v2")

    # Fit KeyNMF
    model = KeyNMF(num_topics, top_n=top_n, vectorizer=vectorizer, encoder=encoder)
    topic_data = model.prepare_topic_data(corpus, embeddings=embeddings)

    # Pickle result
    filename = os.path.join("obj", f"ner_{num_topics}_{top_n}.pkl")
    with open(filename, "wb") as file:
        pickle.dump(topic_data, file)

    print(f"Saved topic data to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KeyNMF on preprocessed corpus and embeddings.")
    parser.add_argument("--num_topics", type=int, default=10, help="Number of topics")
    parser.add_argument("--top_n", type=int, default=15, help="Number of top words per topic")
    args = parser.parse_args()

    main(num_topics=args.num_topics, top_n=args.top_n)
