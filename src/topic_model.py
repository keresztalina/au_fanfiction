import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from turftopic import KeyNMF

def main(num_topics: int, top_n: int):

    # file paths
    corpus_path = os.path.join("obj", "corpus_ner.pkl")
    keywords_path = os.path.join("obj", f"keywords_{top_n}.pkl")
    model_path = os.path.join("obj", f"ner_{num_topics}_{top_n}.pkl")

    # load corpus
    with open(corpus_path, "rb") as f:
        corpus = pickle.load(f)

    # load pre-computed embeddings
    embeddings = np.load(os.path.join("obj", "embeddings.npy"))

    # setup vectorizer and encoder
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
    encoder = SentenceTransformer("paraphrase-mpnet-base-v2")

    # fit KeyNMF
    model = KeyNMF(num_topics, top_n=top_n, vectorizer=vectorizer, encoder=encoder)

    # retrieve keywords
    # check if existing keywords matrix exists for this n keywords, otherwise create one
    #if os.path.exists(keywords_path):
    #    with open(keywords_path, "rb") as f:
    #        keywords = pickle.load(f)
    #    print("Found pre-computed keywords matrix! Proceeding with topic discovery...")
    #else:
    #    print("No pre-computed keywords matrix found! Proceeding with keyword extraction...")
    #    keywords = model.extract_keywords(corpus)
    #    with open(keywords_path, "rb") as f:
    #        pickle.dump(keywords, f)

    # discover topics
    topic_data = model.prepare_topic_data(corpus, embeddings=embeddings)

    # save model
    with open(model_path, "wb") as f:
        pickle.dump(topic_data, f)

    print(f"Saved topic data to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KeyNMF on preprocessed corpus and embeddings.")
    parser.add_argument("--num_topics", type=int, default=10, help="Number of topics")
    parser.add_argument("--top_n", type=int, default=15, help="Number of top words per topic")
    args = parser.parse_args()

    main(num_topics=args.num_topics, top_n=args.top_n)
