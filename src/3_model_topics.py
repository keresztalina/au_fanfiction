import io
import sys
import os
import pickle
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from turftopic import KeyNMF
from tqdm import tqdm

from utils.topic_discovery import remove_named_entities

def main():

    print("Loading tools...")

    # load data
    df = joblib.load(os.path.join("obj", "data", "df_chunked.pkl"))
    corpus = df["body"].tolist()

    # load pre-computed embeddings
    embeddings = np.load(os.path.join("obj", "embeddings", "embeddings.npy"))

    # setup vectorizer and encoder
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
    encoder = SentenceTransformer("paraphrase-mpnet-base-v2")

    #extractor_model = KeyNMF(10, vectorizer=vectorizer)

    for vect in ["NEe"]: #"NEi", 

        print(f"Working on vectorization mode {vect}...")

        if vect == "NEe":
            print("Removing named entities from corpus...")
            corpus = remove_named_entities(corpus)
        else:
            pass

        print(f"Extracting keywords for vectorization mode {vect}...")
        #keywords = extractor_model.extract_keywords(corpus)

        for n_topics in [50, 100]: #10, 20, 30, 40, 

            print(f"Modeling for {n_topics} topics with {vect} vectorization...")
            model = KeyNMF(n_topics, vectorizer=vectorizer, encoder=encoder)
            #topic_data = model.prepare_topic_data(corpus, keywords=keywords, embeddings=embeddings)
            topic_data = model.prepare_topic_data(corpus, embeddings=embeddings)

            joblib.dump(topic_data, os.path.join("obj", "topic_models", f"{vect}_{n_topics}.pkl"))

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()

            # Call the method that only prints
            topic_data.print_topics()

            # Restore original stdout
            sys.stdout = old_stdout

            # Save captured output to a file
            with open(f"topics_output_{vect}_{n_topics}.txt", "w", encoding="utf-8") as f:
                f.write(buffer.getvalue())

if __name__ == "__main__":
    main()




