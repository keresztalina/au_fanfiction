import os
import pickle
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from turftopic import KeyNMF
from tqdm import tqdm

def remove_named_entities(corpus, n_process=8):

    nlp = spacy.load("en_core_web_lg")

    new_corpus = []
    for doc in tqdm(nlp.pipe(corpus, disable=["parser", "tagger"], n_process=n_process), total=len(corpus)):
        tokens = [token.text for token in doc if not token.ent_type_]
        new_corpus.append(" ".join(tokens))

    return new_corpus

