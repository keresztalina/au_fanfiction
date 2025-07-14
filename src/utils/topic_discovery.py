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

    """
    Removes named entities from a list of text documents using spaCy.

    This function processes each document in the input corpus, removes tokens 
    that are part of any named entity (e.g., PERSON, ORG, GPE), and returns 
    a new corpus with named entities removed.

    Parameters:
    -----------
    corpus : list of str
        A list of text documents to be processed.
    n_process : int, optional (default=8)
        Number of parallel processes to use for spaCy's pipeline (for faster processing).

    Returns:
    --------
    new_corpus : list of str
        A list of cleaned documents with named entities removed.
    """

    nlp = spacy.load("en_core_web_lg")

    new_corpus = []
    for doc in tqdm(nlp.pipe(corpus, disable=["parser", "tagger"], n_process=n_process), total=len(corpus)):
        tokens = [token.text for token in doc if not token.ent_type_]
        new_corpus.append(" ".join(tokens))

    return new_corpus

