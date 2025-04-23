import os
import pandas as pd
import spacy
from tqdm import tqdm

# Load DataFrame
path = os.path.join("obj", "df_chunked.pkl")
df = pd.read_pickle(path)

# Convert 'body' column to a list
corpus = df["body"].tolist()

# Load spaCy model
nlp = spacy.load("en_core_web_lg")  # or another model if you prefer

# Process texts and remove named entities using parallelization
corpus_ner = []
for doc in tqdm(nlp.pipe(corpus, disable=["parser", "tagger"], n_process=8), total=len(corpus)):
    tokens = [token.text for token in doc if not token.ent_type_]
    corpus_ner.append(" ".join(tokens))

# Save processed corpus
output_path = os.path.join("obj", "corpus_ner.pkl")
with open(output_path, "wb") as f:
    pd.to_pickle(corpus_ner, f)
