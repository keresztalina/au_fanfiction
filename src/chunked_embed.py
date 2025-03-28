import pandas as pd
import numpy as np
import os
import re
import spacy
from tqdm import tqdm
#from sentence_transformers import SentenceTransformer

def main():

    nlp = spacy.load("en_core_web_sm")

    def segment_sentences(text):
        # Replace newlines with spaces (to avoid breaking sentences across lines)
        text = text.replace('\n', ' ')
        
        # Regex to match sentence boundaries (periods, exclamation marks, question marks)
        sentence_endings = r'(?<=[.!?]) +'
        sentences = re.split(sentence_endings, text)
        
        # Remove empty sentences (if any) and strip leading/trailing whitespace
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        
        return sentences

    def chunk_text(text, max_words=500):
        doc = nlp.make_doc(text) 
        nlp.max_length = float('inf')

        sentences = segment_sentences(text)
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
                word_count = len(sentence.split()) 

                if current_word_count + word_count > max_words:
                    chunks.append(" ".join(current_chunk).strip()) 
                    current_chunk = [] 
                    current_word_count = 0

                current_chunk.append(sentence) 
                current_word_count += word_count 

        if current_chunk: 
            chunks.append(" ".join(current_chunk))

        return chunks

    def process_and_write_chunks(input_df, output_file):
        with open(output_file, mode='w', encoding='utf-8') as f:
            # Set up the CSV writer
            f.write("work_id,body\n")
            
            # Use tqdm to show progress over rows of the dataframe
            for _, row in tqdm(input_df.iterrows(), total=input_df.shape[0], desc="Processing Rows"):
                # Get the chunks for each text
                chunks = chunk_text(row['body'], max_words=500)
                # Write each chunk as a new row in the CSV file
                for chunk in chunks:
                    f.write(f"{row['work_id']},{chunk}\n")
    data_path = os.path.join("obj", "prepped_data.pkl")
    output_path = os.path.join("obj", "chunked_data.csv")
    #embeddings_path = os.path.join("obj", "chunked_embeddings.npy")
    
    df = pd.read_pickle(data_path)
    nlp = spacy.load("en_core_web_sm")

    process_and_write_chunks(df, output_path)

    #model = SentenceTransformer("paraphrase-mpnet-base-v2")
    #chunked_embeddings = model.encode(chunked_corpus, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    #np.save(embeddings_path, chunked_embeddings)
    
if __name__ == "__main__":
    main()