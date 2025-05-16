import pickle
from clustering import run_clustering_pipeline

def main():

    with open("df_chunked.pkl", "rb") as f:
        chunked_df = pickle.load(f)
    with open("prepped_data.pkl", "rb") as f:
        metadata_df = pickle.load(f)

    for i in ['NEi', 'NEe']:
        with open(f"{i}_50_15.pkl", "rb") as f:
            m = pickle.load(f)
        df, gmm_results = run_clustering_pipeline(
            m, chunked_df, metadata_df,
            cluster_counts=[10, 20, 30], type=i)

        df.to_pickle(f"{i}_final.pkl")
        gmm_results.to_csv(f"gmm_output_{i}.csv", index=False)