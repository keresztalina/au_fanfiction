import os
import joblib
from utils.clustering import run_clustering_pipeline

def main():

    chunked_df = joblib.load(os.path.join("obj", "data", "df_chunked.pkl"))
    metadata_df = joblib.load(os.path.join("obj", "data", "prepped_data.pkl"))
    plot_path = os.path.join("obj", "plots")

    for i in ['NEi', 'NEe']:
        m = joblib.load(os.path.join("obj", "topic_models", f"{i}_50_15.pkl"))

        for j in ['l1', 'softmax']:
            print(f"Running clustering for type {i} with normalization {j}")

            df, gmm_results, entropies = run_clustering_pipeline(
                m, chunked_df, metadata_df, j,
                cluster_counts=[10, 20, 50, 100], type=i, out=plot_path)

            df.to_pickle(os.path.join("obj", "data", f"{i}_{j}_final.pkl"))
            gmm_results.to_csv(os.path.join("obj", "text_files", f"gmm_output_{i}_{j}.csv"), index = False)
            entropies.to_csv(os.path.join("obj", "text_files", f"entropies_{i}_{j}.csv"), header=True)


if __name__ == "__main__":
    main()