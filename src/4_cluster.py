import os
import joblib
from utils.clustering import run_clustering_pipeline

def main():

    # paths
    chunked_df = joblib.load(os.path.join("obj", "data", "df_chunked.pkl"))
    metadata_df = joblib.load(os.path.join("obj", "data", "prepped_data.pkl"))
    plot_path = os.path.join("obj", "plots")

    for i in ['NEi', 'NEe']:

        print(f"Loading model for {i}")
        m = joblib.load(os.path.join("obj", "topic_models", f"{i}_50.pkl"))

        if i == 'NEe':
            print("Renaming topics...")
            m.rename_topics({
                0: "Questions",
                1: "Boyfriend bonding",
                2: "Everyday actions",
                3: "Soulmates",
                4: "Soldier bonding",
                5: "Friendly conversation",
                6: "Information exchange",
                7: "Sweet with hyung",
                8: "Thinking",
                9: "Heroes & villains",
                10: "Hesitation",
                11: "Neutral observation",
                12: "Affection",
                13: "Walking out the door",
                14: "Angry staring",
                15: "Agreement",
                16: "Romance",
                17: "Siblings",
                18: "Surprised by a stranger",
                19: "Vampires & werewolves",
                20: "Laughter",
                21: "Royalty",
                22: "Surprise",
                23: "Magic",
                24: "Feelings & emotions",
                25: "Watching",
                26: "Endings",
                27: "Communicating realizations",
                28: "Smiling",
                29: "What happened?",
                30: "Hero's farewell",
                31: "Parents & children",
                32: "Worry & concern",
                33: "Closeness",
                34: "Silent aggression",
                35: "Pensiveness",
                36: "Attractive body",
                37: "Coffee meetup",
                38: "Love & care",
                39: "Careful touch",
                40: "Standing close",
                41: "Entering a room",
                42: "Hurt & tears",
                43: "Boyband",
                44: "Frowning & muttering",
                45: "Waiting",
                46: "Kissing",
                47: "Sex",
                48: "Smirking & teasing",
                49: "Responses"})

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