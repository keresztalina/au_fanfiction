import os
import joblib
from utils.analysis import plot_topic_piechart

def main():
    for norm in ["l1", "softmax"]:
        df = joblib.load(os.path.join("obj", "data", f"NEe_{norm}_final.pkl"))
        for target in ["Fandom", "AU"]:
            plot_topic_piechart(
                df, target, norm,
                os.path.join("obj", "plots"))

if __name__ == "__main__":
    main()