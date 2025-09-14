import os
import joblib
import shap
from utils.analysis import plot_top_shap_class, plot_shap_feature, plot_normalized_confusion_matrix, plot_misclassified_confidence_histograms

def main():

    # paths
    model_extras = os.path.join("obj", "model_extras")
    models = os.path.join("obj", "models")
    plots = os.path.join("obj", "plots")
    df = joblib.load(os.path.join("obj", "data", "NEe_l1_final.pkl"))

    # define pairs for future plotting
    au_pairs = {
        "Coffeeshop": "Coffee meetup",
        "Royalty": "Royalty",
        "Soulmates": "Soulmates",
        "Vampire": "Vampires & werewolves",
    }

    fandom_pairs = {
        "Bnha": "Heroes & villains",
        "Bts": "Boyfriend bonding",
        "Hp": "Magic",
        "Mcu": "Soldier bonding"
    }

    for norm in ["l1", "softmax"]:
        for target in ["AU", "Fandom"]:

            # load objects
            model = joblib.load(os.path.join(models, f"xgb_{target}_{norm}.pkl"))
            X_test = joblib.load(os.path.join(model_extras, f"X_test_{target}_{norm}.pkl"))
            y_test = joblib.load(os.path.join(model_extras, f"y_test_{target}_{norm}.pkl"))
            class_labels = joblib.load(os.path.join(model_extras, f"class_labels_{target}.pkl"))

            # run shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            # make predictions
            preds = model.predict(X_test)
            feature_names = X_test.columns
            
            # extract probabilities and confidences used for prediction
            y_probs = model.predict_proba(X_test)
            confidence = y_probs.max(axis=1)

            # plot most predictive features
            for c in class_labels:
                    plot_top_shap_class(
                        shap_values, 
                        c, class_labels, 
                        feature_names, 
                        y_test, preds, 
                        norm,
                        os.path.join(plots, f"top_shap_{c}_{norm}.png"),
                        mode="correct preds", 
                        top_n=10)

            # plot relationship between shap values and most predictive topics
            if target == "AU":
                pairs = au_pairs
            elif target == "Fandom":
                pairs = fandom_pairs
            else:
                print("Double check target assignments...")

            for class_label, feature in pairs.items():
                plot_shap_feature(
                    shap_values,
                    X_test, y_test, preds,
                    class_label, class_labels,
                    feature, feature_names,
                    norm,
                    os.path.join(plots, f"{class_label}_{feature}_{norm}.png"),
                )

            # confusion matrices
            plot_normalized_confusion_matrix(
                y_test, preds, 
                f"Normalized Confusion Matrix (★ = Most Confused Pair(s)) ({target}, {norm})",
                os.path.join(plots, f"confusion_{target}_{norm}.png"),
                class_names=class_labels)

            # investigate misclassifications
            plot_misclassified_confidence_histograms(
                y_test, preds, confidence, class_labels, 
                f"Misclassification confidences per class ({target}, {norm})",
                os.path.join(plots, f"misclassified_{target}_{norm}.png"),
                bins=10)
                
if __name__ == "__main__":
    main()