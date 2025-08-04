import os
import joblib
import pickle 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm
from joblib import parallel

from sklearn.model_selection import train_test_split, ParameterGrid, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report

from utils.modeling import plot_distribution, plot_correlation, split_and_save, run_eval_model

def main():
    #paths
    dfs = os.path.join("obj", "data")
    plots = os.path.join("obj", "plots")
    model_extras = os.path.join("obj", "model_extras")
    models = os.path.join("obj", "models")
    textfiles = os.path.join("obj", "text_files")

    # DEFINE PARAMS
    # logistic regression
    param_grid_lr = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga'],
        'multi_class': ['multinomial'],
        'class_weight': ['balanced']
    }
    # random forest
    param_grid_rf = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [5, 10, 20, 50],
        'min_samples_split': [5, 10, 20],
        'class_weight': ['balanced'],
        'max_features': [0.6, 0.8, 1],
    }
    # xgboost
    param_grid_xgb = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth' : [5, 10, 20, 50],
        'colsample_bytree': [0.6, 0.8, 1],
        'learning_rate': [2e-3, 2e-2, 2e-1],
        'subsample': [0.6, 0.8, 1.0]
    }

    # DEFINE MODELS
    # logistic regression
    model_lr = LogisticRegression(
        max_iter=1000)
    # random forest
    model_rf = RandomForestClassifier(
        random_state=42)
    # xgboost
    model_xgb = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        eval_metric='mlogloss',
        random_state=42)

    for n in ['l1', 'softmax']: 
        data_path = os.path.join(dfs, f"NEe_{n}_final.pkl")
        with open(data_path, "rb") as f:
            df = pickle.load(f)

        # PLOT CORRELATION BETWEEN PREDICTORS
        X = df.iloc[:, 1:51]
        corr_path = os.path.join(plots, f"topic_corr_{n}.png")
        plot_correlation(X, corr_path)

        # RUN FOR AU AND FANDOM
        for target in ["AU", "Fandom"]:

            # get class distribution
            distr_path = os.path.join(plots, f"{target}_distr.png")
            plot_distribution(df, target, distr_path)

            # select y
            y = df[target]

            # encode class labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            class_labels = label_encoder.classes_
            joblib.dump(class_labels, os.path.join(model_extras, f"class_labels_{target}.pkl"))
        
            # split data and save for future analysis
            X_train, X_test, y_train, y_test = split_and_save(X, y_encoded, 0.2, model_extras, target, n, stratify=True)

            # dummy classifier to set baseline
            dummy_clf = DummyClassifier(strategy="most_frequent")
            dummy_clf.fit(X_train, y_train)
            d_cr = classification_report(y_test, dummy_clf.predict(X_test), output_dict=True)

            # run logistic regression
            lr_best, lr_cr = run_eval_model(
                model_lr, "lr", 
                X_train, X_test, y_train, y_test, 
                param_grid_lr, 
                5, 'f1_macro', -1, 
                class_labels, models, target, n)

            # run random forest
            rf_best, rf_cr = run_eval_model(
                model_rf, "rf", 
                X_train, X_test, y_train, y_test, 
                param_grid_rf, 
                5, 'f1_macro', -1, 
                class_labels, models, target, n)

            # run xgboost
            xgb_best, xgb_cr = run_eval_model(
                model_xgb, "xgb", 
                X_train, X_test, y_train, y_test, 
                param_grid_xgb, 
                5, 'f1_macro', -1, 
                class_labels, models, target, n)
            
            with open(os.path.join(textfiles, f"best_models_{target}_{n}.txt"), 'w') as f:
                f.write(f"Dummy best params: 'most_frequent'\n")
                f.write(str(d_cr) + "\n")
                f.write("\n")
                f.write(f"LR best params: {lr_best}\n")
                f.write(str(lr_cr) + "\n")
                f.write("\n")
                f.write(f"RF best params: {rf_best}\n")
                f.write(str(rf_cr) + "\n")
                f.write("\n")
                f.write(f"XGB best params: {xgb_best}\n")
                f.write(str(xgb_cr) + "\n")

if __name__ == "__main__":
    main()