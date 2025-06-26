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

# progress bar wrapper for joblib
class TqdmJoblib:
    def __init__(self, tqdm_object):
        self.tqdm = tqdm_object

    def __enter__(self):
        self.original_callback = joblib.parallel.BatchCompletionCallBack

        def new_callback(*args, **kwargs):
            self.tqdm.update()
            return self.original_callback(*args, **kwargs)

        joblib.parallel.BatchCompletionCallBack = new_callback

    def __exit__(self, exc_type, exc_val, exc_tb):
        joblib.parallel.BatchCompletionCallBack = self.original_callback

def plot_distribution(df, column_name, output_file):
    plt.figure(figsize=(8, 5))
    plot = sns.countplot(
        data=df, 
        x=column_name, 
        order=df[column_name].value_counts().index, 
        palette='viridis')
    plt.title(f'Distribution of {column_name} labels')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Annotate counts on the bars
    for p in plot.patches:
        plot.annotate(format(p.get_height(), '.0f'), 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='center', 
                      xytext=(0, 9), 
                      textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(output_file)

def plot_correlation(predictors, output_file):
    plt.figure(figsize=(12, 10))
    corr = predictors.corr()
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True, cbar_kws={'shrink': .5})
    plt.title('Correlation matrix of predictors')
    plt.tight_layout()
    plt.savefig(output_file)

def split_and_save(X, y_encoded, test_size, output_folder, y_colname, stratify=False):
    
    if stratify:
        strat = y_encoded
    else:
        strat = None

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y_encoded, 
        test_size=test_size, 
        random_state=42, 
        stratify=strat
    )

    # dump
    #joblib.dump(X_train, os.path.join(output_folder, f"X_train_{y_colname}.pkl"))
    joblib.dump(X_test, os.path.join(output_folder, f"X_test_{y_colname}.pkl"))
    #joblib.dump(y_train, os.path.join(output_folder, f"y_train_{y_colname}.pkl"))
    joblib.dump(y_test, os.path.join(output_folder, f"y_test_{y_colname}.pkl"))
    
    return X_train, X_test, y_train, y_test

def run_eval_model(model, model_name, X_train, X_test, y_train, y_test, param_grid, cv, scoring, n_jobs, classes, models_folder, y_colname, norm):

    # prep for progress bar
    n_total = len(ParameterGrid(param_grid)) * cv

    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    # run gridsearch
    with tqdm(total=n_total, desc=f"GridSearchCV for {norm}-normalized {y_colname} w/ model {model_name}") as progress_bar:
        with TqdmJoblib(progress_bar):
            grid.fit(X_train, y_train)

    # classification report
    best_params = grid.best_params_
    classification_rep = classification_report(
        y_test, 
        grid.predict(X_test), 
        target_names=classes, 
        output_dict=True)
    
    # save model
    joblib.dump(grid.best_estimator_, os.path.join(models_folder, f"{model_name}_{y_colname}_{norm}.pkl"))

    return best_params, classification_rep