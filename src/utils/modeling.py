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
    """
    Context manager to integrate `tqdm` progress bars with `joblib.Parallel`.

    This wrapper temporarily overrides joblib's internal batch completion callback
    to update a tqdm progress bar during parallel processing.

    Parameters:
    -----------
    tqdm_object : tqdm.tqdm
        A tqdm progress bar instance to update as batches complete.
    """

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
    """
    Plots and saves a count distribution of values in a specified column of a DataFrame.

    This function creates a bar plot showing the frequency of each unique value
    in the given column, annotates each bar with its count, and saves the plot to a file.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.
    column_name : str
        The name of the column in `df` whose value distribution will be plotted.
    output_file : str
        File path to save the resulting plot (e.g., "output.png" or "plots/distribution.pdf").
    """

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
    
    for p in plot.patches:
        plot.annotate(format(p.get_height(), '.0f'), 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='center', 
                      xytext=(0, 9), 
                      textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(output_file)

def plot_correlation(predictors, output_file):
    """
    Plots and saves a heatmap of the correlation matrix for a given set of predictors.

    This function computes the Pearson correlation coefficients between all pairs of 
    columns in the input DataFrame and visualizes them as a heatmap.

    Parameters:
    -----------
    predictors : pandas.DataFrame
        A DataFrame containing only numeric predictor variables.
    output_file : str
        Path to save the generated heatmap image (e.g., "correlation_matrix.png").
    """

    plt.figure(figsize=(12, 10))
    corr = predictors.corr()
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True, cbar_kws={'shrink': .5})
    plt.title('Correlation matrix of predictors')
    plt.tight_layout()
    plt.savefig(output_file)

def split_and_save(X, y_encoded, test_size, output_folder, y_colname, norm, stratify=False):
    """
    Splits data into training and test sets and saves the test sets to disk.

    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix to be split.
    y_encoded : array-like
        Encoded target labels corresponding to X.
    test_size : float
        Proportion of the dataset to include in the test split (e.g., 0.2 for 20%).
    output_folder : str
        Path to the folder where the test split files will be saved.
    y_colname : str
        Name of the target variable (used in file naming).
    norm : str
        Normalization label to include in filenames (e.g., "zscore", "minmax").
    stratify : bool, optional (default=False)
        Whether to stratify the split by the class distribution in y_encoded.

    Returns:
    --------
    X_train : array-like
        Training set features.
    X_test : array-like
        Test set features.
    y_train : array-like
        Training set labels.
    y_test : array-like
        Test set labels.

    Notes:
    ------
    - Here only the test splits (`X_test`, `y_test`) are saved to disk as .pkl files.
    """
    
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
    joblib.dump(X_test, os.path.join(output_folder, f"X_test_{y_colname}_{norm}.pkl"))
    #joblib.dump(y_train, os.path.join(output_folder, f"y_train_{y_colname}.pkl"))
    joblib.dump(y_test, os.path.join(output_folder, f"y_test_{y_colname}_{norm}.pkl"))
    
    return X_train, X_test, y_train, y_test

def run_eval_model(model, model_name, X_train, X_test, y_train, y_test, param_grid, cv, scoring, n_jobs, classes, models_folder, y_colname, norm):
    """
    Performs model training and evaluation using GridSearchCV with progress tracking,
    and saves the best model to disk.

    Parameters:
    -----------
    model : estimator object
        The base model to be optimized using GridSearchCV.
    model_name : str
        Name of the model (used for progress bar and filename).
    X_train : array-like
        Training feature set.
    X_test : array-like
        Test feature set.
    y_train : array-like
        Training labels.
    y_test : array-like
        Test labels.
    param_grid : dict
        Dictionary with parameters names as keys and lists of parameter settings to try.
    cv : int
        Number of cross-validation folds.
    scoring : str or callable
        Scoring strategy to evaluate the performance of the cross-validated model.
    n_jobs : int
        Number of jobs to run in parallel (used in GridSearchCV).
    classes : list of str
        Class labels (used in classification report).
    models_folder : str
        Directory to save the trained model.
    y_colname : str
        Name of the target variable (used for filename generation).
    norm : str
        Normalization label (used for display and filenames).

    Returns:
    --------
    best_params : dict
        Best parameter combination found by GridSearchCV.
    classification_rep : dict
        Classification report as a dictionary, containing precision, recall, f1-score, etc.

    Notes:
    ------
    - Uses `TqdmJoblib` context manager to integrate tqdm with joblib for progress updates.
    """

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