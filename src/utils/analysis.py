import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_top_shap_class(shap_values, class_name, class_names, feature_names, y_true, y_pred, norm, output_path, mode="all preds", top_n=10):
    """
    Plots a horizontal bar chart of the top SHAP features for a specific class.

    Parameters:
    - shap_values: numpy array of SHAP values with shape (n_samples, n_features, n_classes)
    - class_name: string, name of the class to plot SHAP values for
    - class_names: list of class names matching the class indices in y_true/y_pred
    - feature_names: list of feature names corresponding to the features used in the model
    - y_true: array-like of true class indices
    - y_pred: array-like of predicted class indices
    - norm: string or label indicating the normalization method used (included in the plot title)
    - output_path: string, path to save the output plot image
    - mode: string, determines which data points to include in the analysis:
        - "all preds": use all instances
        - "entire class": use instances predicted as the class
        - "correct preds": use instances correctly predicted as the class
    - top_n: integer, number of top features to display (ranked by mean absolute SHAP value)

    The plot shows the top `top_n` features contributing to the selected class, with 
    green bars indicating a positive mean SHAP contribution and red bars indicating 
    a negative contribution. Features are sorted by their average absolute SHAP value.
    """
    try:
        i = list(class_names).index(class_name)
    except ValueError:
        raise ValueError(f"Class '{class_name}' not found in class_names.")

    valid_modes = {"all preds", "entire class", "correct preds"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: '{mode}'. Choose from {valid_modes}.")

    if mode == "all preds":
        y_mask = np.ones(len(y_true), dtype=bool)
    elif mode == "entire class":
        y_mask = y_pred == i
    elif mode == "correct preds":
        y_mask = (y_true == i) & (y_pred == i)
    
    class_shap = shap_values[y_mask, :, i]
    mean_abs = np.abs(class_shap).mean(axis=0)
    mean_signed = class_shap.mean(axis=0)

    top_idx = np.argsort(mean_abs)[-top_n:]
    top_features = np.array(feature_names)[top_idx]
    top_signed_vals = mean_signed[top_idx]
    colors = ['seagreen' if v > 0 else 'crimson' for v in top_signed_vals]

    plt.figure(figsize=(10, 6))
    plt.barh(top_features, top_signed_vals, color=colors)
    plt.title(f"Top {top_n} topics by absolute value ({class_name}, {norm}-normalized)")
    plt.xlabel("Mean SHAP Value")
    plt.axvline(0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_shap_feature(
    shap_values, 
    X, y_true, y_pred, 
    class_name, class_names, 
    feature_name, feature_names, norm, output_path):

    """
    Plots a scatter plot of SHAP values for a specific feature and class, colored by whether predictions were correct.

    Parameters:
    - shap_values: numpy array of SHAP values with shape (n_samples, n_features, n_classes)
    - X: pandas DataFrame of input features used for prediction
    - y_true: array-like of true class indices
    - y_pred: array-like of predicted class indices
    - class_name: string, name of the class to plot SHAP values for
    - class_names: list of class names, matching indices in y_true/y_pred
    - feature_name: string, name of the feature (e.g., topic) to plot
    - feature_names: list of feature names matching columns in X
    - norm: string or value describing the SHAP normalization used (for plot title)
    - output_path: string, file path to save the resulting plot (e.g., "shap_plot.png")

    The plot shows SHAP values for all instances of the selected class, 
    with color indicating whether the prediction was correct (green) or incorrect (red).
    """
    
    try:
        i = list(class_names).index(class_name)
    except ValueError:
        raise ValueError(f"Class '{class_name}' not found in class_names.")
    
    # Get feature index
    try:
        f = list(feature_names).index(feature_name)
    except ValueError:
        raise ValueError(f"Feature '{feature_name}' not found in feature_names.")

    y_mask = y_true == i
    feature_values = X.iloc[y_mask, f]
    shap_vals_for_class = shap_values[y_mask, f, i]
    correct = y_true[y_mask] == y_pred[y_mask]
    colors = ['seagreen' if c else 'crimson' for c in correct]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(feature_values, shap_vals_for_class, c=colors, alpha=0.4, s=20, linewidths=0)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel(f"Proportion of topic {feature_name}")
    plt.ylabel(f"SHAP values for class '{class_name}'")
    plt.title(f"SHAP values for '{class_name}' plotted against most predictive topic (norm: {norm})")
    legend_handles = [
        Patch(color='seagreen', label='Correctly classified'),
        Patch(color='crimson', label='Misclassified')
    ]
    plt.legend(handles=legend_handles, loc='best')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_misclassified_confidence_histograms(y_true, y_pred, confidences, class_labels, title, output_path, bins=10):
    """
    For each class, plot a histogram of confidence values for datapoints misclassified *as* that class.

    Parameters:
    - y_true: array of true class indices
    - y_pred: array of predicted class indices
    - confidences: array of confidence scores for the predictions (same shape)
    - class_labels: list or array of class names, one per class index
    - bins: number of bins in the histogram (default: 10)
    """
    n_classes = len(class_labels)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4), sharey=True)

    for i in range(n_classes):
        ax = axes[i]
        # Identify datapoints predicted as class i, but actually wrong
        mask = (y_pred == i) & (y_pred != y_true)
        wrong_confidences = confidences[mask]

        ax.hist(wrong_confidences, bins=bins, range=(0, 1), color='salmon', edgecolor='black')
        ax.set_title(f'Misclassified as "{class_labels[i]}"\n(n={len(wrong_confidences)})')
        ax.set_xlabel("Confidence")
        if i == 0:
            ax.set_ylabel("Count")
        ax.set_xlim(0, 1)

    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.85)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_normalized_confusion_matrix(y_true, y_pred, title, output_path, class_names=None, highlight_top=1):
    """
    Plots a normalized confusion matrix with optional highlighting of the most confused class pairs.

    Parameters:
    - y_true: array-like of shape (n_samples,), true class labels.
    - y_pred: array-like of shape (n_samples,), predicted class labels.
    - class_names: list of strings, names of classes. If None, integer labels are used.
    - highlight_top: number of most confused off-diagonal cells to highlight (default is 1).
    """
    # Define labels
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(labels)

    if class_names is None:
        class_names = [str(lbl) for lbl in labels]

    # Compute normalized confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)

    # Mask diagonal and find top off-diagonal confused pairs
    off_diag_mask = ~np.eye(n_classes, dtype=bool)
    flat_values = cm_df.values[off_diag_mask]
    top_threshold = np.sort(flat_values)[-highlight_top:] if highlight_top > 0 else []

    # Create annotated labels
    annot = cm_df.round(2).astype(str)
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm_df.iloc[i, j] in top_threshold:
                annot.iloc[i, j] += " ★"

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_df,
        annot=annot,
        fmt='',
        cmap='Blues',
        cbar=True,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_topic_piechart(df, target, norm, output_path, topic_start=1, topic_end=51, threshold=0.015):
    """
    Plots a pie chart of topic proportions for each group in a DataFrame.

    Parameters:
    - df: pandas DataFrame with topic columns and a group column.
    - target: column name to group by (e.g. 'AU', 'Main fandom').
    - topic_start: index of the first topic column (default: 1).
    - topic_end: index after the last topic column (default: 51).
    - threshold: minimum proportion for a topic to be included in the chart (default: 0.015).
    """
    topic_cols = df.columns[topic_start:topic_end]

    # Create color list (tab20 + tab20b + tab20c)
    colors = list(cm.get_cmap('tab20').colors) + \
             list(cm.get_cmap('tab20b').colors) + \
             list(cm.get_cmap('tab20c').colors)
    colors = colors[:50]  # Ensure only 50 colors

    for group_value, group in df.groupby(target):
        topic_sums = group[topic_cols].sum()
        topic_props = topic_sums / topic_sums.sum()

        mask = topic_props >= threshold
        top_props = topic_props[mask].sort_values(ascending=False)
        other_props_sum = topic_props[~mask].sum()

        if other_props_sum > 0:
            top_props["Other"] = other_props_sum

        pie_colors = colors[:len(top_props) - 1]
        if "Other" in top_props.index:
            pie_colors.append("lightgrey")

        plt.figure(figsize=(14, 14))
        plt.pie(
            top_props,
            labels=top_props.index,
            autopct='%1.1f%%',
            colors=pie_colors,
            startangle=140
        )
        plt.title(f"Topic Distribution for {target}: {group_value} ({norm})", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"piechart_{target}_{group_value}_{norm}.png"), bbox_inches='tight')
        plt.close()