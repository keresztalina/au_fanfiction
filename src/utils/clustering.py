import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from turftopic import KeyNMF
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_mutual_info_score

def prepare_topic_df(model, 
                     chunked_df: pd.DataFrame, 
                     metadata_df: pd.DataFrame,
                     norm_mode: str = 'l1') -> pd.DataFrame:
    """Prepare a work-level topic dataframe from a chunk-level model.

    Parameters:
        model: Trained topic model with a .document_topic_matrix and .topic_names.
        chunked_df: DataFrame with 'work_id' and 'chunk_id' columns.
        metadata_df: DataFrame with metadata to merge on 'work_id'.
        norm_mode: Either 'l1' for row-sum normalization, or 'softmax' for softmax scaling.

    Returns:
        A DataFrame with topic proportions per work and metadata merged.
    """
    # Check if input is valid
    valid_modes = {'l1', 'softmax'}
    if norm_mode not in valid_modes:
        raise ValueError(f"Invalid norm_mode '{norm_mode}'. Choose from: {valid_modes}")

    # Topic coefficients
    topics = pd.DataFrame(model.document_topic_matrix, columns=model.topic_names)

    # Combine with chunk IDs
    df = pd.concat([chunked_df[['work_id', 'chunk_id']].reset_index(drop=True),
                    topics.reset_index(drop=True)], axis=1)

    # Aggregate over chunks per work
    df = df.drop(columns='chunk_id').groupby('work_id').sum()

    # Normalize per work
    if norm_mode == 'l1':
        df = df.div(df.sum(axis=1), axis=0)
    elif norm_mode == 'softmax':
        df = pd.DataFrame(softmax(df.values, axis=1), 
                          index=df.index, 
                          columns=df.columns)

    # Merge with metadata
    df = df.reset_index().merge(metadata_df, on='work_id')
    return df


def cluster(topic_df: pd.DataFrame, 
            topic_cols: pd.DataFrame, 
            n_components_list: list[int]):
    """Run Gaussian Mixture Model clustering on topic vectors and return annotated dataframe and results."""
    gmm_results = []
    gmm_models = {}

    for n_components in tqdm(n_components_list, desc="Running GMM"):
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(topic_cols)
        cluster_labels = gmm.predict(topic_cols)
        gmm_models[n_components] = gmm

        ami_fandom = adjusted_mutual_info_score(topic_df['Fandom'], cluster_labels)
        ami_au = adjusted_mutual_info_score(topic_df['AU'], cluster_labels)
        bic = gmm.bic(topic_cols)

        gmm_results.append({
            'n_components': n_components,
            'AMI_fandom': ami_fandom,
            'AMI_au': ami_au,
            'BIC': bic
        })

    # best model by BIC
    best_result = min(gmm_results, key=lambda x: x['BIC'])
    best_n = best_result['n_components']
    best_labels = gmm_models[best_n].predict(topic_cols)
    topic_df[f'gmm_{best_n}'] = best_labels

    return topic_df, pd.DataFrame(gmm_results)


def plot_cluster(topic_df: pd.DataFrame, 
                 cluster_col: str, 
                 category_col: str, 
                 output_path: str,
                 type: str):
    """Plot the distribution of a category (e.g., Fandom or AU) across clusters."""
    cluster_counts = topic_df.groupby([cluster_col, category_col]).size().unstack().fillna(0)
    cluster_props = cluster_counts.div(cluster_counts.sum(axis=1), axis=0)

    dominant_props = cluster_props.max(axis=1)

    ax = dominant_props.plot(kind='bar', figsize=(12, 6), color='seagreen')
    plt.title(f'Most popular {category_col} proportion in each GMM cluster ({type})')
    plt.xlabel('GMM Cluster')
    plt.ylabel('Proportion')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def run_clustering_pipeline(model, 
                            chunked_df: pd.DataFrame, 
                            metadata_df: pd.DataFrame,
                            norm_mode: str,
                            cluster_counts: list[int], 
                            type: str,
                            out: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full clustering pipeline from model and metadata to results."""
    df = prepare_topic_df(model, chunked_df, metadata_df, norm_mode=norm_mode)
    topic_cols = df.iloc[:, 1:51]  # assumes first col is work_id, next 50 are topic scores
    
    entropies = df.iloc[:, 1:51].apply(entropy, axis=1)
    ent_summary = entropies.describe()

    df, gmm_output = cluster(df, topic_cols, cluster_counts)

    for category in ['AU', 'Fandom']:
        cluster_col = df.columns[-1]  # last column is the best cluster label
        output_path = os.path.join(out, f"{type}_{category}_{norm_mode}_composition.png")
        plot_cluster(df, cluster_col, category, output_path, type)

    return df, gmm_output, ent_summary