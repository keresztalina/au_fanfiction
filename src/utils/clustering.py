import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from turftopic import KeyNMF
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_mutual_info_score

def prepare_topic_df(model, 
                     chunked_df: pd.DataFrame, 
                     metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a work-level topic dataframe from a chunk-level model."""
    # Topic coefficients
    topics = pd.DataFrame(model.document_topic_matrix, columns=model.topic_names)

    # Combine with chunk IDs
    df = pd.concat([chunked_df[['work_id', 'chunk_id']].reset_index(drop=True),
                    topics.reset_index(drop=True)], axis=1)

    # Aggregate over chunks per work
    df = df.drop(columns='chunk_id').groupby('work_id').sum()
    df = df.div(df.sum(axis=1), axis=0).reset_index()

    # Merge with metadata
    df = df.merge(metadata_df, on='work_id')
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
                 output_path: str):
    """Plot the distribution of a category (e.g., Fandom or AU) across clusters."""
    cluster_counts = topic_df.groupby([cluster_col, category_col]).size().unstack().fillna(0)
    cluster_props = cluster_counts.div(cluster_counts.sum(axis=1), axis=0)

    ax = cluster_props.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set1')
    plt.title(f'{category_col} composition of each GMM cluster')
    plt.xlabel('GMM Cluster')
    plt.ylabel('Proportion')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def run_clustering_pipeline(model, 
                            chunked_df: pd.DataFrame, 
                            metadata_df: pd.DataFrame,
                            cluster_counts: list[int], 
                            type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full clustering pipeline from model and metadata to results."""
    df = prepare_topic_df(model, chunked_df, metadata_df)
    topic_cols = df.iloc[:, 1:51]  # assumes first col is work_id, next 50 are topic scores

    df, gmm_output = cluster(df, topic_cols, cluster_counts)

    for category in ['Fandom', 'AU']:
        cluster_col = df.columns[-1]  # last column is the best cluster label
        output_path = f"{type}_{category}_composition.png"
        plot_cluster(df, cluster_col, category, output_path)

    return df, gmm_output