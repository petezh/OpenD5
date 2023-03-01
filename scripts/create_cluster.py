"""
Code for creating clusters from embeddings.

Author: Peter Zhang, Ruiqi Zhong
"""

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
import glob
import json
import os
from typing import Tuple

import numpy as np
import sklearn.cluster
import sklearn.decomposition
import sklearn.mixture
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import tqdm

def load_data(embed_dir: str, subset_size: int) -> Tuple[np.array, np.array]:
    """
    Accepts a directory with embeddings and a sample size to use.
    """

    f_prefixes = sorted([f.split('.')[0] for f in os.listdir(
        embed_dir) if f.endswith('.npy')], key=lambda x: int(x))
    all_embeddings, all_texts = [], []
    for f in tqdm.tqdm(f_prefixes):
        new_embeddings = np.load(os.path.join(embed_dir, f + '.npy'))
        if len(new_embeddings.shape) == 2:
            all_embeddings.extend(new_embeddings)
            all_texts.extend(
                json.load(open(os.path.join(embed_dir, f + '.json'))))
            if len(all_embeddings) >= subset_size:
                break

    all_embeddings = np.array(all_embeddings)[:subset_size]
    all_texts = all_texts[:subset_size]

    return all_embeddings, all_texts


def make_clusters(
    all_embeddings,
    first_pc,
    last_pc,
    cluster_method,
    k
):

    # loading the embeddings and texts

    print(f'finished loading {len(all_embeddings)} embeddings')

    # first run PCA
    pca = sklearn.decomposition.PCA(n_components=1+last_pc)

    # fit the PCA model to the embeddings
    all_embs = pca.fit_transform(all_embeddings)
    all_embs = all_embs[:, first_pc:last_pc+1]
    print('finished PCA')

    # GMM clustering
    # defining the clustering model
    if cluster_method == 'gmm':
        cluster = sklearn.mixture.GaussianMixture(
            n_components=k, covariance_type='full')
    elif cluster_method == 'kmeans':
        cluster = KMeans(n_clusters=k)

    cluster.fit(all_embs)
    if cluster_method == 'gmm':
        centers = cluster.means_
    elif cluster_method == 'kmeans':
        centers = cluster.cluster_centers_

    print('finished clustering')
    cluster_idxes = cluster.predict(all_embs)

    print('finished predicting probabilities')
    center_pairwise_distances = euclidean_distances(centers, centers)

    return cluster_idxes, center_pairwise_distances


def save_results(save_dir, cluster_idxes, all_texts, center_pairwise_distances):
    """
    Save the results of the clustering.
    """
    
    # saving the results
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    clusters = defaultdict(list)
    for cluster, text in zip(cluster_idxes, all_texts):
        clusters[int(cluster)].append(text)
    json.dump(clusters, open(os.path.join(save_dir, 'clusters.json'), 'w'))
    l2_distances = dict(
        enumerate(map(list, center_pairwise_distances.astype(float))))
    json.dump(l2_distances, open(os.path.join(
        save_dir, 'l2_distance.json'), 'w'))


def main():

    parser = ArgumentParser()
    parser.add_argument('--make_all', action='store_true')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--first_pc', type=int, default=1)
    parser.add_argument('--last_pc', type=int, default=30)
    parser.add_argument('--subset_size', type=int, default=100000)
    parser.add_argument('--sqrt_size', action='store_true')
    parser.add_argument('--k', type=int, default=128)
    parser.add_argument('--cluster_method', type=str, default='kmeans')

    args = parser.parse_args()
    make_all = args.make_all
    dataset = args.dataset
    first_pc = args.first_pc
    last_pc = args.last_pc
    subset_size = args.subset_size
    sqrt_size = args.sqrt_size
    k = args.k
    cluster_method = args.cluster_method

    if make_all:
        datasets = glob.glob('results/*_embeddings')
    else:
        datasets = [dataset]

    for dataset in datasets:

        embed_dir = f'results/{dataset}_embeddings'

        all_embeddings, all_texts = load_data(embed_dir, subset_size)

        if sqrt_size:
            k = int(np.sqrt(len(all_embeddings)) / 2)
            print(f'using sqrt size for dataset {dataset}, k={k}')

        cluster_idxes, center_pairwise_distances = make_clusters(
            all_embeddings,
            first_pc,
            last_pc,
            cluster_method,
            k
        )

        time = datetime.now().strftime("%Y%d%m_%H%M%S")

        if sqrt_size:
            save_dir = f'results/{dataset}_{time}_clusters_sqrtsize'
        else:
            save_dir = f'results/{dataset}_{time}_clusters_{k}'

        save_results(
            save_dir,
            cluster_idxes,
            all_texts,
            center_pairwise_distances,
        )

if __name__ == '__main__':
    main()