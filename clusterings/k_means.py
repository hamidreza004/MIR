from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def get_advanced_results(tf_idf, w2v, tags, n_cluster_tf, n_cluster_w2v):
    random_tf = get_best_randomeness(tf_idf, tags, n_cluster_tf)
    random_w2v = get_best_randomeness(w2v, tags, n_cluster_w2v)
    Purity_tf, AMI_tf, NMI_tf, ARI_tf, inertia_tf = evaluate(tf_idf, tags, n_clusters=n_cluster_tf,
                                                             random_state=random_tf)
    Purity_w2v, AMI_w2v, NMI_w2v, ARI_w2v, inertia_w2v = evaluate(w2v, tags, n_clusters=n_cluster_w2v,
                                                                  random_state=random_w2v)
    df = pd.DataFrame({'Purity': [Purity_tf, Purity_w2v], 'Adjusted Mutual Info': [AMI_tf, AMI_w2v],
                       'Normalized Mutual Info': [NMI_tf, NMI_w2v], 'Adjusted Rand Index': [ARI_tf, ARI_w2v],
                       'Inertia': [inertia_tf, inertia_w2v]},
                      index=['tf_idf', 'w2v'])
    return df, random_tf, random_w2v


def get_evaluation_dataframe(tf_idf, w2v, tags, n_cluster_tf, n_cluster_w2v):
    Purity_tf, AMI_tf, NMI_tf, ARI_tf, inertia_tf = evaluate(tf_idf, tags, n_clusters=n_cluster_tf)
    Purity_w2v, AMI_w2v, NMI_w2v, ARI_w2v, inertia_w2v = evaluate(w2v, tags, n_clusters=n_cluster_w2v)
    return (pd.DataFrame({'Purity': [Purity_tf, Purity_w2v], 'Adjusted Mutual Info': [AMI_tf, AMI_w2v],
                          'Normalized Mutual Info': [NMI_tf, NMI_w2v], 'Adjusted Rand Index': [ARI_tf, ARI_w2v],
                          'Inertia': [inertia_tf, inertia_w2v]},
                         index=['tf_idf', 'w2v']))


def get_cost(vector, random_state=12, n_clusters=5):
    X = np.array(vector)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300, tol=1e-4).fit(X)
    return kmeans.inertia_


def evaluate(vector, tags, n_clusters=5, random_state=12):
    X = np.array(vector)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300, tol=1e-4).fit(X)
    labels_true = tags
    labels_pred = kmeans.labels_

    Purity = round(purity_score(labels_true, labels_pred), 6)
    AMI = round((metrics.cluster.adjusted_mutual_info_score(labels_true, labels_pred)), 6)
    NMI = round((metrics.cluster.normalized_mutual_info_score(labels_true, labels_pred)), 6)
    ARI = round(metrics.cluster.adjusted_rand_score(labels_true, labels_pred), 6)

    return Purity, AMI, NMI, ARI, kmeans.inertia_


def evaluate_AMI(vector, tags, n_clusters=5, random_state=12):
    X = np.array(vector)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300, tol=1e-4).fit(X)
    labels_true = tags
    labels_pred = kmeans.labels_

    AMI = round((metrics.cluster.adjusted_mutual_info_score(labels_true, labels_pred)), 6)
    return AMI


def get_best_randomeness(vector, tags, n_clusters):
    best_score = 0
    best_random = 0
    for i in range(3):
        score = evaluate_AMI(vector, tags, n_clusters, i * 53)
        if score > best_score:
            best_score = score
            best_random = i * 53
    return best_random


def purity_score(labels_true, labels_pred):
    matrix = contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)


def get_labels(vector, n_clusters=5, random_state=12):
    X = np.array(vector)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300, tol=1e-4).fit(X)
    return kmeans.labels_


def PCA2_plot(vectors, n_clusters, random_state, labels_true, title):
    labels_pred = get_labels(vectors, n_clusters=n_clusters, random_state=random_state)
    pca_reduction = PCA(2, random_state=12).fit_transform(vectors)
    fig, axes = plt.subplots(1, 2, figsize=(28, 8))
    axes[0].scatter(pca_reduction[:, 0], pca_reduction[:, 1], c=labels_pred, cmap='viridis')
    axes[0].set_title('Prediction')
    axes[1].scatter(pca_reduction[:, 0], pca_reduction[:, 1], c=labels_true, cmap='viridis')
    axes[1].set_title('Ground Truth')
    fig.suptitle(title)
    plt.grid()
    plt.show()
