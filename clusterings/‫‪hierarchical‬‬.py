import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix


def get_result_df(tf_idf, w2v, tags):
    vectors = ['tf_idf', 'w2v']
    n_clusters = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    linkages = ['ward', 'complete', 'average', 'single ']
    affinitys = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']

    result = []
    for vector in vectors:
        for n_cluster in n_clusters:
            for linkage in linkages:
                for affinity in affinitys:
                    if linkage == 'ward' and affinity != 'euclidiean':
                        continue
                    if vector == 'tf_idf':
                        score_AMI = get_AMI(tf_idf, tags, n_cluster, linkage, affinity)
                        score_Purity, score_NMI, score_ARI = get_others(tf_idf, tags, n_cluster, linkage,
                                                                        affinity)
                    elif vector == 'w2v':
                        score_AMI = get_AMI(w2v, tags, n_cluster, linkage, affinity)
                        score_Purity, score_NMI, score_ARI = get_others(w2v, tags, n_cluster, linkage,
                                                                        affinity)
                    result.append([vector, n_cluster, linkage, affinity, score_AMI, 'AMI'])
                    result.append([vector, n_cluster, linkage, affinity, score_Purity, 'Purity'])
                    result.append([vector, n_cluster, linkage, affinity, score_NMI, 'NMI'])
                    result.append([vector, n_cluster, linkage, affinity, score_ARI, 'ARI'])
                    print(len(result) / 4)

    return pd.DataFrame(data=result, columns=['vector', 'n_clusters', 'linkage', 'affinity', 'score', 'metric'])


def get_AMI(vector, tags, n_clusters, linkage, affinity, random_state=12):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity,
                                         random_state=random_state).fit(vector)
    labels_true = tags
    labels_pred = clustering.labels_
    AMI = round((metrics.cluster.adjusted_mutual_info_score(labels_true, labels_pred)), 6)
    return AMI


def get_others(vector, tags, n_clusters, linkage, affinity, random_state=12):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity,
                                         random_state=random_state).fit(vector)
    labels_true = tags
    labels_pred = clustering.labels_
    Purity = round(purity_score(labels_true, labels_pred), 6)
    NMI = round((metrics.cluster.normalized_mutual_info_score(labels_true, labels_pred)), 6)
    ARI = round(metrics.cluster.adjusted_rand_score(labels_true, labels_pred), 6)
    return Purity, NMI, ARI


def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)


def purity_score(labels_true, labels_pred):
    matrix = contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)

# plt.title('Hierarchical Clustering Dendrogram')
# # plot the top three levels of the dendrogram
# plot_dendrogram(clustering, truncate_mode='level', p=3)
# plt.xlabel("Number of points in node (or index of point if no parenthesis).")
# plt.show()
