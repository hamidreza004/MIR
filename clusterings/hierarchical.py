import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix
from sklearn.decomposition import PCA


def get_result_df(tf_idf, w2v, tags):
    vectors = ['tf_idf', 'w2v']
    n_clusters = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    linkages = ['ward', 'complete', 'average', 'single']
    affinitys = ['euclidean', 'l1', 'l2', 'manhattan']

    result = []
    for vector in vectors:
        for n_cluster in n_clusters:
            for linkage in linkages:
                for affinity in affinitys:
                    if linkage == 'ward' and affinity != 'euclidean':
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

    return pd.DataFrame(data=result, columns=['vector', 'n_clusters', 'linkage', 'affinity', 'score', 'metric'])


def get_evaluation_dataframe(tf_idf, tf_param, w2v, w2v_param, tags):
    Purity_tf, AMI_tf, NMI_tf, ARI_tf, labels_pred_tf = evaluate(tf_idf, tags, tf_param['n_clusters'],
                                                                 tf_param['linkage'], tf_param['affinity'])
    Purity_w2v, AMI_w2v, NMI_w2v, ARI_w2v, labels_pred_w2v = evaluate(w2v, tags, w2v_param['n_clusters'],
                                                                      w2v_param['linkage'], w2v_param['affinity'])
    PCA2_plot(tf_idf, labels_pred_tf, tags, "PCA Reduction of TF_IDF")
    PCA2_plot(w2v, labels_pred_w2v, tags, "PCA Reduction of Word2Vec")
    return (pd.DataFrame({'Purity': [Purity_tf, Purity_w2v], 'Adjusted Mutual Info': [AMI_tf, AMI_w2v],
                          'Normalized Mutual Info': [NMI_tf, NMI_w2v], 'Adjusted Rand Index': [ARI_tf, ARI_w2v]},
                         index=['tf_idf', 'w2v']))


def evaluate(vector, tags, n_clusters, linkage, affinity):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity).fit(vector)
    labels_true = tags
    labels_pred = clustering.labels_
    AMI = round((metrics.cluster.adjusted_mutual_info_score(labels_true, labels_pred)), 6)
    Purity = round(purity_score(labels_true, labels_pred), 6)
    NMI = round((metrics.cluster.normalized_mutual_info_score(labels_true, labels_pred)), 6)
    ARI = round(metrics.cluster.adjusted_rand_score(labels_true, labels_pred), 6)
    return Purity, AMI, NMI, ARI, labels_pred


def get_AMI(vector, tags, n_clusters, linkage, affinity):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity).fit(vector)
    labels_true = tags
    labels_pred = clustering.labels_
    AMI = round((metrics.cluster.adjusted_mutual_info_score(labels_true, labels_pred)), 6)
    return AMI


def get_others(vector, tags, n_clusters, linkage, affinity):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity).fit(vector)
    labels_true = tags
    labels_pred = clustering.labels_
    Purity = round(purity_score(labels_true, labels_pred), 6)
    NMI = round((metrics.cluster.normalized_mutual_info_score(labels_true, labels_pred)), 6)
    ARI = round(metrics.cluster.adjusted_rand_score(labels_true, labels_pred), 6)
    return Purity, NMI, ARI


def purity_score(labels_true, labels_pred):
    matrix = contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)


def PCA2_plot(vectors, labels_pred, labels_true, title):
    pca_reduction = PCA(2, random_state=12).fit_transform(vectors)
    fig, axes = plt.subplots(1, 2, figsize=(28, 8))
    axes[0].scatter(pca_reduction[:, 0], pca_reduction[:, 1], c=labels_pred, cmap='plasma')
    axes[0].set_title('Prediction')
    axes[1].scatter(pca_reduction[:, 0], pca_reduction[:, 1], c=labels_true, cmap='plasma')
    axes[1].set_title('Ground Truth')
    fig.suptitle(title)
    plt.grid()
    plt.show()


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
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)


def show_dendrogram(vector, linkage, affinity):
    plt.title('Hierarchical Clustering Dendrogram')
    clustering = AgglomerativeClustering(n_clusters=None, linkage=linkage, affinity=affinity,
                                         distance_threshold=1e-3).fit(vector)
    plot_dendrogram(clustering, truncate_mode='level', p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()


def final(tf_idf, tf_param, w2v, w2v_param, tags, links):
    Purity_tf, AMI_tf, NMI_tf, ARI_tf, labels_pred_tf = evaluate(tf_idf, tags, tf_param['n_clusters'],
                                                                 tf_param['linkage'], tf_param['affinity'])
    Purity_w2v, AMI_w2v, NMI_w2v, ARI_w2v, labels_pred_w2v = evaluate(w2v, tags, w2v_param['n_clusters'],
                                                                      w2v_param['linkage'], w2v_param['affinity'])
    save_csv(links, labels_pred_tf, labels_pred_w2v)
    return (pd.DataFrame(
        {'method': ['Hierarchical', 'Hierarchical'], 'vector': ['tf_idf', 'w2v'], 'Purity': [Purity_tf, Purity_w2v],
         'Adjusted Mutual Info': [AMI_tf, AMI_w2v], 'Normalized Mutual Info': [NMI_tf, NMI_w2v],
         'Adjusted Rand Index': [ARI_tf, ARI_w2v]}))


def save_csv(links, labels_pred_tf, labels_pred_w2v):
    pd.DataFrame({'link': links, 'predicted label': labels_pred_tf}).to_csv(
        "reports/phase3/csv_files/hierarchical_tfidf.csv")
    pd.DataFrame({'link': links, 'predicted label': labels_pred_w2v}).to_csv(
        "reports/phase3/csv_files/hierarchical_w2v.csv")
