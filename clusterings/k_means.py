from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix
import numpy as np


def get_cost(vector, random_state=452, n_clusters=3):
    X = np.array(vector)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300, tol=1e-4).fit(X)
    return kmeans.inertia_


def evaluate(vector, tags, random_state=452, n_clusters=5):
    X = np.array(vector)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300, tol=1e-4).fit(X)
    labels_true = tags
    labels_pred = kmeans.labels_

    Purity = round(purity_score(labels_true, labels_pred), 6)
    AMI = round((metrics.cluster.adjusted_mutual_info_score(labels_true, labels_pred)), 6)
    NMI = round((metrics.cluster.normalized_mutual_info_score(labels_true, labels_pred)), 6)
    ARI = round(metrics.cluster.adjusted_rand_score(labels_true, labels_pred), 6)

    return Purity, AMI, NMI, ARI


# def get_best_randomeness(vector, tags, links):
#     best_score = 0
#     best_random = 0
#     for i in range(40):
#         score, random = k_means(vector, tags, links, i * 53, 14)
#         if score > best_score:
#             best_score = score
#             best_random = random
#         print(i)
#
#     print(best_score, best_random)


def purity_score(labels_true, labels_pred):
    matrix = contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)
