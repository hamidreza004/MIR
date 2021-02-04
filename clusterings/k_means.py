from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix
import numpy as np


def k_means(vector, tags, links, random_state, n_clusters):
    X = np.array(vector)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300, tol=1e-4).fit(X)
    labels_true = tags
    labels_pred = kmeans.labels_

    # print("--------------------------------------------------------------------------")
    # print("purity_score:", purity_score(labels_true, labels_pred))
    # print("adjusted_mutual_info:", round((metrics.cluster.adjusted_mutual_info_score(labels_true, labels_pred)), 6))
    # print("adjusted_rand_index:", round(metrics.cluster.adjusted_rand_score(labels_true, labels_pred), 6))
    # print("--------------------------------------------------------------------------")

    return round((metrics.cluster.adjusted_mutual_info_score(labels_true, labels_pred)), 6), random_state


def get_best_randomeness(vector, tags, links):
    best_score = 0
    best_random = 0
    for i in range(40):
        score, random = k_means(vector, tags, links, i*53, 14)
        if score > best_score:
            best_score = score
            best_random = random
        print(i)

    print(best_score, best_random)


def purity_score(labels_true, labels_pred):
    matrix = contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)
