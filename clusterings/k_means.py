from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np


def k_means(tf_idf, w2v):
    print(tf_idf['title'])

# X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0], [5, 5], [12, 3], [2, 10], [5, 12]])
# kmeans = KMeans(n_clusters=3, n_init=10, max_iter=300, tol=1e-4).fit(X)
# print(kmeans.labels_)
# print(kmeans.predict(X))
# # print(kmeans.cluster_centers_)
# print(kmeans.inertia_)
#
# labels_true = kmeans.labels_
# labels_pred = kmeans.predict(X)
#
# print(metrics.cluster.adjusted_rand_score(labels_true, labels_pred))
# print(metrics.cluster.adjusted_mutual_info_score(labels_true, labels_pred))
# print(metrics.cluster.normalized_mutual_info_score(labels_true, labels_pred))