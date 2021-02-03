from sklearn.cluster import AgglomerativeClustering
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

# linkage{‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
# ward’ minimizes the variance of the clusters being merged.
#
# ‘average’ uses the average of the distances of each observation of the two sets.
#
# ‘complete’ or ‘maximum’ linkage uses the maximum distances between all observations of the two sets.
#
# ‘single’ uses the minimum of the distances between all observations of the two sets.


# Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”. If linkage is “ward”, only “euclidean” is accepted.
clustering = AgglomerativeClustering(n_clusters=None, linkage='ward', affinity='euclidean', distance_threshold=1).fit(X)

print(clustering.labels_)

