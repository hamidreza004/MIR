import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris


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
clustering = AgglomerativeClustering(n_clusters=None, linkage='ward', affinity='euclidean', distance_threshold=0.1).fit(X)

# print(clustering.labels_)

plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(clustering, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
