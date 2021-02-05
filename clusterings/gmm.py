import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn import metrics


# tf_idf = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
# w2v = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
# tags = np.array([0, 1, 1, 2, 0, 1])


def get_result_df(tf_idf, w2v, tags):
    vectors = ['tf_idf', 'w2v']
    n_components = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # n_components = [2, 3, 4]
    covariance_types = ['full', 'tied', 'diag', 'spherical']
    max_iters = [100, 200]

    result = []
    for vector in vectors:
        for n_component in n_components:
            for covariance_type in covariance_types:
                for max_iter in max_iters:
                    if vector == 'tf_idf':
                        score = get_AMI(tf_idf, tags, n_component, covariance_type, max_iter)
                    elif vector == 'w2v':
                        score = get_AMI(w2v, tags, n_component, covariance_type, max_iter)
                    result.append([vector, n_component, covariance_type, max_iter, score])

    return pd.DataFrame(data=result, columns=['vector', 'n_components', 'covariance_type', 'max_iter', 'AMI'])


def get_AMI(vector, tags, n_component, covariance_type, max_iter):
    gmm = GaussianMixture(n_components=n_component, covariance_type=covariance_type, max_iter=max_iter,
                          random_state=12).fit(vector)

    labels_true = tags
    labels_pred = gmm.predict(vector)
    AMI = round((metrics.cluster.adjusted_mutual_info_score(labels_true, labels_pred)), 6)

    return AMI
