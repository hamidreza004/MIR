import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix


def get_result_df(tf_idf, w2v, tags):
    vectors = ['tf_idf', 'w2v']
    n_components = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
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


def get_evaluation_dataframe(tf_idf, tf_param, w2v, w2v_param, tags):
    Purity_tf, AMI_tf, NMI_tf, ARI_tf = evaluate(tf_idf, tags, tf_param['n_components'], tf_param['covariance_type'],
                                                 tf_param['max_iter'])
    Purity_w2v, AMI_w2v, NMI_w2v, ARI_w2v = evaluate(w2v, tags, w2v_param['n_components'], w2v_param['covariance_type'],
                                                     w2v_param['max_iter'])

    return (pd.DataFrame({'Purity': [Purity_tf, Purity_w2v], 'Adjusted Mutual Info': [AMI_tf, AMI_w2v],
                          'Normalized Mutual Info': [NMI_tf, NMI_w2v], 'Adjusted Rand Index': [ARI_tf, ARI_w2v]},
                         index=['tf_idf', 'w2v']))


def get_AMI(vector, tags, n_component, covariance_type, max_iter, random_state=12):
    gmm = GaussianMixture(n_components=n_component, covariance_type=covariance_type, max_iter=max_iter,
                          random_state=random_state).fit(vector)
    labels_true = tags
    labels_pred = gmm.predict(vector)
    AMI = round((metrics.cluster.adjusted_mutual_info_score(labels_true, labels_pred)), 6)

    return AMI


def get_best_randomeness(vector, tags, n_component, covariance_type, max_iter):
    best_score = 0
    best_random = 0
    for i in range(5):
        score = get_AMI(vector, tags, n_component, covariance_type, max_iter, i * 53)
        if score > best_score:
            best_score = score
            best_random = i * 53
    return best_random


def evaluate(vector, tags, n_component, covariance_type, max_iter):
    best_random_state = get_best_randomeness(vector, tags, n_component, covariance_type, max_iter)
    gmm = GaussianMixture(n_components=n_component, covariance_type=covariance_type, max_iter=max_iter,
                          random_state=best_random_state).fit(vector)
    labels_true = tags
    labels_pred = gmm.predict(vector)
    Purity = round(purity_score(labels_true, labels_pred), 6)
    AMI = round((metrics.cluster.adjusted_mutual_info_score(labels_true, labels_pred)), 6)
    NMI = round((metrics.cluster.normalized_mutual_info_score(labels_true, labels_pred)), 6)
    ARI = round(metrics.cluster.adjusted_rand_score(labels_true, labels_pred), 6)

    return Purity, AMI, NMI, ARI


def purity_score(labels_true, labels_pred):
    matrix = contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)
