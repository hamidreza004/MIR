import numpy as np
from sklearn.mixture import GaussianMixture

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# covariance_type{‘full’, ‘tied’, ‘diag’, ‘spherical’}
# ‘full’
# each component has its own general covariance matrix
#
# ‘tied’
# all components share the same general covariance matrix
#
# ‘diag’
# each component has its own diagonal covariance matrix
#
# ‘spherical’
# each component has its own single variance
gmm = GaussianMixture(n_components=4, covariance_type='full', tol=1e-3, max_iter=100, random_state=0).fit(X)

print(gmm.means_)
print(gmm.predict(X))
