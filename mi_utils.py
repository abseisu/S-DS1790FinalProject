from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma

def _ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 5, metric: str = "chebyshev") -> float:
    x = np.asarray(x, dtype=float).reshape(-1,1) if x.ndim==1 else x
    y = np.asarray(y, dtype=float).reshape(-1,1) if y.ndim==1 else y
    xy = np.hstack([x,y]); n = xy.shape[0]
    if n <= k + 1: return 0.0
    nbrs_xy = NearestNeighbors(metric=metric, n_neighbors=k+1).fit(xy)
    d = nbrs_xy.kneighbors(xy, return_distance=True)[0][:, -1]
    nbrs_x = NearestNeighbors(metric=metric).fit(x)
    nbrs_y = NearestNeighbors(metric=metric).fit(y)
    nx = np.array([len(nbrs_x.radius_neighbors([x[i]], radius=d[i]-1e-15, return_distance=False)[0]) - 1 for i in range(n)])
    ny = np.array([len(nbrs_y.radius_neighbors([y[i]], radius=d[i]-1e-15, return_distance=False)[0]) - 1 for i in range(n)])
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx+1) + digamma(ny+1))
    return float(max(mi, 0.0))

def mutual_information(x, y, k: int = 5, y_is_discrete: Optional[bool] = None, seed: int = 0) -> float:
    x = np.asarray(x); y = np.asarray(y)
    if y_is_discrete is None:
        y_is_discrete = (np.unique(y).size < max(20, int(0.01 * y.size)))
    if y_is_discrete:
        X = x.reshape(-1,1) if x.ndim==1 else x
        return float(mutual_info_classif(X, y.astype(int), random_state=seed).mean())
    else:
        return _ksg_mi(x, y, k=k)

def permutation_test_mi(x, y, n_perm: int = 500, k: int = 5, y_is_discrete: Optional[bool] = None, seed: int = 0):
    rng = np.random.default_rng(seed)
    obs = mutual_information(x, y, k=k, y_is_discrete=y_is_discrete, seed=seed)
    cnt = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        if mutual_information(x, yp, k=k, y_is_discrete=y_is_discrete, seed=seed) >= obs:
            cnt += 1
    pval = (cnt + 1) / (n_perm + 1)
    return obs, pval
