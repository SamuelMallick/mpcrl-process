import casadi as cs
import numpy as np
from scipy.io import loadmat


class MahalanobisDistance:

    def __init__(self, data_ref: list[np.ndarray], scaling: list[tuple] = []):
        if scaling:
            if len(scaling) != len(data_ref):
                raise ValueError("Length of scaling must match number of features.")
            data_ref = [
                (data_ref[i] - scaling[i][1]) / (scaling[i][0] + 1e-12)
                for i in range(len(data_ref))
            ]
        self.mu = [np.mean(data_ref[i], axis=0) for i in range(len(data_ref))]
        self.S = [np.cov(data_ref[i], rowvar=False) for i in range(len(data_ref))]
        self.S_inv = [np.linalg.inv(s) for s in self.S]
        self.scaling = scaling

    def mahalanobis_distance(self, data_new: np.ndarray, return_all: bool = False):
        if self.scaling:
            data_new = [
                (data_new - self.scaling[i][1]) / (self.scaling[i][0] + 1e-12)
                for i in range(len(self.scaling))
            ]
        dists = [
            (d - mu) @ s_inv @ (d - mu).T
            for d, mu, s_inv in zip(data_new, self.mu, self.S_inv)
        ]
        for i in range(len(dists)):
            if dists[i].ndim == 0:
                dists[i] = np.array([[dists[i]]])
        dists = [np.diag(d) for d in dists]
        dists = [np.sqrt(d.flatten()) for d in dists]
        return dists if return_all else [np.mean(d) for d in dists]
