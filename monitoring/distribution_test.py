import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from numpy.linalg import inv
import pandas as pd

with open("monitoring/monitoring_data_set_144.pkl", "rb") as f:
        X = pickle.load(f)

N, p = X.shape

# Estimate mean and covariance
mu = X.mean(axis=0)
Sigma = np.cov(X, rowvar=False)
Sigma_inv = inv(Sigma)

# Squared Mahalanobis distances
d2 = np.array([
    (x - mu).T @ Sigma_inv @ (x - mu)
    for x in X
])

# Sort empirical distances
d2_sorted = np.sort(d2)

# Theoretical chi-square quantiles
q = chi2.ppf((np.arange(1, N+1) - 0.5) / N, df=p)

# Plot Q-Q
plt.figure()
plt.plot(q, d2_sorted, 'o', markersize=4)
plt.plot(q, q, 'r--', linewidth=2)
plt.xlabel(r'Theoretical $\chi^2_8$ quantiles')
plt.ylabel('Empirical squared Mahalanobis distances')
plt.title(r'$\chi^2$ Q–Q plot (Mahalanobis distances)')
plt.grid(True)

# plt.figure()
# plt.plot(d2_sorted, 'o', markersize=4)


# df = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])

# pd.plotting.scatter_matrix(
#     df,
#     figsize=(10, 10),
#     diagonal='kde',
#     alpha=0.6
# )

# plt.suptitle('Pairwise scatter plots of X', y=1.02)
plt.show()

    