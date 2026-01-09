import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
from monitoring.mahalanobis_distance import MahalanobisDistance


def plot_mal_distance(data: dict):

    with open("monitoring/monitoring_data_set_short.pkl", "rb") as f:
        monitoring_data_set = pickle.load(f)

    monitoring_distance_calculator = MahalanobisDistance(
        [monitoring_data_set],
        [
            (
                np.std(monitoring_data_set, axis=0),
                np.mean(monitoring_data_set, axis=0),
            )
        ],
    )

    skip_first = 10
    P_loads = data["P_loads"][:, skip_first:]
    y = data["y"][:, skip_first:-1]
    r = data["rewards"][:, skip_first:]
    P_loads, y, r = (
        P_loads.reshape(-1, P_loads.shape[2]),
        y.reshape(-1, y.shape[2]),
        r.reshape(-1),
    )
    # r = r[r != 0]

    # economic cost
    economic_cost = data["economic_cost"][:, skip_first:]
    economic_cost = economic_cost.reshape(-1, 1)
    # efficiency
    eff = -np.sum(P_loads, axis=1, keepdims=True) / y[:, [19]]

    # constraint violations
    violation_cost = data["constraint_violation_cost"][:, skip_first:]
    violation_cost = violation_cost.reshape(-1, 1)

    # disturbance sum
    disturbance_sum = -np.sum(P_loads, axis=1, keepdims=True)

    # create features
    window_length = 72
    feature_vector = np.hstack((eff, economic_cost, violation_cost, disturbance_sum)).T

    # cumulative sums along axis=1
    c1 = np.cumsum(feature_vector, axis=1)
    c2 = np.cumsum(feature_vector**2, axis=1)

    # shift for window sums
    sum1 = c1[:, window_length - 1 :] - np.concatenate(
        [np.zeros((feature_vector.shape[0], 1)), c1[:, :-window_length]], axis=1
    )
    sum2 = c2[:, window_length - 1 :] - np.concatenate(
        [np.zeros((feature_vector.shape[0], 1)), c2[:, :-window_length]], axis=1
    )

    # moving mean and variance
    mean = sum1 / window_length
    var = (sum2 / window_length) - mean**2

    X = np.vstack((mean, var)).T
    dists = monitoring_distance_calculator.mahalanobis_distance(X, return_all=True)[0]

    skip_first_for_plot = 10
    dists = dists[skip_first_for_plot:]
    r = r[skip_first_for_plot:]
    monitoring_data_set = monitoring_data_set[skip_first_for_plot:]
    X = X[skip_first_for_plot:]

    _, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(dists, "--", label="post")
    ax.plot(r, "o", label="during")

    fig, ax = plt.subplots(3, 2, sharex=False)
    fig.suptitle("Features")
    ax[0, 0].plot(
        monitoring_data_set[:, 3], monitoring_data_set[:, 0], "o", color="blue"
    )
    ax[1, 0].plot(
        monitoring_data_set[:, 3], monitoring_data_set[:, 1], "o", color="blue"
    )
    ax[2, 0].plot(
        monitoring_data_set[:, 3], monitoring_data_set[:, 2], "o", color="blue"
    )
    ax[0, 1].plot(
        monitoring_data_set[:, 7], monitoring_data_set[:, 4], "o", color="blue"
    )
    ax[1, 1].plot(
        monitoring_data_set[:, 7], monitoring_data_set[:, 5], "o", color="blue"
    )
    ax[2, 1].plot(
        monitoring_data_set[:, 7], monitoring_data_set[:, 6], "o", color="blue"
    )
    ax[0, 0].plot(X[:, 3], X[:, 0], "o", color="red", alpha=0.5)
    ax[1, 0].plot(X[:, 3], X[:, 1], "o", color="red", alpha=0.5)
    ax[2, 0].plot(X[:, 3], X[:, 2], "o", color="red", alpha=0.5)
    ax[0, 1].plot(X[:, 7], X[:, 4], "o", color="red", alpha=0.5)
    ax[1, 1].plot(X[:, 7], X[:, 5], "o", color="red", alpha=0.5)
    ax[2, 1].plot(X[:, 7], X[:, 6], "o", color="red", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
        with open(file_name, "rb") as f:
            data = pickle.load(f)
    else:
        with open(
            "results/learn_bo_u_offset/2026-01-06_16-32_ep39_step288.pkl", "rb"
        ) as f:
            data = pickle.load(f)
    plot_mal_distance(data)
