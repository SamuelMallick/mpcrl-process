import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
from misc.tikz import save2tikz
from monitoring.mahalanobis_distance import MahalanobisDistance


def basic_plot(data_1: dict, data_2: dict, max_lens: list[int] = [100000000, 100000000]):
    # sim data plot

    skip = 10
    _, ax = plt.subplots(3, 1, sharex=True)
    colors = ["C0", "C1", "C2", "C3", "C4"]

    for data, max_len, idx in zip([data_1, data_2], max_lens, [0, 1]):
        start = idx * (max_lens[0] + 10)
        x_axis = np.arange(start, start + max_len, skip)
        # outputs and inputs plot
        u, y, T_s_min, T_r_min, q_r_min = (
            data["u"],
            data["y"],
            data["T_s_min"],
            data["T_r_min"],
            data["q_r_min"],
        )
        u, y, T_s_min, T_r_min, q_r_min = (
            u.reshape(-1),
            y.reshape(-1, y.shape[2]),
            T_s_min.reshape(-1),
            T_r_min.reshape(-1),
            q_r_min.reshape(-1),
        )
        Ts = y[:, [0, 3, 6, 9, 12]]
        qr = y[:, 16]

        ax[0].plot(x_axis, T_s_min[:max_len:skip], "black", label="_T_s_min")
        for T, c in zip(Ts.T, colors):
            ax[0].plot(
                x_axis,
                T[:max_len:skip],
                label=["T1", "T2", "T3", "T4", "T5"],
                color=c,
        )

        ax[1].plot(x_axis, q_r_min[:max_len:skip], "black", label="_q_r_min")
        ax[1].plot(x_axis, qr[0:max_len:skip], label="qr", color="C0")

        window_length = 144
        with open(f"monitoring/monitoring_data_set.pkl", "rb") as f:
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

        skip_first = 0
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

        start = idx * (max_lens[0] + 10) + window_length
        x_axis = np.arange(start, start + max_len - window_length, skip)
        ax[2].plot(
            x_axis,
            dists[: max_len - window_length : skip],
            label="Mahalanobis distance",
            color=colors[0],
        )
        ax[2].axhline(15.51, color="black", label="Threshold")
        # ax[2].set_yscale("log")

        # ax[0].legend()

        # save2tikz(plt.gcf())

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
        with open(file_name, "rb") as f:
            sample_data = pickle.load(f)
    else:
        with open("results/learn_q_2/2026-01-12_16-37_ep0_step5472.pkl", "rb") as f:
            sample_data = pickle.load(f)
    if len(sys.argv) > 2:
        max_len = int(sys.argv[2])
        basic_plot(sample_data, max_len=max_len)
    else:
        basic_plot(sample_data, sample_data, [288*3, 288*3])
