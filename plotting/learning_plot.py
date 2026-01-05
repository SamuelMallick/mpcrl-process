import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np


def basic_plot(data: dict):
    plot_ep = 0
    # sim data plot
    P_loads, elec_price = data["P_loads"][plot_ep], data["elec_price"][plot_ep]
    _, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(P_loads, label=["P1", "P2", "P3", "P4", "P5"])
    ax[1].plot(elec_price, label="Electricity Price")
    ax[0].legend()
    ax[1].legend()

    # outputs and inputs plot
    u, y, T_s_min, T_r_min, q_r_min = (
        data["u"][plot_ep],
        data["y"][plot_ep],
        data["T_s_min"][plot_ep],
        data["T_r_min"][plot_ep],
        data["q_r_min"][plot_ep],
    )
    Ts = y[:, [0, 3, 6, 9, 12]]
    Tb = y[:, 17]

    Tr = y[:, [1, 4, 7, 10, 13]]
    Tr_tot = y[:, 15]

    q = y[:, [2, 5, 8, 11, 14]]
    qr = y[:, 16]

    _, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(
        u,
        "--",
        label="u",
    )
    ax[0].plot(Ts, label=["T1", "T2", "T3", "T4", "T5"])
    ax[0].plot(Tb, label="Tb")
    ax[0].plot(T_s_min, "k:", label="_T_s_min")

    ax[1].plot(Tr, label=["T1", "T2", "T3", "T4", "T5"])
    ax[1].plot(Tr_tot, label="Tr")
    ax[1].plot(T_r_min, "k:", label="_T_r_min")

    ax[2].plot(q, label=["q1", "q2", "q3", "q4", "q5"])
    ax[2].plot(qr, label="qr")
    ax[2].plot(q_r_min, "k:", label="_q_r_min")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    # costs plot
    r = data["rewards"]
    r = r[r != 0]
    r = r[20:]
    _, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(r.T, label="Reward")

    # parameters plot
    updates = data["agent_updates_history"]
    _, ax = plt.subplots(len(updates), 1, sharex=True)
    for i, (name, vals) in enumerate(updates.items()):
        val = np.asarray(vals)
        val = val.reshape(val.shape[0], -1)
        ax[i].plot(val)
        ax[i].set_ylabel(name)

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
        with open(file_name, "rb") as f:
            sample_data = pickle.load(f)
    else:
        with open("results/learn_q/2026-01-05_13-48_step864.pkl", "rb") as f:
            sample_data = pickle.load(f)
    basic_plot(sample_data)
