import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np


def basic_plot(data: dict, max_len: int = 100000000):
    # sim data plot
    P_loads, elec_price = data["P_loads"], data["elec_price"]
    P_loads, elec_price = P_loads.reshape(-1, P_loads.shape[2]), elec_price.reshape(-1)
    _, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(P_loads, label=["P1", "P2", "P3", "P4", "P5"])
    ax[1].plot(elec_price, label="Electricity Price")
    ax[0].legend()
    ax[1].legend()

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
    Tb = y[:, 17]

    Tr = y[:, [1, 4, 7, 10, 13]]
    Tr_tot = y[:, 15]

    q = y[:, [2, 5, 8, 11, 14]]
    qr = y[:, 16]

    _, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(
        u[:max_len],
        "--",
        label="u",
    )
    ax[0].plot(Ts[:max_len], label=["T1", "T2", "T3", "T4", "T5"])
    ax[0].plot(Tb[:max_len], label="Tb")
    ax[0].plot(T_s_min[:max_len], "k:", label="_T_s_min")

    ax[1].plot(Tr[:max_len], label=["T1", "T2", "T3", "T4", "T5"])
    ax[1].plot(Tr_tot[:max_len], label="Tr")
    ax[1].plot(T_r_min[:max_len], "k:", label="_T_r_min")

    ax[2].plot(q[:max_len], label=["q1", "q2", "q3", "q4", "q5"])
    ax[2].plot(qr[:max_len], label="qr")
    ax[2].plot(q_r_min[:max_len], "k:", label="_q_r_min")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    # costs plot
    r = data["rewards"]
    if "agent_td_errors" in data:
        perf = data["monitoring_distance"].squeeze()
    elif "agent_policy_performances" in data:
        perf = data["agent_policy_performances"]
    else:
        perf = np.sum(r, axis=1)
    _, ax = plt.subplots(2, 1, sharex=False)
    ax[0].plot(r.reshape(-1)[:max_len], "o", label="Reward")
    ax[1].plot(perf[:max_len], label="Performance")
    # ax[1].set_yscale("log")

    # parameters plot
    updates = data["agent_updates_history"]
    _, ax = plt.subplots(len(updates), 1, sharex=True)
    if len(updates) == 1:
        ax = [ax]
    for i, (name, vals) in enumerate(updates.items()):
        val = np.asarray(vals)
        val = val.reshape(val.shape[0], -1)
        ax[i].plot(val[:max_len])
        ax[i].set_ylabel(name)

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
        basic_plot(sample_data)
