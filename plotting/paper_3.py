import os
import pickle
import sys

import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from misc.tikz import save2tikz


def basic_plot(data: dict, max_len: int = 100000000):
    plot_ep = 0
    # sim data plot
    P_loads, elec_price, T_s_min, T_r_min = (
        data["P_loads"][plot_ep, :max_len],
        data["elec_price"][plot_ep, :max_len],
        data["T_s_min"][plot_ep, :max_len],
        data["T_r_min"][plot_ep, :max_len],
    )
    _, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(P_loads, label=["P1", "P2", "P3", "P4", "P5"])
    ax[1].plot(elec_price, label="Electricity Price")
    ax[2].plot(T_s_min, label="T_s_min")
    ax[2].plot(T_r_min, label="T_r_min")
    # ax[0].legend()
    # ax[1].legend()
    # ax[2].legend()

    save2tikz(plt.gcf())
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
        basic_plot(sample_data, 4000)
