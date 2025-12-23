import numpy as np
import matplotlib.pyplot as plt
import pickle


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

    plt.show()


if __name__ == "__main__":
    with open("test_mpc_data.pkl", "rb") as f:
        sample_data = pickle.load(f)
    basic_plot(sample_data)
