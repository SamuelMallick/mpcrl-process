import numpy as np
import matplotlib.pyplot as plt
import pickle


def simulation_data_plot(data: dict):
    plot_ep = 0
    # sim data plot
    P_loads_env = data["P_loads"][plot_ep]
    P_loads_mpc = data["mpc_P_loads"]
    P_loads_mhe = data["mhe_P_loads_estimation_data"]
    _, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(P_loads_env, color="k")
    for i in range(P_loads_mpc.shape[0]):
        ax.plot(
            np.arange(i, P_loads_mpc[i].shape[1] + i),
            P_loads_mpc[i].T,
            alpha=1,
            color="r",
            linestyle="--",
        )
    for i in range(P_loads_mhe.shape[0]):
        ax.plot(
            np.arange(i, P_loads_mhe[i].shape[1] + i),
            P_loads_mhe[i].T,
            alpha=1,
            color="g",
            linestyle=":",
        )

    plt.show()


if __name__ == "__main__":
    with open("test_mpc_data.pkl", "rb") as f:
        sample_data = pickle.load(f)
    simulation_data_plot(sample_data)
