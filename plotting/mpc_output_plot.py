import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.animation import FuncAnimation


def mpc_output_live_plot(data: dict, plot_error: bool = False):
    plot_ep = 0

    if plot_error:
        y = data["y"][plot_ep, 1:, :17]
        y_open_loop = np.linalg.norm(data["mhe_y_open_loop"] - y, axis=1)
        y_estimated = np.linalg.norm(data["mhe_y_estimated"] - y, axis=1)
    else:
        y = data["y"][plot_ep, :, 16]
        y_prediction = [data["mpc_y_prediction"][k]["q_r"] for k in range(len(y) - 1)]
        N = y_prediction[0].shape[0]
        y_open_loop = data["mhe_y_open_loop"][:, 16]
        y_estimated = data["mhe_y_estimated"][:, 16]

    fig, ax = plt.subplots(1, figsize=(8, 4))
    (line_y,) = ax.plot([], [], "b-", label="y")
    (line_y_open_loop,) = ax.plot([], [], "g--", label="mhe_y_open_loop")
    (line_y_estimated,) = ax.plot([], [], "m-", label="mhe_y_estimated")
    (line_y_prediction,) = ax.plot([], [], "r--", label="mpc_y_pred")
    ax.set_ylim(np.min(y) - 2, np.max(y) + 2)
    ax.set_xlim(0, len(y) - 1)
    ax.legend()

    num_points = y.shape[0] - 1

    def update(k):
        ax.set_title(f"Output at step {k}")
        # Plot real output up to current time
        line_y_open_loop.set_data(np.arange(k + 1), y_open_loop[: k + 1])
        line_y_estimated.set_data(np.arange(k + 1), y_estimated[: k + 1])

        # Plot predicted horizon from k
        if not plot_error:
            line_y.set_data(np.arange(k + 1), y[: k + 1])
            horizon = np.arange(k, k + N)
            line_y_prediction.set_data(horizon, y_prediction[k])

        return line_y, line_y_prediction, line_y_open_loop, line_y_estimated

    ani = FuncAnimation(
        fig, update, frames=num_points, interval=100, blit=True, repeat=False
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    with open("base_2025-12-25_15-00.pkl", "rb") as f:
        sample_data = pickle.load(f)
    mpc_output_live_plot(sample_data, plot_error=False)
