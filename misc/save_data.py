import pickle
import numpy as np
from simulation_model.monitor_episodes import MonitorEpisodes

import sys, os

sys.path.append(os.getcwd())
from mpc.mpc_recorder import MpcRecorder
from mpc.observer.mhe_recorder import MheRecorder


def save_simulation_data(
    save_name: str,
    env: MonitorEpisodes,
    mpc: MpcRecorder | None = None,
    mhe: MheRecorder | None = None,
) -> None:
    data = {
        "u": np.asarray(env.actions),
        "y": np.asarray(env.observations),
        "rewards": np.asarray(env.rewards),
    }
    for key, values in env.extra_data.items():
        data[key] = np.asarray(values)

    if mpc is not None:
        data["mpc_solver_time"] = np.asarray(mpc.solver_time)
        data["mpc_y_prediction"] = mpc.y_prediction
        data["mpc_P_loads"] = np.asarray(mpc.P_loads)

    if mhe is not None:
        data["mhe_y_open_loop"] = np.asarray(mhe.y_open_loop)
        data["mhe_y_estimated"] = np.asarray(mhe.y_estimated)
        data["mhe_action_step"] = np.asarray(mhe.action_step)
        data["mhe_P_loads_estimation_data"] = np.asarray(mhe.P_loads_estimation_data)

    with open(f"{save_name}.pkl", "wb") as f:
        pickle.dump(data, f)

    print(f"saved simulation data to {save_name}.pkl")
