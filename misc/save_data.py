import pickle
import numpy as np
from simulation_model.monitor_episodes import MonitorEpisodes


def save_simulation_data(save_name: str, env: MonitorEpisodes):
    data = {
        "u": np.asarray(env.actions),
        "y": np.asarray(env.observations),
        "rewards": np.asarray(env.rewards),
    }
    for key, values in env.extra_data.items():
        data[key] = np.asarray(values)
    with open(f"{save_name}.pkl", "wb") as f:
        pickle.dump(data, f)
