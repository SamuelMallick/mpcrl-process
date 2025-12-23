from env import DHSSystem
from monitor_episodes import MonitorEpisodes
import pickle
import numpy as np

import sys, os

sys.path.append(os.getcwd())
from misc.save_data import save_simulation_data

# get simulation data
loads_folder = "simulation_data/loads_5"
with open(f"{loads_folder}/loads_min.pkl", "rb") as f:
    P_loads = pickle.load(f)
with open(f"{loads_folder}/elec_price.pkl", "rb") as f:
    elec_price = pickle.load(f)
with open(f"{loads_folder}/limits.pkl", "rb") as f:
    data = pickle.load(f)
    T_s_min = data["Ts_min"]
    T_r_min = data["Tr_min"]

env = MonitorEpisodes(
    DHSSystem(
        step_size=5.0 * 60,
        sim_data={
            "P_loads": P_loads,
            "elec_price": elec_price,
            "T_s_min": T_s_min,
            "T_r_min": T_r_min,
        },
        monitoring_data_set=np.zeros((1, 10)),
    )
)

env.reset()
for _ in range(100):
    env.step(75 * np.ones((1, 1)))
env.force_episode_end()

save_simulation_data("test_env_data", env)
