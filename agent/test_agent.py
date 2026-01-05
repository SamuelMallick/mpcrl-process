import os
import pickle
import sys

import numpy as np
from gymnasium.wrappers import TimeLimit

from agent import DhsAgent

sys.path.append(os.getcwd())
from misc.save_data import save_simulation_data
from mpc.mpc import DhsMpc
from mpc.mpc_recorder import MpcRecorder
from mpc.observer.mhe import Mhe
from mpc.observer.mhe_recorder import MheRecorder
from simulation_model.env import DHSSystem
from simulation_model.monitor_episodes import MonitorEpisodes

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
    TimeLimit(
        DHSSystem(
            step_size=5.0 * 60,
            sim_data={
                "P_loads": P_loads,
                "elec_price": elec_price,
                "T_s_min": T_s_min,
                "T_r_min": T_r_min,
            },
            monitoring_data_set=np.zeros((1, 10)),
        ),
        max_episode_steps=200,
    )
)


mpc_pars = {
    "T_ref": 75 * np.ones((5,)),
    "w": 20 * np.ones((1,)),
    "c_t": 1e1 * np.ones((1,)),
    "V0": np.zeros((1,)),
    "f": np.zeros((1 + 7 + 1,)),
    "Q": np.zeros((1 + 7 + 1, 1 + 7 + 1)),
    "T_lim_off": np.zeros((5,)),
    "q_lim_off": np.zeros((1,)),
}
mpc = MpcRecorder(
    DhsMpc(
        dt=5.0 * 60,
        prediction_horizon=72,
        layers_path="mpc/prediction_model/layers_full",
        input_scaler_path="mpc/prediction_model/input_scaler_full",
        output_scaler_path="mpc/prediction_model/output_scaler_full",
        pars_init=mpc_pars,
        num_inputs=3,
        gamma=1,
    )
)
mhe = MheRecorder(
    Mhe(
        prediction_horizon=50,
        layers_path="mpc/prediction_model/layers_full",
        input_scaler_path="mpc/prediction_model/input_scaler_full",
        output_scaler_path="mpc/prediction_model/output_scaler_full",
    )
)

agent = DhsAgent(mpc=mpc, observer=mhe, fixed_parameters={})

agent.evaluate(env=env, episodes=1, seed=1, raises=True)

save_simulation_data("test_agent_data", env, mpc, mhe)
