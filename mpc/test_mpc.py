import os
import pickle
import sys

import numpy as np
from mpc_recorder import MpcRecorder
from observer.mhe import Mhe
from observer.mhe_recorder import MheRecorder

from mpc import DhsMpc

sys.path.append(os.getcwd())
from misc.save_data import save_simulation_data
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

env.reset()
for k in range(200):
    if k > 0:
        mhe_data = {
            "P_loads": np.asarray(
                env.ep_extra_data["P_loads"][-mhe.prediction_horizon :]
            ).T,
            "T_s": np.asarray(env.ep_observations[-mhe.prediction_horizon - 1 : -1])[
                :, [18]
            ].T,
            "y": np.asarray(env.ep_observations[-mhe.prediction_horizon - 1 : -1])[
                :, :17
            ].T,
        }
        mhe.update_state(data=mhe_data)
    pars = {
        "x": mhe.get_x(),
        "P_loads": P_loads[:, k : k + mpc.prediction_horizon],
        "elec_price": elec_price[k : k + mpc.prediction_horizon],
        "T_s_min": T_s_min[k : k + mpc.prediction_horizon],
        "T_r_min": T_r_min[k : k + mpc.prediction_horizon],
    }
    sol = mpc.solve(pars=pars, vals0=sol.vals if k > 0 else {})
    if not sol.success:
        raise RuntimeError("MPC solver failed")
    u = sol.vals["T_b_s"][0, 0].full()
    mhe.step(u=np.vstack((u, P_loads[:, [k]])))
    env.step(u)
env.force_episode_end()

save_simulation_data("test_mpc_data", env, mpc, mhe)
