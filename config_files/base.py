import pickle

import numpy as np


class BaseConfig:
    id = "base"

    # simulations definition
    sim_len = 288 * 200
    episodes = 1
    ts = 5.0 * 60
    w = 20.0
    with open("monitoring/monitoring_data_set.pkl", "rb") as f:
        monitoring_data_set = pickle.load(f)
    monitoring_window = 288
    u_offset = 0

    # simulation data
    loads_folder = "simulation_data/loads_5"

    # mpc
    layers_path = "mpc/prediction_model/layers_full.mat"
    input_scaler_path = "mpc/prediction_model/input_scaler_full.mat"
    output_scaler_path = "mpc/prediction_model/output_scaler_full.mat"
    gamma = 1
    N = 72
    input_block = 1
    num_inputs = 3

    # mhe
    open_loop = False
    mhe_horizon = 50

    mpc_pars = {
        "T_ref": 75 * np.ones((5,)),
        "w": w * np.ones((1,)),
        "c_t": 1e1 * np.ones((1,)),
        "V0": np.zeros((1,)),
        "f": np.zeros((1 + 7 + 1,)),
        "Q": np.zeros((1 + 7 + 1, 1 + 7 + 1)),
        "T_lim_off": np.zeros((5,)),
        "q_lim_off": np.zeros((1,)),
        "u_offset": u_offset * np.ones((1,)),
    }

    # learning
    learning_rate = 0
    learnable_pars = []
    learnable_pars_bounds = {}
    learn_type = "none"
    optimizer = None
    experience = None
    exploration = None
    update_strategy = None
    rollout_length = None

    def __init__(self):
        self.P_loads, self.elec_price, self.T_s_min, self.T_r_min = (
            self.generate_sim_data(self.sim_len, self.loads_folder)
        )

    def generate_sim_data(self, sim_len, loads_folder):
        with open(f"{loads_folder}/loads_min.pkl", "rb") as f:
            P_loads = pickle.load(f)
            num_repeats = int(np.ceil(sim_len / max(P_loads.shape)))
            P_loads = np.tile(P_loads, (1, num_repeats))
            P_loads = np.hstack(
                [P_loads, P_loads[:, -1][:, None].repeat(self.N + 10, axis=1)]
            )  # pad for N steps
        with open(f"{loads_folder}/elec_price.pkl", "rb") as f:
            elec_price = pickle.load(f)
            num_repeats = int(np.ceil(sim_len / max(elec_price.shape)))
            elec_price = np.tile(elec_price, (1, num_repeats))
            elec_price = np.hstack(
                [elec_price, elec_price[:, -1] * np.ones((1, self.N + 10))]
            ).squeeze()  # pad for N steps
        with open(f"{loads_folder}/limits.pkl", "rb") as f:
            data = pickle.load(f)
            T_s_min = data["Ts_min"]
            T_r_min = data["Tr_min"]
            num_repeats = int(np.ceil(sim_len / max(T_s_min.shape)))
            T_s_min = np.tile(T_s_min, (1, num_repeats))
            T_r_min = np.tile(T_r_min, (1, num_repeats))
            T_s_min = np.hstack(
                [T_s_min, T_s_min[:, -1] * np.ones((1, self.N + 10))]
            ).squeeze()  # pad for N steps
            T_r_min = np.hstack(
                [T_r_min, T_r_min[:, -1] * np.ones((1, self.N + 10))]
            ).squeeze()  # pad for N steps
        return P_loads, elec_price, T_s_min, T_r_min
