import pickle
import numpy as np

class BaseConfig:
    id = "base"

    # simulations definition
    sim_len = 288 * 200
    ts = 5.0 * 60
    w = 20.0

    # simulation data
    loads_folder = "simulation_data/loads_5"
    with open(f"{loads_folder}/loads_min.pkl", "rb") as f:
        P_loads = pickle.load(f)
    with open(f"{loads_folder}/elec_price.pkl", "rb") as f:
        elec_price = pickle.load(f)
    with open(f"{loads_folder}/limits.pkl", "rb") as f:
        data = pickle.load(f)
        T_s_min = data["Ts_min"]
        T_r_min = data["Tr_min"]

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
    }
