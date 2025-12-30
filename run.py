import os
import sys
import importlib
import numpy as np
from gymnasium.wrappers import TimeLimit
from datetime import datetime

from agent.agent import DhsAgent
from simulation_model.env import DHSSystem
from mpc.mpc import DhsMpc
from simulation_model.monitor_episodes import MonitorEpisodes
from misc.save_data import save_simulation_data
from mpc.observer.mhe import Mhe
from mpc.observer.mhe_recorder import MheRecorder
from mpc.mpc_recorder import MpcRecorder

# if a config file passed on command line, otherwise use default config file
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    mod = importlib.import_module(f"learning_configs.{config_file}")
    config = mod.Config()
else:
    from config_files.generate_data import Config  # type: ignore

    config = Config()

env = MonitorEpisodes(
    TimeLimit(
        DHSSystem(
            step_size=config.ts,
            sim_data={
                "P_loads": config.P_loads,
                "elec_price": config.elec_price,
                "T_s_min": config.T_s_min,
                "T_r_min": config.T_r_min,
            },
            monitoring_data_set=config.monitoring_data_set,
            monitoring_window=config.monitoring_window,
        ),
        max_episode_steps=config.sim_len,
    )
)

mpc = MpcRecorder(
    DhsMpc(
        dt=config.ts,
        prediction_horizon=config.N,
        layers_path=config.layers_path,
        input_scaler_path=config.input_scaler_path,
        output_scaler_path=config.output_scaler_path,
        pars_init=config.mpc_pars,
        num_inputs=config.num_inputs,
        gamma=config.gamma,
    )
)
mhe = MheRecorder(
    Mhe(
        prediction_horizon=config.mhe_horizon,
        layers_path=config.layers_path,
        input_scaler_path=config.input_scaler_path,
        output_scaler_path=config.output_scaler_path,
    )
)

now = datetime.now()
s = now.strftime("%Y-%m-%d_%H-%M")
os.makedirs(f"results/{config.id}", exist_ok=True)
agent = DhsAgent(
    mpc=mpc,
    observer=mhe,
    fixed_parameters={},
    save_frequency=72,
    save_location=f"results/{config.id}/{s}",
)


agent.evaluate(env=env, episodes=1, seed=1, raises=True)
save_simulation_data(f"results/{config.id}/{s}", env, mpc, mhe)
