import os
import sys
import importlib
from gymnasium.wrappers import TimeLimit
from datetime import datetime
import casadi as cs
import numpy as np

from mpcrl import LearnableParameter, LearnableParametersDict

from agent.agent import DhsAgent, DhsDpgAgent, DhsQLearningAgent
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
    mod = importlib.import_module(f"config_files.{config_file}")
    config = mod.Config()
else:
    from config_files.learn_dpg import Config  # type: ignore

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
if config.learning_rate == 0:
    agent = DhsAgent(
        mpc=mpc,
        observer=mhe,
        fixed_parameters={},
        save_frequency=288,
        save_location=f"results/{config.id}/{s}",
    )
    agent.evaluate(env=env, episodes=1, seed=1, raises=True)
else:
    learnable_pars_init = {
        name: config.mpc_pars[name] for name in config.learnable_pars
    }
    learnable_pars = LearnableParametersDict[cs.SX](
        (
            LearnableParameter(name, val.shape, val)
            for name, val in learnable_pars_init.items()
        )
    )
    if config.ddpg:
        agent = DhsDpgAgent(
            mpc=mpc,
            observer=mhe,
            discount_factor=config.gamma,
            update_strategy=config.update_strategy,
            optimizer=config.optimizer,
            learnable_parameters=learnable_pars,
            experience=config.experience,
            exploration=config.exploration,
            rollout_length=config.rollout_length,
            fixed_parameters={},
            save_frequency=72,
            save_location=f"results/{config.id}/{s}",
        )
    else:
        agent = DhsQLearningAgent(
            mpc=mpc,
            observer=mhe,
            discount_factor=config.gamma,
            update_strategy=config.update_strategy,
            optimizer=config.optimizer,
            learnable_parameters=learnable_pars,
            experience=config.experience,
            exploration=config.exploration,
            fixed_parameters={},
            save_frequency=72,
            save_location=f"results/{config.id}/{s}",
        )
    agent.train(env=env, episodes=1, seed=1, raises=True)
save_simulation_data(f"results/{config.id}/{s}", env, mpc, mhe)
