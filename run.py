import importlib
import logging
import os
import sys
from datetime import datetime

import casadi as cs
import numpy as np
from gymnasium.wrappers import TimeLimit
from mpcrl import LearnableParameter, LearnableParametersDict
from mpcrl.wrappers.agents import Log, RecordUpdates

from agent.agent import (DhsAgent, DhsDpgAgent, DhsGlobOptAgent,
                         DhsQLearningAgent)
from misc.save_data import save_simulation_data
from mpc.mpc import DhsMpc
from mpc.mpc_recorder import MpcRecorder
from mpc.observer.mhe import Mhe
from mpc.observer.mhe_recorder import MheRecorder
from simulation_model.env import DHSSystem
from simulation_model.monitor_episodes import MonitorEpisodes

# if a config file passed on command line, otherwise use default config file
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    mod = importlib.import_module(f"config_files.{config_file}")
    config = mod.Config()
else:
    from config_files.learn_bo_u_offset import Config  # type: ignore

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
            use_distance_reward=config.use_distance_reward,
            w=config.w,
            u_offset=config.u_offset,
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
if config.learn_type == "none":
    agent = Log(
        DhsAgent(
            mpc=mpc,
            observer=mhe,
            fixed_parameters={},
        ),
        level=logging.DEBUG,
        log_frequencies={"on_env_step": 1},
    )
    agent.evaluate(
        env=env,
        episodes=1,
        seed=1,
        raises=True,
        save_frequency=72,
        save_location=f"results/{config.id}/{s}",
    )
else:
    learnable_pars_init = {
        name: config.mpc_pars[name] for name in config.learnable_pars
    }
    learnable_pars = LearnableParametersDict[cs.SX](
        (
            LearnableParameter(
                name,
                learnable_pars_init[name].shape,
                learnable_pars_init[name],
                config.learnable_pars_bounds.get(name, (-np.inf, +np.inf))[0],
                config.learnable_pars_bounds.get(name, (-np.inf, +np.inf))[1],
            )
            for name in learnable_pars_init.keys()
        )
    )
    if config.learn_type == "dpg":
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
            record_policy_performance=True,
            record_policy_gradient=True,
        )
    elif config.learn_type == "q_learning":
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
            record_td_errors=True,
        )
    elif config.learn_type == "bo":
        agent = DhsGlobOptAgent(
            mpc=mpc,
            observer=mhe,
            learnable_parameters=learnable_pars,
            optimizer=config.optimizer,
            fixed_parameters={},
        )
    agent = Log(
        RecordUpdates(agent), level=logging.DEBUG, log_frequencies={"on_env_step": 1}
    )
    agent.train(
        env=env,
        episodes=config.episodes,
        seed=1,
        raises=True,
        save_frequency=288,
        save_location=f"results/{config.id}/{s}",
        update_recorder=agent,
    )
save_simulation_data(f"results/{config.id}/{s}", env, mpc, mhe, agent)
