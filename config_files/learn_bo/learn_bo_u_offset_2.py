import pickle

import numpy as np
from mpcrl import ExperienceReplay, UpdateStrategy
from mpcrl.core.exploration import EpsilonGreedyExploration
from mpcrl.core.schedulers import ExponentialScheduler
from mpcrl.optim import NewtonMethod

from config_files.base import BaseConfig
from optimizers.bo import BoTorchOptimizer


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.id = "learn_bo_u_offset_2"

        self.sim_len = 288
        self.episodes = 100

        self.u_offset = -1

        with open("monitoring/monitoring_data_set_short.pkl", "rb") as f:
            self.monitoring_data_set = pickle.load(f)
        self.monitoring_window = 72

        # learning
        self.learn_type = "bo"
        self.acquisition_function = "ucb"
        self.optimizer = BoTorchOptimizer(initial_random=2, seed=1, acquisition_function=self.acquisition_function)
        self.learnable_pars = [
            # "T_ref",
            # "w",
            # "c_t",
            # "V0",
            # "f",
            # "Q",
            # "T_lim_off",
            # "q_lim_off",
            "u_offset",
        ]
        self.learnable_pars_bounds = {
            "T_ref": (73 * np.ones((5,)), 77 * np.ones((5,))),
            "w": (10 * np.ones((1,)), 100 * np.ones((1,))),
            "c_t": (9 * np.ones((1,)), 11 * np.ones((1,))),
            "V0": (-0.1 * np.ones((1,)), 0.1 * np.ones((1,))),
            "f": (-0.002 * np.ones((1 + 7 + 1,)), 0.002 * np.ones((1 + 7 + 1,))),
            "Q": (
                -0.1 * np.ones((1 + 7 + 1, 1 + 7 + 1)),
                0.1 * np.ones((1 + 7 + 1, 1 + 7 + 1)),
            ),
            "T_lim_off": (-0.1 * np.ones((5,)), 0.1 * np.ones((5,))),
            "q_lim_off": (-0.2 * np.ones((1,)), 0.2 * np.ones((1,))),
            "u_offset": (-2 * np.ones((1,)), 2 * np.ones((1,))),
        }
