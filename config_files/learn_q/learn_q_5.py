import numpy as np
from mpcrl import ExperienceReplay, UpdateStrategy
from mpcrl.core.exploration import EpsilonGreedyExploration
from mpcrl.core.schedulers import ExponentialScheduler
from mpcrl.optim import NewtonMethod

from config_files.base import BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.id = "learn_q_5"

        self.use_distance_reward = False
        self.u_offset = -1

        # learning
        self.learn_type = "q_learning"
        self.learning_rate = 1e-3
        self.update_strategy = UpdateStrategy(288, hook="on_timestep_end", skip_first=1)
        self.optimizer = NewtonMethod(learning_rate=self.learning_rate)
        self.experience = ExperienceReplay(
            maxlen=288, sample_size=288, include_latest=0
        )
        self.exploration = None
        self.learnable_pars = [
            "T_ref",
            "w",
            "c_t",
            "V0",
            "f",
            "Q",
            "T_lim_off",
            "q_lim_off",
        ]
