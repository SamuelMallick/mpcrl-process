import numpy as np
from mpcrl import ExperienceReplay, UpdateStrategy
from mpcrl.core.exploration import (EpsilonGreedyExploration,
                                    OrnsteinUhlenbeckExploration)
from mpcrl.core.schedulers import ExponentialScheduler
from mpcrl.optim import NewtonMethod

from config_files.base import BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.id = "learn_dpg"

        self.layers_path = "mpc/prediction_model/layers_low.mat"
        self.input_scaler_path = "mpc/prediction_model/input_scaler_low.mat"
        self.output_scaler_path = "mpc/prediction_model/output_scaler_low.mat"

        # learning
        self.ddpg = True
        self.learning_rate = 1e-1
        self.update_strategy = UpdateStrategy(288, hook="on_timestep_end", skip_first=1)
        self.optimizer = NewtonMethod(learning_rate=self.learning_rate)
        self.experience = ExperienceReplay(maxlen=3, sample_size=3, include_latest=0)
        self.exploration = OrnsteinUhlenbeckExploration(0.0, 5, mode="additive")
        self.rollout_length = 96
        self.learnable_pars = [
            "T_ref",
            "w",
            "c_t",
            "V0",
            "f",
            # "Q",
            "T_lim_off",
            "q_lim_off",
        ]

        np.random.seed(1)
        load_scale = 0.8 * np.random.random((5, 1)) + 0.6
        self.P_loads *= load_scale
