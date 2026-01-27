import numpy as np
from mpcrl import ExperienceReplay, UpdateStrategy
from mpcrl.core.exploration import EpsilonGreedyExploration
from mpcrl.core.schedulers import ExponentialScheduler
from mpcrl.optim import NewtonMethod

from config_files.base import BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.id = "case_4"

        self.use_distance_reward = False

        self.layers_path = "mpc/prediction_model/layers_full.mat"
        self.input_scaler_path = "mpc/prediction_model/input_scaler_full.mat"
        self.output_scaler_path = "mpc/prediction_model/output_scaler_full.mat"

        # learning
        self.learn_type = "q_learning"
        self.learning_rate = 0

        np.random.seed(1)
        load_scale = 1.2 * np.random.random((5, 1)) + 0.4
        self.P_loads *= load_scale
