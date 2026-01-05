from mpcrl import ExperienceReplay, UpdateStrategy
from mpcrl.optim import NewtonMethod
from mpcrl.core.exploration import EpsilonGreedyExploration
from mpcrl.core.schedulers import ExponentialScheduler
from config_files.base import BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.id = "learn_dpg"

        self.layers_path = "mpc/prediction_model/layers_low.mat"
        self.input_scaler_path = "mpc/prediction_model/input_scaler_low.mat"
        self.output_scaler_path = "mpc/prediction_model/output_scaler_low.mat"

        # learning
        self.learning_rate = 5e-3
        self.update_strategy = UpdateStrategy(288, hook="on_timestep_end", skip_first=1)
        self.optimizer = NewtonMethod(learning_rate=5e-3)
        self.experience = ExperienceReplay(
            maxlen=288, sample_size=288, include_latest=0
        )
        self.exploration = EpsilonGreedyExploration(
            epsilon=ExponentialScheduler(0.99, 0.928),
            strength=25,
            hook="on_update",
            mode="gradient-based",
        )
        self.rollout_length = 100
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
