from config_files.base import BaseConfig
import numpy as np
import matplotlib.pyplot as plt


class Config(BaseConfig):

    def __init__(self):
        super().__init__()
        self.id = "generate_data"

        self.sim_len = 288 * 3

        self.P_loads, self.elec_price, self.T_s_min, self.T_r_min = (
            self.generate_sim_data(self.sim_len, self.loads_folder)
        )

        np.random.seed(1)
        for i in range(288, self.sim_len, 288):
            load_scale = 0.8 * np.random.random((5, 1)) + 0.6
            self.P_loads[:, i : i + 288] *= load_scale
