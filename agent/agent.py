from mpcrl import Agent, LstdDpgAgent, LstdQLearningAgent
import numpy as np
import sys, os

from misc.save_data import save_simulation_data
from mpc.observer.mhe import Mhe
from simulation_model.env import DHSSystem


class DhsAgent(Agent):
    save_frequency = 0

    def __init__(
        self,
        *args,
        observer: Mhe,
        save_frequency: int = 0,
        save_location: str = "",
        **kwargs,
    ):
        self.observer = observer
        self.save_frequency = save_frequency
        self.save_location = save_location
        super().__init__(*args, **kwargs)

    def on_episode_start(self, env: DHSSystem, episode, state):
        env = env.unwrapped  # TODO is there a more elegant solution?
        self.observer.reset()
        self.fixed_parameters["x"] = self.observer.get_x()
        sim_data = env.get_sim_data(self.V.prediction_horizon)
        for key, value in sim_data.items():
            self.fixed_parameters[key] = value
        self.time_step = 0

        return super().on_episode_start(env, episode, state)

    def on_env_step(self, env: DHSSystem, episode, timestep):
        self.time_step += 1

        if self.time_step % self.save_frequency == 0 and self.save_frequency > 0:
            save_simulation_data(
                f"{self.save_location}_step{self.time_step}",
                env,
                self.V,
                self.observer,
                episode_in_progress=True,
            )

        # handle observer update
        if env.ep_observations:
            mhe_data = {
                "P_loads": np.asarray(
                    env.ep_extra_data["P_loads"][-self.observer.prediction_horizon :]
                ).T,
                "T_s": np.asarray(
                    env.ep_observations[-self.observer.prediction_horizon - 1 : -1]
                )[:, [18]].T,
                "y": np.asarray(
                    env.ep_observations[-self.observer.prediction_horizon - 1 : -1]
                )[:, :17].T,
            }
        else:  # episode over
            mhe_data = {  # dummy data
                "P_loads": np.zeros((5, 1)),
                "y": np.zeros((17, 1)),
                "T_s": np.zeros((1, 1)),
            }
        self.observer.update_state(mhe_data)
        self.fixed_parameters["x"] = self.observer.get_x()

        env = env.unwrapped  # TODO is there a more elegant solution?
        sim_data = env.get_sim_data(self.V.prediction_horizon)
        for key, value in sim_data.items():
            self.fixed_parameters[key] = value

        return super().on_env_step(env, episode, timestep)

    def state_value(
        self, state, deterministic=False, vals0=None, action_space=None, **kwargs
    ):
        action, sol = super().state_value(
            state, deterministic, vals0, action_space, **kwargs
        )
        self.observer.step(
            np.vstack((action.full(), self.fixed_parameters["P_loads"][:, [0]]))
        )
        return action, sol


class DhsDpgAgent(DhsAgent, LstdDpgAgent):

    def __init__(
        self,
        *args,
        observer: Mhe,
        save_frequency: int = 0,
        save_location: str = "",
        **kwargs,
    ):
        self.observer = observer
        self.save_frequency = save_frequency
        self.save_location = save_location
        LstdDpgAgent.__init__(self, *args, **kwargs)


class DhsQLearningAgent(DhsAgent, LstdQLearningAgent):

    def __init__(
        self,
        *args,
        observer: Mhe,
        save_frequency: int = 0,
        save_location: str = "",
        **kwargs,
    ):
        self.observer = observer
        self.save_frequency = save_frequency
        self.save_location = save_location
        LstdQLearningAgent.__init__(self, *args, **kwargs)
