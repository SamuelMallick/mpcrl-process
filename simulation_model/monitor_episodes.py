from collections import deque
from time import perf_counter
from typing import Any, Deque, Optional, SupportsFloat, TypeVar

import casadi as cs
import numpy as np
import numpy.typing as npt
from gymnasium import Env, Wrapper, utils

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class MonitorEpisodes(
    Wrapper[ObsType, ActType, ObsType, ActType], utils.RecordConstructorArgs
):
    """This wrapper keeps track of observations, actions, rewards, episode lengths, and
    execution times of each episode.

    These are saved in the following fields:
     - observations (:attr:`observations`)
     - actions (:attr:`actions`)
     - costs/rewards (:attr:`rewards`)
     - episode length (:attr:`MonitorEpisodes.episode_lengths`)
     - episode execution time (:attr:`exec_times`)

    that the environment is subject to during the learning process. Note that these are
    effectively saved in each corresponding field only when the episode is done
    (terminated or truncated). This means that if an episode, e.g., the last one, has
    not been terminated or truncated, these fields will not have recorded its data
    (which can be found in the internal attributes).

    Parameters
    ----------
    env : Env[ObsType, ActType]
        The environment to apply the wrapper to.
    deque_size : int, optional
        The maximum number of episodes to hold as historical data in the internal
        deques. By default, `None`, i.e., unlimited.

    Examples
    --------
    After the completion of an episode, these fields will look like this:

    >>> env.observations = <deque of each episode's observations>
    ... env.actions = <deque of each episode's actions>
    ... env.rewards = <deque of each episode's rewards>
    ... env.episode_lengths = <deque of each episode's episode length>
    ... env.exec_times = <deque of each episode's execution time>
    """

    def __init__(
        self, env: Env[ObsType, ActType], deque_size: Optional[int] = None
    ) -> None:
        utils.RecordConstructorArgs.__init__(self, deque_size=deque_size)
        Wrapper.__init__(self, env)
        # long-term storages
        self.observations: Deque[npt.NDArray[ObsType]] = deque(maxlen=deque_size)
        self.actions: Deque[npt.NDArray[ActType]] = deque(maxlen=deque_size)
        self.rewards: Deque[npt.NDArray[np.floating]] = deque(maxlen=deque_size)

        self.extra_data: dict[str, Deque[npt.NDArray[np.floating]]] = {
            "P_loads": deque(maxlen=deque_size),
            "elec_price": deque(maxlen=deque_size),
            "T_s_min": deque(maxlen=deque_size),
            "T_r_min": deque(maxlen=deque_size),
            "economic_cost": deque(maxlen=deque_size),
            "efficiency": deque(maxlen=deque_size),
            "constraint_violation_cost": deque(maxlen=deque_size),
            "monitoring_distance": deque(maxlen=deque_size),
            "q_r_min": deque(maxlen=deque_size),
        }

        self.episode_lengths: Deque[int] = deque(maxlen=deque_size)
        self.exec_times: Deque[float] = deque(maxlen=deque_size)
        # current-episode-storages
        self.ep_observations: list[ObsType] = []
        self.ep_actions: list[ActType] = []
        self.ep_rewards: list[SupportsFloat] = []

        self.ep_extra_data: dict[str, list[SupportsFloat]] = {
            key: [] for key in self.extra_data.keys()
        }

        self.t0: float = perf_counter()
        self.ep_length: int = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        self._clear_ep_data()
        self.ep_observations.append(observation)
        return observation, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.unwrapped.set_observed_data(
            {**self.ep_extra_data, "y": self.ep_observations}
        )
        obs, reward, terminated, truncated, info = super().step(action)

        # accumulate data
        self.ep_observations.append(obs)
        if isinstance(action, cs.DM):
            action = action.full()
        self.ep_actions.append(
            action.reshape(
                -1,
            )
        )
        self.ep_rewards.append(reward)

        for key in self.extra_data.keys():
            val = info.get(key, np.zeros(1)).reshape(
                -1,
            )
            self.ep_extra_data[key].append(val)

        self.ep_length += 1

        # if episode is done, save the current data to history
        if terminated or truncated:
            self.force_episode_end()

        return obs, reward, terminated, truncated, info

    def force_episode_end(self) -> None:
        """Appends all the accumulated data from the current/last episode to the main
        deques (as would happen if the episode ended) and clears the current episode's
        data."""
        # append data
        self.observations.append(np.asarray(self.ep_observations))
        self.actions.append(np.asarray(self.ep_actions))
        self.rewards.append(np.asarray(self.ep_rewards))

        for key, deque_attr in self.extra_data.items():
            deque_attr.append(np.asarray(self.ep_extra_data[key]))

        self.episode_lengths.append(self.ep_length)
        self.exec_times.append(perf_counter() - self.t0)

        # clear this episode's data
        self._clear_ep_data()

    def _clear_ep_data(self) -> None:
        # clear this episode's lists and reset counters
        self.ep_observations.clear()
        self.ep_actions.clear()
        self.ep_rewards.clear()

        for key in self.extra_data.keys():
            self.ep_extra_data[key].clear()

        self.t0 = perf_counter()
        self.ep_length = 0
