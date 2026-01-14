import os
from typing import Any

import gymnasium as gym
import numpy as np
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave

from monitoring.mahalanobis_distance import MahalanobisDistance


class DHSSystem(gym.Env[np.ndarray, np.ndarray]):
    cp = 4186  # specific heat capacity of water J/(kg K)

    internal_step_size = 0.01  # seconds

    eta_gen = 0.84  # boiler efficiency
    q_b_min = 2.0  # minimum boiler flow kg/s
    q_r_min = 10  # minimum return flow kg/s

    def __init__(
        self,
        step_size: float,
        sim_data: dict[str, Any],
        monitoring_data_set: np.ndarray | None,
        monitoring_window: int,
        use_distance_reward: bool = True,
        w: float = 20.0,
        u_offset: float = 0,
    ):
        super().__init__()
        self.step_size = step_size
        self.num_internal_steps = int(self.step_size / self.internal_step_size)
        self.time = 0.0
        self.use_distance_reward = use_distance_reward

        if {"P_loads", "elec_price", "T_s_min", "T_r_min"} > sim_data.keys():
            raise ValueError(
                "sim_data must contain 'P_loads', 'elec_price', 'T_s_min', and 'T_r_min' keys."
            )
        self.P_loads, self.elec_price, self.T_s_min, self.T_r_min = (
            sim_data[k] for k in ["P_loads", "elec_price", "T_s_min", "T_r_min"]
        )
        if isinstance(u_offset, float) or isinstance(u_offset, int):
            self.u_offset = u_offset * np.ones((self.elec_price.shape[0],))
        else:
            self.u_offset = u_offset

        if monitoring_data_set is None:
            self.monitoring_distance_calculator = None
            self.monitoring_state_size = 0
        else:
            self.monitoring_distance_calculator = MahalanobisDistance(
                [monitoring_data_set],
                [
                    (
                        np.std(monitoring_data_set, axis=0),
                        np.mean(monitoring_data_set, axis=0),
                    )
                ],
            )
            self.monitoring_state_size = monitoring_data_set.shape[1]
        self.monitoring_window = monitoring_window
        self.observed_data = None

        self.w = w
        if os.name == "nt":
            self.fmu_filename = "simulation_model/dhs_storage_win.fmu"
        else:
            self.fmu_filename = "simulation_model/dhs_storage_linux.fmu"

        model_description = self.reset_fmu()

        # collect the value references for fmu variables
        value_references = {}
        for variable in model_description.modelVariables:
            value_references[variable.name] = variable.valueReference
        input_names = [
            "mfr_stes",
            "T_boiler_ref",
            "P_load1",
            "P_load2",
            "P_load3",
            "P_load4",
            "P_load5",
        ]
        output_names = [
            "Ts_load[1]",  # 0
            "Tr_load[1]",  # 1
            "mfr_load[1]",  # 2
            "Ts_load[2]",  # 3
            "Tr_load[2]",  # 4
            "mfr_load[2]",  # 5
            "Ts_load[3]",  # 6
            "Tr_load[3]",  # 7
            "mfr_load[3]",  # 8
            "Ts_load[4]",  # 9
            "Tr_load[4]",  # 10
            "mfr_load[4]",  # 11
            "Ts_load[5]",  # 12
            "Tr_load[5]",  # 13
            "mfr_load[5]",  # 14
            "T_ret",  # 15
            "mfr_ret",  # 16
            "T_boiler_out",  # 17
            "T_supply",  # 18
            "P_boiler_out",  # 19
            # "mfr_supply",  # 20
            # "T_tes",  # 21
            # "mfr_boiler",  # 22
            # "T_boiler_in",  # 23
        ]
        self.inputs = [value_references[name] for name in input_names]
        self.outputs = [value_references[name] for name in output_names]

    def reset_fmu(self):
        self.time = 0.0
        self.step_counter = 0

        # FMU setup
        model_description = read_model_description(self.fmu_filename)
        unzipdir = extract(self.fmu_filename)
        self.fmu = FMU2Slave(
            guid=model_description.guid,
            unzipDirectory=unzipdir,
            modelIdentifier=model_description.coSimulation.modelIdentifier,
            instanceName="instance1",
        )
        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=0.0)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

        return model_description

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.reset_fmu()

        warm_up_action = np.array([[0.0], [70.0]])
        warm_up_action = np.vstack((warm_up_action, self.P_loads[:, [0]]))
        self.fmu.setReal(self.inputs, list(warm_up_action))
        for _ in range(30000 * 10):
            self.fmu.doStep(
                currentCommunicationPoint=self.time,
                communicationStepSize=self.internal_step_size,
            )
            self.time += self.internal_step_size

        self.y = self.fmu.getReal(self.outputs)
        return np.hstack((np.asarray(self.y), np.zeros(self.monitoring_state_size))), {}

    def get_costs(self, output: np.ndarray) -> float:
        P = output[19]  # boiler power
        T_s = [output[i] for i in [0, 3, 6, 9, 12]]
        T_r = output[15]
        q_r = output[16]
        elec_price = self.elec_price[self.step_counter]
        T_s_min = self.T_s_min[self.step_counter]
        T_r_min = self.T_r_min[self.step_counter]
        economic_cost = (elec_price * (1 / 3600.0) * (P / 1000.0)) / self.eta_gen
        constraint_violation_cost = np.sum(np.maximum(0, T_s_min - T_s)) + np.sum(
            np.maximum(0, self.q_r_min - q_r) + np.maximum(0, T_r_min - T_r)
        )
        return (
            economic_cost * self.step_size,
            constraint_violation_cost * self.w * self.step_size,
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Steps the system."""
        if (
            isinstance(action, float) or action.shape[0] == 1
        ):  # set 0 for storage flow if not provided
            action = np.vstack((0, action))
        elif action.shape[0] == 3:
            action = np.array([action[1] - action[0], action[2]])

        internal_step_counter = 0
        economic_cost, constraint_violation_cost = self.get_costs(self.y)

        if self.monitoring_distance_calculator is not None:
            if self.observed_data is not None:
                dist = self.monitoring_distance_calculator.mahalanobis_distance(
                    self.observed_data, return_all=True
                )
                monitoring_distance = dist[0].item()
                monitoring_state = self.observed_data
                self.observed_data = None
            else:
                monitoring_distance = 0.0
                monitoring_state = np.zeros(self.monitoring_state_size)
        else:
            monitoring_distance = 0.0
            monitoring_state = np.zeros(self.monitoring_state_size)

        if self.use_distance_reward:
            r = monitoring_distance
        else:
            r = economic_cost + constraint_violation_cost

        P_loads = self.P_loads[:, [self.step_counter]]
        elec_price = self.elec_price[self.step_counter]
        T_s_min = self.T_s_min[self.step_counter]
        T_r_min = self.T_r_min[self.step_counter]

        offset = self.u_offset[self.step_counter]
        action[1] += offset

        efficiency = -np.sum(P_loads) / self.y[19]

        u = np.vstack([action, P_loads])
        self.fmu.setReal(self.inputs, list(u))

        while internal_step_counter < self.num_internal_steps:
            self.fmu.doStep(
                currentCommunicationPoint=self.time,
                communicationStepSize=self.internal_step_size,
            )
            self.time += self.internal_step_size

            # at first internal step check storage flow and modify if needed
            if internal_step_counter == 0:
                y_new = self.fmu.getReal(self.outputs)
                if y_new[-2] < self.q_b_min:
                    action[0] = self.q_b_min - y_new[16]
                    print(f"Storage flow modified to {action[0]}")
                    u = np.vstack([action, P_loads])
                    self.fmu.setReal(self.inputs, list(u))
            internal_step_counter += 1

        y_new = self.fmu.getReal(self.outputs)
        info = {
            "P_loads": P_loads,
            "elec_price": elec_price,
            "T_s_min": T_s_min,
            "T_r_min": T_r_min,
            "economic_cost": economic_cost,
            "constraint_violation_cost": constraint_violation_cost,
            "monitoring_distance": np.asarray(monitoring_distance),
            "q_r_min": np.asarray(self.q_r_min),
            "efficiency": efficiency,
        }
        self.step_counter += 1
        self.y = y_new
        return (
            np.hstack((np.asarray(self.y), monitoring_state)),
            r,
            False,
            False,
            info,
        )

    def get_sim_data(self, N: int) -> dict[str, np.ndarray]:
        return {
            "P_loads": self.P_loads[:, self.step_counter : self.step_counter + N],
            "elec_price": self.elec_price[self.step_counter : self.step_counter + N],
            "T_s_min": self.T_s_min[self.step_counter : self.step_counter + N],
            "T_r_min": self.T_r_min[self.step_counter : self.step_counter + N],
        }

    def set_observed_data(self, observed_data: dict[str, np.ndarray]) -> None:
        if len(observed_data["P_loads"]) < self.monitoring_window:
            self.observed_data = None
        else:
            data = np.hstack(
                [
                    np.asarray(observed_data[key])[-self.monitoring_window :]
                    for key in [
                        "efficiency",
                        "economic_cost",
                        "constraint_violation_cost",
                    ]
                ]
            )
            data = np.hstack(
                (
                    data,
                    -np.sum(
                        np.asarray(observed_data["P_loads"])[
                            -self.monitoring_window :, :
                        ],
                        axis=1,
                        keepdims=True,
                    ),
                )
            )
            self.observed_data = np.hstack(
                (np.mean(data, axis=0), np.var(data, axis=0))
            )
