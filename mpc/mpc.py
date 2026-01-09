import os
import sys
from typing import override

import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp.multistart.multistart_nlp import ParallelMultistartNlp
from csnlp.wrappers import Mpc

sys.path.append(os.getcwd())
from mpc.prediction_model.dynamic_model import load_data, model


class DhsMpc(Mpc):
    delta_u_lim = 5
    q_min = 10
    q_max = 25
    P_b_min = 1e3
    P_b_max = 10e6
    T_s_max = 85
    T_r_max = 75
    T_b_min = 65

    cp = 4186  # specific heat capacity of water J/(kg K)
    eta = 0.84  # efficiency of boiler from Lorenzo paper

    parameters = {}

    def __init__(
        self,
        dt: float,
        prediction_horizon: int,
        layers_path: str,
        input_scaler_path: str,
        output_scaler_path: str,
        pars_init: dict,
        num_inputs: int = 0,
        gamma: float = 1.0,
    ):
        nlp = Nlp[cs.SX](sym_type="SX")
        input_spacing = int(cs.ceil(prediction_horizon / num_inputs))
        super().__init__(nlp, prediction_horizon, input_spacing=input_spacing)

        # artificially adding states for dpg value
        self._initial_states["y0"] = self.parameter("y0", (28,))

        self.dt = dt
        self.pars_init = pars_init

        # parameters that define MPC problem
        T_ref = self.parameter("T_ref", (5,))  # reference for load temps
        w = self.parameter("w", (1,))  # weight for slacks in cost
        c_t = self.parameter(
            "c_t", (1,)
        )  # weight for load temp penalty tracking in cost
        V0 = self.parameter("V0", (1,))  # cost offset term
        f = self.parameter(
            "f", (1 + 7 + 1,)
        )  # linear cost term (1 input, 7 NN outputs, P_b)
        Q = self.parameter("Q", (1 + 7 + 1, 1 + 7 + 1))  # quadratic cost term
        T_lim_off = self.parameter("T_lim_off", (5,))  # temperature limit offset
        q_lim_off = self.parameter("q_lim_off", (1,))  # mass flow limit offset
        u_offset = self.parameter("u_offset", (1,))  # input offset

        # input variables
        u_, u = self.action("T_b_s", 1, lb=self.T_b_min, ub=self.T_s_max)
        u = u + u_offset

        # NN variables and parameters
        P_loads = self.parameter("P_loads", (5, prediction_horizon))
        x = (
            self.parameter("x", (30, 1))
            if type(layers_path) == str
            else self.parameter("x", (30, len(layers_path)))
        )
        # system dynamics
        o = self.create_model(
            x,
            cs.vertcat(u, P_loads),
            prediction_horizon,
            layers_path,
            input_scaler_path,
            output_scaler_path,
        )
        y = o[0]
        T_i_s = y[:5, :]
        T_r = y[5, :]
        q_r = y[6, :]

        # boiler power
        P_b = self.cp * q_r * (u - T_r)

        # limits
        T_s_min = self.parameter("T_s_min", (1, prediction_horizon))
        T_r_min = self.parameter("T_r_min", (1, prediction_horizon))
        s_T, _, _ = self.variable(
            "s_T", (5, 1), lb=0
        )  # slacks for soft constraints on load temperature
        self.constraint(
            "T_s_min", T_i_s + s_T + T_lim_off, ">=", cs.repmat(T_s_min, 5, 1)
        )
        self.constraint("T_s_max", T_i_s, "<=", self.T_s_max + s_T)

        s_mfr, _, _ = self.variable("s_mfr", (1, 1), lb=0)
        self.constraint("q_r_min", q_r + s_mfr + q_lim_off, ">=", self.q_min)
        self.constraint("q_r_max", q_r - s_mfr, "<=", self.q_max)

        self.constraint("T_r_min", T_r, ">=", T_r_min)
        self.constraint("T_r_max", T_r, "<=", self.T_r_max)

        self.constraint("P_b_min", P_b, ">=", self.P_b_min)
        self.constraint("P_b_max", P_b, "<=", self.P_b_max)

        # cost
        elec_price = self.parameter("elec_price", (1, prediction_horizon))
        gammapowers = cs.DM(gamma ** np.arange(prediction_horizon)).T
        vars = cs.vertcat(u, y, P_b)
        self.minimize(
            V0
            + ((gammapowers * elec_price * (self.dt / 3600.0)) @ (P_b.T / 1000.0))
            / self.eta  # /1000 for kW price and dt/3600 for hours
            + gammapowers[-1] * c_t * cs.sum1((T_i_s[:, -1] - T_ref) ** 2)
            + w * cs.sum1(cs.sum2(s_T))
            + w * cs.sum1(cs.sum2(s_mfr))
            + cs.sum2(f.T @ vars)
            + cs.trace(vars.T @ Q @ vars)
        )

        # create expressions for getting internal variables from solution
        self.f_y = cs.Function(
            "F_y",
            [u_] + list(self.parameters.values()),
            [y],
            ["T_b_s"] + list(self.parameters.keys()),
            ["y"],
        )

        self.initialize_solver()

    def create_model(
        self,
        x,
        u,
        N,
        layers_path,
        input_scaler_path,
        output_scaler_path,
    ):
        layers_dicts, input_scaler_dict, output_scaler_dict = load_data(
            layers_path, input_scaler_path, output_scaler_path
        )
        layers_dicts_pars = [
            {k: self.parameter(f"{k}_{i}", v.shape) for k, v in layer.items()}
            for i, layer in enumerate(layers_dicts)
        ]
        input_scaler_dict_pars = {
            k: self.parameter(
                f"{k}_input", v.shape if isinstance(v, np.ndarray) else (1,)
            )
            for k, v in input_scaler_dict.items()
            if k in ["scale", "bias"]
        }
        output_scaler_dict_pars = {
            k: self.parameter(
                f"{k}_output", v.shape if isinstance(v, np.ndarray) else (1,)
            )
            for k, v in output_scaler_dict.items()
            if k in ["scale", "bias"]
        }
        self.pars_init.update(
            {
                f"{k}_{i}": v
                for i, layer in enumerate(layers_dicts)
                for k, v in layer.items()
            }
        )
        self.pars_init.update(
            {
                f"{k}_input": v
                for k, v in input_scaler_dict.items()
                if k in ["scale", "bias"]
            }
        )
        self.pars_init.update(
            {
                f"{k}_output": v
                for k, v in output_scaler_dict.items()
                if k in ["scale", "bias"]
            }
        )
        return model(
            x,
            u,
            N,
            layers_dicts_pars,
            input_scaler_dict_pars,
            output_scaler_dict_pars,
            which_outputs=[0, 3, 6, 9, 12, 15, 16],  # T_i_s, T_r, q_r
        )

    def initialize_solver(self):
        # solver
        linear_solver = "mumps"  #  if os.name == "nt" else "ma57"
        opts = {
            "expand": True,
            "show_eval_warnings": False,
            "warn_initial_bounds": True,
            "print_time": False,
            "record_time": True,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": False,
            "ipopt": {
                "sb": "yes",
                "print_level": 1,
                "max_iter": 20000,
                "print_user_options": "yes",
                "print_options_documentation": "no",
                "linear_solver": linear_solver,  # spral
                "nlp_scaling_method": "gradient-based",
                "nlp_scaling_max_gradient": 10,  # TODO should this be tuned?
            },
        }
        self.init_solver(opts, solver="ipopt")

    def parameter(self, name, shape):
        param = self.nlp.parameter(name, shape)
        self.parameters[name] = param
        return param

    def solve(
        self,
        pars,
        vals0,
    ):
        pars = {
            **pars,
            **{k: v for k, v in self.pars_init.items() if k not in pars},
        }
        return self.nlp.solve(pars, vals0)
