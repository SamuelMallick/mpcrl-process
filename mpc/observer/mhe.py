import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
import sys, os

sys.path.append(os.getcwd())
from mpc.prediction_model.dynamic_model import load_data, model


class Mhe(Mpc):

    def __init__(
        self,
        prediction_horizon,
        layers_path,
        input_scaler_path,
        output_scaler_path,
        open_loop: bool = False,
    ):
        self.open_loop = open_loop
        nlp = Nlp[cs.SX](sym_type="SX")
        Mpc.__init__(self, prediction_horizon=prediction_horizon, nlp=nlp)

        # inputs variables
        T_s = self.parameter("T_s", (1, prediction_horizon))
        P_loads = self.parameter("P_loads", (5, prediction_horizon))

        nx = 30
        ny = 17
        # output variables
        y = self.parameter("y", (ny, prediction_horizon))

        # NN DHS dynamics
        v, _, _ = self.variable("v", (ny, prediction_horizon))  # measurement noise
        x, _, _ = self.variable("x", (nx, prediction_horizon + 1))  # NN internal state
        self.constraint("x_ub", x, "<=", 1)
        self.constraint("x_lb", x, ">=", -1)
        w, _, _ = self.variable("w", (nx, prediction_horizon))  # model disturbance

        self.create_model(
            x,
            cs.vertcat(T_s, P_loads),
            y,
            v,
            w,
            N=prediction_horizon,
            layers_path=layers_path,
            input_scaler_path=input_scaler_path,
            output_scaler_path=output_scaler_path,
        )

        x_0 = self.parameter("x_0", (nx, 1))
        P_x_0 = np.eye(nx)
        P_v = np.eye(ny)
        P_w = np.eye(nx)

        self.create_cost(x, x_0, v, w, P_x_0, P_v, P_w)
        self.initialize_solver()

        self.x_est = np.zeros((30, 1))
        self.x_0 = np.zeros((30, 1))
        self.x_open_loop = np.zeros((30, 1))

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

    def set_model_parameters(self, params: dict):
        for k, v in params.items():
            if k in self.model_pars:
                self.model_pars[k] = v.value

    def create_cost(self, x, x_0, v, w, P_x_0, P_v, P_w):
        self.minimize(
            (x[:, 0] - x_0).T @ P_x_0 @ (x[:, 0] - x_0)
            + cs.trace(v.T @ P_v @ v)
            + cs.trace(w.T @ P_w @ w)
        )

    def create_model(
        self, x, u, y, v, w, N, layers_path, input_scaler_path, output_scaler_path
    ):
        self.layers_dicts, self.input_scaler_dict, self.output_scaler_dict = load_data(
            layers_path, input_scaler_path, output_scaler_path
        )
        layers_dicts_pars = [
            {k: self.parameter(f"{k}_{i}", v.shape) for k, v in layer.items()}
            for i, layer in enumerate(self.layers_dicts)
        ]
        input_scaler_dict_pars = {
            k: self.parameter(
                f"{k}_input", v.shape if isinstance(v, np.ndarray) else (1,)
            )
            for k, v in self.input_scaler_dict.items()
            if k in ["scale", "bias"]
        }
        output_scaler_dict_pars = {
            k: self.parameter(
                f"{k}_output", v.shape if isinstance(v, np.ndarray) else (1,)
            )
            for k, v in self.output_scaler_dict.items()
            if k in ["scale", "bias"]
        }
        self.model_pars = {
            f"{k}_{i}": v
            for i, layer in enumerate(self.layers_dicts)
            for k, v in layer.items()
        }

        self.model_pars.update(
            {
                f"{k}_input": v
                for k, v in self.input_scaler_dict.items()
                if k in ["scale", "bias"]
            }
        )
        self.model_pars.update(
            {
                f"{k}_output": v
                for k, v in self.output_scaler_dict.items()
                if k in ["scale", "bias"]
            }
        )
        for k in range(N):
            o = model(
                x[:, k],
                u[:, k],
                1,
                layers_dicts_pars,
                input_scaler_dict_pars,
                output_scaler_dict_pars,
            )
            self.constraint(f"x_{k}", x[:, k + 1] - o[1], "==", w[:, k])
            self.constraint(f"y_{k}", y[:, k] - o[0], "==", v[:, k])

    def reset(self):
        self.x_est = np.zeros((30, 1))
        self.x_0 = np.zeros((30, 1))
        self.x_open_loop = np.zeros((30, 1))
        return self.x_est

    def step(self, u):
        if isinstance(u, list):
            u = cs.vertcat(*u)
        elif isinstance(u, np.ndarray):
            u = cs.DM(u)
        o = model(
            self.x_est,
            u,
            1,
            self.layers_dicts,
            self.input_scaler_dict,
            self.output_scaler_dict,
        )
        self.x_est = o[1]
        y_est = o[0].full()
        o = model(
            self.x_open_loop,
            u,
            1,
            self.layers_dicts,
            self.input_scaler_dict,
            self.output_scaler_dict,
        )
        self.x_open_loop = o[1]
        y_ol = o[0].full()
        return y_est, y_ol

    def update_state(self, data):
        if "y" not in data or "T_s" not in data or "P_loads" not in data:
            raise ValueError("data must contain 'y', 'T_s', and 'P_loads' keys")

        # open loop step if not enough data for full horizon
        if data["y"].shape[1] < self.prediction_horizon:
            self.step([data["T_s"][:, -1]] + data["P_loads"][:, -1].tolist())
            return None, self.x_est

        data["x_0"] = self.x_0
        for k, v in self.model_pars.items():
            data[k] = v
        sol = self.solve(data)
        if sol.success:
            self.x_est = sol.vals["x"][:, -1]
            self.x_0 = sol.vals["x"][:, 1]  # shifted by 1 for next iteration
        else:
            raise ValueError("MHE infeasible!")
        return sol, self.x_est

    def get_x(self):
        if self.open_loop:
            return self.x_open_loop
        return self.x_est
