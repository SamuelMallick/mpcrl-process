from typing import Any, TypeVar

import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Wrapper

SymType = TypeVar("SymType", cs.SX, cs.MX)


class MpcRecorder(Wrapper[SymType]):
    def __init__(self, nlp: Nlp[SymType]) -> None:
        super().__init__(nlp)
        self.solver_time: list[float] = []
        self.y_prediction: list[dict[str, np.ndarray]] = []
        self.P_loads: list[np.ndarray] = []

    def solve(self, *args: Any, **kwds: Any) -> Any:
        sol = self.nlp.solve(*args, **kwds)
        self.solver_time.append(sol.stats["t_wall_total"])
        pars_arg = kwds["pars"] if "pars" in kwds else args[0]
        self.P_loads.append(pars_arg["P_loads"])

        pars = {
            **pars_arg,
            **{k: v for k, v in self.pars_init.items() if k not in pars_arg},
        }
        o = self.f_y(T_b_s=sol.vals["T_b_s"], **pars)
        y = np.asarray(o["y"])
        self.y_prediction.append(
            {
                "T_i_s": y[:5, :],
                "T_r": y[5, :],
                "q_r": y[6, :],
            }
        )
        print(f"Solve time: {self.solver_time[-1]:.3f} seconds")
        return sol
