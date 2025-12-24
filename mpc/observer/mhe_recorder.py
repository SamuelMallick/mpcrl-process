from typing import Any, TypeVar

import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Wrapper

SymType = TypeVar("SymType", cs.SX, cs.MX)


class MheRecorder(Wrapper[SymType]):
    def __init__(self, nlp: Nlp[SymType]) -> None:
        super().__init__(nlp)
        self.y_open_loop: list[np.ndarray] = []
        self.y_estimated: list[np.ndarray] = []
        self.action_step: list[np.ndarray] = []
        self.P_loads_estimation_data: list[np.ndarray] = []

    def step(self, *args: Any, **kwds: Any) -> Any:
        y_est, y_ol = self.nlp.step(*args, **kwds)
        self.y_estimated.append(y_est.squeeze())
        self.y_open_loop.append(y_ol.squeeze())
        self.action_step.append(np.asarray(kwds["u"] if "u" in kwds else args[0]))
        return y_est, y_ol

    def update_state(self, *args: Any, **kwds: Any) -> Any:
        sol, _ = self.nlp.update_state(*args, **kwds)
        if sol is not None:
            self.P_loads_estimation_data.append(
                kwds["data"]["P_loads"] if "data" in kwds else args[0]["P_loads"]
            )
