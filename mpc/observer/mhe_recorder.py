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

    def step(self, *args: Any, **kwds: Any) -> Any:
        y_est, y_ol = self.nlp.step(*args, **kwds)
        self.y_estimated.append(y_est.squeeze())
        self.y_open_loop.append(y_ol.squeeze())
        return y_est, y_ol
