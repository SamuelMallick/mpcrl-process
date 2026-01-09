from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import torch
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from mpcrl.optim import GradientFreeOptimizer
from scipy.stats.qmc import LatinHypercube


class BoTorchOptimizer(GradientFreeOptimizer):
    """Implements a Bayesian Optimization optimizer based on BoTorch."""

    prefers_dict = False  # ask-and-tell methods should receive arrays, not dicts

    def __init__(
        self,
        initial_random: int = 5,
        initial_points: list[np.ndarray] = [],
        acquisition_function: str = "lei",
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the optimizer.

        Parameters
        ----------
        initial_random : int, optional
            Number of initial random guesses, by default ``5``. Must be positive.
        initial_points : list[np.ndarray], optional
            List of points to use in place of random initial guesses, by default ``[]``.
        seed : int, optional
            Seed for the random number generator, by default ``None``.
        """
        if initial_random <= 0:
            raise ValueError("`initial_random` must be positive.")
        super().__init__(**kwargs)
        self._initial_random = initial_random
        self._initial_points = initial_points
        self._acquisition_function = acquisition_function
        self._seed = seed

    def _init_update_solver(self) -> None:
        # compute the current bounds on the learnable parameters
        pars = self.learnable_parameters
        values = pars.value
        lb, ub = (values + bnd for bnd in self._get_update_bounds(values))

        # use latin hypercube sampling to generate the initial random guesses
        lhs = LatinHypercube(pars.size, seed=self._seed)
        self._train_inputs = (
            lhs.random(self._initial_random - len(self._initial_points)) * (ub - lb)
            + lb
        )
        if len(self._initial_points) > 0:
            self._train_inputs = np.vstack((self._initial_points, self._train_inputs))
        self._train_targets = np.empty((0,))  # we dont know the targets yet
        self._n_ask = -1  # to track the number of ask iterations

    def ask(self) -> tuple[npt.NDArray[np.floating], None]:
        self._n_ask += 1

        # if still in the initial random phase, just return the next random guess
        if self._n_ask < self._initial_random:
            return self._train_inputs[self._n_ask], None

        # otherwise, use BO to find the next guess
        # prepare data for fitting GP
        train_inputs = torch.from_numpy(self._train_inputs)
        train_targets = standardize(torch.from_numpy(self._train_targets).unsqueeze(-1))

        # fit the GP
        values = self.learnable_parameters.value
        bounds = torch.from_numpy(
            np.stack([values + bnd for bnd in self._get_update_bounds(values)])
        )
        normalize = Normalize(train_inputs.shape[-1], bounds=bounds)
        gp = SingleTaskGP(train_inputs, train_targets, input_transform=normalize)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))

        # maximize the acquisition function to get the next guess
        if self._acquisition_function == "lei":
            af = LogExpectedImprovement(gp, train_targets.amin(), maximize=False)
        elif self._acquisition_function == "ucb":
            af = UpperConfidenceBound(gp, train_targets.amin(), maximize=False)
        else:
            raise ValueError(
                f"Acquisition function '{self._acquisition_function}' not recognized."
            )
        seed = self._seed + self._n_ask
        acqfun_optimizer = (
            optimize_acqf(af, bounds, 1, 16, 64, {"seed": seed})[0].numpy().reshape(-1)
        )
        return acqfun_optimizer, None

    def tell(self, values: npt.NDArray[np.floating], objective: float) -> None:
        iteration = self._n_ask
        if iteration < 0:
            raise RuntimeError("`ask` must be called before `tell`.")

        # append the new datum to the training data
        if iteration < self._initial_random:
            assert (
                values == self._train_inputs[iteration]
            ).all(), "`tell` called with a different value than the one given by `ask`."
            self._train_inputs[iteration] = values
        else:
            self._train_inputs = np.append(
                self._train_inputs, values.reshape(1, -1), axis=0
            )
        self._train_targets = np.append(self._train_targets, objective)
