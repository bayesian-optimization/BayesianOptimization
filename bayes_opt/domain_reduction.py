from typing import Optional, Union, List

import numpy as np
from .target_space import TargetSpace


class DomainTransformer():
    '''The base transformer class'''

    def __init__(self, **kwargs):
        pass

    def initialize(self, target_space: TargetSpace):
        raise NotImplementedError

    def transform(self, target_space: TargetSpace):
        raise NotImplementedError


class SequentialDomainReductionTransformer(DomainTransformer):
    """
    A sequential domain reduction transformer bassed on the work by Stander, N. and Craig, K:
    "On the robustness of a simple domain reduction scheme for simulation‐based optimization"
    """

    def __init__(
        self,
        gamma_osc: float = 0.7,
        gamma_pan: float = 1.0,
        eta: float = 0.9,
        minimum_window: Optional[Union[List[float], float]] = 0.0
    ) -> None:
        self.gamma_osc = gamma_osc
        self.gamma_pan = gamma_pan
        self.eta = eta
        self.minimum_window_value = minimum_window

    def initialize(self, target_space: TargetSpace) -> None:
        """Initialize all of the parameters"""
        self.original_bounds = np.copy(target_space.bounds)
        self.bounds = [self.original_bounds]

        # Set the minimum window to an array of length bounds
        if isinstance(self.minimum_window_value, list) or isinstance(self.minimum_window_value, np.ndarray):
            assert len(self.minimum_window_value) == len(target_space.bounds)
            self.minimum_window = self.minimum_window_value
        else:
            self.minimum_window = [self.minimum_window_value] * len(target_space.bounds)

        self.previous_optimal = np.mean(target_space.bounds, axis=1)
        self.current_optimal = np.mean(target_space.bounds, axis=1)
        self.r = target_space.bounds[:, 1] - target_space.bounds[:, 0]

        self.previous_d = 2.0 * \
            (self.current_optimal - self.previous_optimal) / self.r

        self.current_d = 2.0 * (self.current_optimal -
                                self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d
        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (self.gamma_pan * (1.0 + self.c_hat) +
                            self.gamma_osc * (1.0 - self.c_hat))

        self.contraction_rate = self.eta + \
            np.abs(self.current_d) * (self.gamma - self.eta)

        self.r = self.contraction_rate * self.r

        # check if the minimum window fits in the orignal bounds
        self._window_bounds_compatiblity(self.original_bounds)

    def _update(self, target_space: TargetSpace) -> None:

        # setting the previous
        self.previous_optimal = self.current_optimal
        self.previous_d = self.current_d

        self.current_optimal = target_space.params[
            np.argmax(target_space.target)
        ]

        self.current_d = 2.0 * (self.current_optimal -
                                self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d

        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (self.gamma_pan * (1.0 + self.c_hat) +
                            self.gamma_osc * (1.0 - self.c_hat))

        self.contraction_rate = self.eta + \
            np.abs(self.current_d) * (self.gamma - self.eta)

        self.r = self.contraction_rate * self.r

    def _trim(self, new_bounds: np.array, global_bounds: np.array) -> np.array:
        for i, variable in enumerate(new_bounds):
            if variable[0] < global_bounds[i, 0]:
                variable[0] = global_bounds[i, 0]
            if variable[1] > global_bounds[i, 1]:
                variable[1] = global_bounds[i, 1]
        for i, entry in enumerate(new_bounds):
            if entry[0] > entry[1]:
                new_bounds[i, 0] = entry[1]
                new_bounds[i, 1] = entry[0]
            window_width = abs(entry[0] - entry[1])
            if window_width < self.minimum_window[i]:
                dw = (self.minimum_window[i] - window_width) / 2.0
                left_expansion_space = abs(global_bounds[i, 0] - entry[0]) # should be non-positive
                right_expansion_space = abs(global_bounds[i, 1] - entry[1]) # should be non-negative
                # conservative
                dw_l = min(dw, left_expansion_space)
                dw_r = min(dw, right_expansion_space)
                # this crawls towards the edge
                ddw_r = dw_r + max(dw - dw_l, 0)
                ddw_l = dw_l + max(dw - dw_r, 0)
                new_bounds[i, 0] -= ddw_l
                new_bounds[i, 1] += ddw_r
        return new_bounds

    def _window_bounds_compatiblity(self, global_bounds: np.array) -> bool:
        """Checks if global bounds are compatible with the minimum window sizes."""
        for i, entry in enumerate(global_bounds):
            global_window_width = abs(entry[1] - entry[0])
            if global_window_width < self.minimum_window[i]:
                raise ValueError(
                    "Global bounds are not compatible with the minimum window size.")

    def _create_bounds(self, parameters: dict, bounds: np.array) -> dict:
        return {param: bounds[i, :] for i, param in enumerate(parameters)}

    def transform(self, target_space: TargetSpace) -> dict:

        self._update(target_space)

        new_bounds = np.array(
            [
                self.current_optimal - 0.5 * self.r,
                self.current_optimal + 0.5 * self.r
            ]
        ).T

        self._trim(new_bounds, self.original_bounds)
        self.bounds.append(new_bounds)
        return self._create_bounds(target_space.keys, new_bounds)
