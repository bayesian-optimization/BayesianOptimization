from typing import Optional, Union, List

import numpy as np
from .target_space import TargetSpace
from warnings import warn


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
    A sequential domain reduction transformer based on the work by Stander, N. and Craig, K:
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
        """Initialize all of the parameters.
        """
    
        # Set the original bounds
        self.original_bounds = np.copy(target_space.bounds)
        self.bounds = [self.original_bounds]

        # Set the minimum window to an array of length bounds
        if isinstance(self.minimum_window_value, list) or isinstance(self.minimum_window_value, np.ndarray):
            assert len(self.minimum_window_value) == len(target_space.bounds)
            self.minimum_window = self.minimum_window_value
        else:
            self.minimum_window = [self.minimum_window_value] * len(target_space.bounds)

        # Set initial values
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

        # check if the minimum window fits in the original bounds
        self._window_bounds_compatibility(self.original_bounds)

    def _update(self, target_space: TargetSpace) -> None:
        """ Updates contraction rate, window size, and window center.
        """
        # setting the previous
        self.previous_optimal = self.current_optimal
        self.previous_d = self.current_d
        
        self.current_optimal = self._windowed_max(target_space)

        self.current_d = 2.0 * (self.current_optimal -
                                self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d

        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (self.gamma_pan * (1.0 + self.c_hat) +
                            self.gamma_osc * (1.0 - self.c_hat))

        self.contraction_rate = self.eta + \
            np.abs(self.current_d) * (self.gamma - self.eta)

        self.r = self.contraction_rate * self.r
        
    def _windowed_max(self, target_space: TargetSpace) -> np.array:
        """Returns the parameters that produce the greatest target value within its bounds.
        """
        # extract the components we need from the target space
        params = np.copy(target_space.params.T)
        target = np.copy(target_space.target)
        bounds = np.copy(target_space.bounds)
        
        # create a mask by checking each params against its bounds
        mask = np.zeros_like(target)
        for n, row in enumerate(params):
            lower_bound = bounds[n, 0]
            upper_bound = bounds[n, 1]
            mask += (row <= lower_bound).astype(int)
            mask += (row >= upper_bound).astype(int)
        mask = mask<1

        # apply the mask
        params = params.T[mask]
        targets = target[mask]

        best_params = params[np.argmax(targets)]

        return best_params

    def _trim(self, new_bounds: np.array, global_bounds: np.array) -> np.array:
        """
        Adjust the new_bounds and verify that they adhere to global_bounds and minimum_window.

        Parameters:
        -----------
        new_bounds : np.array
            The proposed new_bounds that (may) need adjustment.

        global_bounds : np.array
            The maximum allowable bounds for each parameter.

        Returns:
        --------
        new_bounds : np.array
            The adjusted bounds after enforcing constraints.
        """

        #sort bounds
        new_bounds = np.sort(new_bounds)

        # Validate each parameter's bounds against the global_bounds
        for i, pbounds in enumerate(new_bounds):
            # If the lower bound of the parameter is outside the global bounds, reset the lower bound
            if (pbounds[0] < global_bounds[i, 0] or pbounds[0] > global_bounds[i, 1]):
                pbounds[0] = global_bounds[i, 0]
                warn("""Domain Reduction Warning:
                    A parameter's lower bound has exceeded its global limit.
                    The offensive boundary has been reset, but be cautious of optimizer convergence.""")

            # If the upper bound bound of the parameter is outside the global bounds, reset the lower bound
            if (pbounds[1] > global_bounds[i, 1] or pbounds[1] < global_bounds[i, 0]):
                pbounds[1] = global_bounds[i, 1]
                warn("""Domain reduction warning:
                    A parameter's lower bound has exceeded its global limit.
                    The offensive boundary has been reset, but be cautious of optimizer convergence.""")

        # Adjust new_bounds to ensure they respect the minimum window width for each parameter
        for i, pbounds in enumerate(new_bounds):
            current_window_width = abs(pbounds[0] - pbounds[1])

            # If the window width is less than the minimum allowable width, adjust it
            # Note that minimum_window < width of the global bounds one side always has more space than required
            if current_window_width < self.minimum_window[i]:
                width_deficit = (self.minimum_window[i] - current_window_width) / 2.0
                available_left_space = abs(global_bounds[i, 0] - pbounds[0])
                available_right_space = abs(global_bounds[i, 1] - pbounds[1])
                
                # determine how much to expand on the left and right
                expand_left = min(width_deficit, available_left_space)
                expand_right = min(width_deficit, available_right_space)
                
                # calculate the deficit on each side
                expand_left_deficit = width_deficit - expand_left
                expand_right_deficit = width_deficit - expand_right

                # shift the deficit to the side with more space
                adjust_left = expand_left + max(expand_right_deficit, 0)
                adjust_right = expand_right + max(expand_left_deficit, 0)
                
                # adjust the bounds
                pbounds[0] -= adjust_left
                pbounds[1] += adjust_right

        return new_bounds

    def _window_bounds_compatibility(self, global_bounds: np.array) -> bool:
        """Checks if global bounds are compatible with the minimum window sizes.
        """
        for i, entry in enumerate(global_bounds):
            global_window_width = abs(entry[1] - entry[0])
            if global_window_width < self.minimum_window[i]:
                raise ValueError(
                    "Global bounds are not compatible with the minimum window size.")

    def _create_bounds(self, parameters: dict, bounds: np.array) -> dict:
        return {param: bounds[i, :] for i, param in enumerate(parameters)}

    def transform(self, target_space: TargetSpace) -> dict:
        """Reduces the bounds of the target space.
        """
        self._update(target_space)

        new_bounds = np.array(
            [
                self.current_optimal - 0.5 * self.r,
                self.current_optimal + 0.5 * self.r
            ]
        ).T

        new_bounds = self._trim(new_bounds, self.original_bounds)
        self.bounds.append(new_bounds)
        return self._create_bounds(target_space.keys, new_bounds)
