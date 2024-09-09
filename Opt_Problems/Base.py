from abc import ABC
import numpy as np
from Opt_Problems.Options import StochasticApproximationType
from Opt_Problems.Utilities import create_rng, generate_stochastic_batch


class Problem(ABC):
    """
    Defines structure of problems.

    All problems defined in this class have:

    1. Single objective function
    2. Continuous Variables
    3. Use numpy arrays for input and output
    4. Only Deterministic Constraints Currently

    All inequality constraints are of the form c(x) <= 0
    """

    def __init__(
        self,
        name: str,
        d: int,
        eq_const_number: int = 0,
        ineq_const_number: int = 0,
        number_of_datapoints: int = 1
    ) -> None:
        self._name = name
        self._d = d
        self._number_of_eq_constraints = eq_const_number
        self._number_of_ineq_constraints = ineq_const_number
        self._number_of_datapoints = number_of_datapoints 

    @property
    def name(self) -> str:
        return self._name

    @property
    def d(self) -> int:
        return self._d

    @property
    def number_of_eq_constraints(self) -> int:
        return self._number_of_eq_constraints

    @property
    def number_of_ineq_constraints(self) -> int:
        return self._number_of_ineq_constraints

    @property
    def number_of_datapoints(self) -> int:
        return self._number_of_datapoints

    @property
    def has_bound_constraints(self) -> bool:
        if (self.variable_lower_bounds() > -1e20).any():
            return True
        if (self.variable_upper_bounds() < 1e20).any():
            return True

        return False

    def initial_point(self) -> np.array:
        raise NotImplementedError

    def objective(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def hessian(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def constraints_eq(self, x: np.ndarray) -> np.ndarray:
        return np.empty((0, 1))

    def constraints_eq_jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.empty((0, self.d))

    def constraints_eq_hessian(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def constraints_ineq(self, x: np.ndarray) -> np.ndarray:
        return np.empty((0, 1))

    def constraints_ineq_jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.empty((0, self.d))

    def constraints_ineq_hessian(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def variable_upper_bounds(self) -> np.ndarray:
        return np.zeros((self.d, 1)) + float('inf')

    def variable_lower_bounds(self) -> np.ndarray:
        return np.zeros((self.d, 1)) - float('inf')

    '''Helper Functions'''

    def generate_stochastic_batch(self, type: StochasticApproximationType, batch_size: int, seed: int, data_indices: list) -> None:

        if type is StochasticApproximationType.FullBatch:
            data_indices = np.arange(0, self.number_of_datapoints)

        elif type is StochasticApproximationType.SpecifiedIndices:
            if data_indices is None:
                raise ValueError("data_indices cannot be None if type is {}".format(type.value))
        
        elif type is StochasticApproximationType.MiniBatch:
            if batch_size is None:
                raise ValueError("batch_size cannot be None if type is {}".format(type.value))

            if batch_size > self.number_of_datapoints:
                raise Exception("Batch size specified is larger than size of dataset")
            rng = create_rng(seed)        
            data_indices = generate_stochastic_batch(n=self.number_of_datapoints, batch_size=batch_size, rng=rng)
        else:
            raise Exception("Unknown type {}".format(type.value))

        return data_indices