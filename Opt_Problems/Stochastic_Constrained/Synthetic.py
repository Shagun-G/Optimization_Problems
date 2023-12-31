from Opt_Problems.utils import Problem
import numpy as np
from typing import Callable


class Generator(Problem):
    """
    Generates a deterministic unconstrained problem using the specified, function, gradient and hessian oracles in the framework of this repository.
    """

    def __init__(
        self,
        name: str,
        d: int,
        x_init: None | np.ndarray,
        objective: Callable | None = None,
        gradient: Callable | None = None,
        hessian: Callable | None = None,
        equality: Callable | None = None,
        equality_jacobian: Callable | None = None,
        equality_hessian: Callable | None = None,
        inequality: Callable | None = None,
        inequality_jacobian: Callable | None = None,
        inequality_hessian: Callable | None = None,
        number_of_eq_constraints:int = 0,
        number_of_ineq_constraints:int = 0,
        number_of_datapoints: int | None = None
    ) -> None:
        self.x_init = np.reshape(x_init, (d, 1))
        self.func = objective
        self.grad = gradient
        self.hess = hessian
        if objective == None and gradient == None and hessian == None:
            raise Exception("Need atleast one of function, gradient or hessian for the optimization problem")

        self.eq_const = equality
        self.eq_jacobian = equality_jacobian
        self.eq_hessian = equality_hessian
        self.ineq_const = inequality
        self.ineq_jacobian = inequality_jacobian
        self.ineq_hessian = inequality_hessian

        if number_of_eq_constraints > 0:
            if equality == None and equality_jacobian == None and equality_hessian == None:
                raise Exception("Need specifications of the equality constraints")

        if number_of_ineq_constraints > 0:
            if inequality == None and inequality_jacobian == None and equality_hessian == None:
                raise Exception("Need specifications of the inquality constraints")

        if number_of_datapoints is None:
            self._number_of_datapoints = 1
        else:
            self._number_of_datapoints = number_of_datapoints
        
        super().__init__(name = name, d = d, eq_const_number=number_of_eq_constraints, ineq_const_number=number_of_ineq_constraints, number_of_datapoints=number_of_datapoints)

    def initial_point(self) -> np.array:
        if self.x_init.any() == None:
            return np.zeros((self._d, 1))
        return self.x_init

    def objective(self, x: np.array, type: str, batch_size: int = 0, seed: int | None = None) -> float:
        if self.func == None:
            raise Exception("function oracle not available for " + self.name)
        return self.func(x, type, batch_size, seed)

    def gradient(self, x: np.array, type: str, batch_size: int = 0, seed: int | None = None) -> np.array:
        if self.grad == None:
            raise Exception("gradient oracle not available for " + self.name)
        return self.grad(x, type, batch_size, seed)

    def hessian(self, x: np.array, type: str, batch_size: int = 0, seed: int | None = None) -> np.array:
        if self.hess == None:
            raise Exception("hessian oracle not available for " + self.name)
        return self.hess(x, type, batch_size, seed)

    def constraints_eq(self, x: np.array) -> np.array:
        if self.eq_const == None:
            raise Exception("eq constraint oracle not available for " + self.name)
        return self.eq_const(x)

    def constraints_ineq(self, x: np.array) -> np.array:
        if self.ineq_const == None:
            raise Exception("ineq constraint oracle not available for " + self.name)
        return self.ineq_const(x)

    def constraints_eq_jacobian(self, x: np.array) -> np.array:
        if self.eq_jacobian == None:
            raise Exception("eq constraint jacobian oracle not available for " + self.name)
        return self.eq_jacobian(x)

    def constraints_ineq_jacobian(self, x: np.array) -> np.array:
        if self.ineq_jacobian == None:
            raise Exception("ineq constraint jacobian oracle not available for " + self.name)
        return self.ineq_jacobian(x)

    def constraints_eq_hessian(self, x: np.array) -> np.array:
        if self.eq_hessian == None:
            raise Exception("eq constraint hessian oracle not available for " + self.name)
        return self.eq_hessian(x)

    def constraints_ineq_hessian(self, x: np.array) -> np.array:
        if self.ineq_hessian == None:
            raise Exception("ineq constraint hessian oracle not available for " + self.name)
        return self.ineq_hessian(x)