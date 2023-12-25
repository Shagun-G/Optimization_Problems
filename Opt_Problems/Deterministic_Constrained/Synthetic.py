from Opt_Problems.Base_classes import Constrained
import numpy as np
from typing import Callable


class Generator(Constrained):
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
        inequality: Callable | None = None,
        inequality_jacobian: Callable | None = None,
        number_of_eq_constraints: int = 0,
        number_of_ineq_constraints: int = 0,
    ) -> None:
        self.x_init = np.reshape(x_init, (d, 1))
        self.func = objective
        self.grad = gradient
        self.hess = hessian
        if objective == None and gradient == None and hessian == None:
            raise Exception(
                "Need atleast one of function, gradient or hessian for the optimization problem"
            )

        self.eq_const = equality
        self.eq_jacobian = equality_jacobian
        self.ineq_const = inequality
        self.ineq_jacobian = inequality_jacobian

        if number_of_eq_constraints > 0:
            if equality == None and equality_jacobian == None:
                raise Exception("Need specifications of the equality constraints")

        if number_of_ineq_constraints > 0:
            if inequality == None and inequality_jacobian == None:
                raise Exception("Need specifications of the inquality constraints")

        super().__init__(
            name=name,
            d=d,
            eq_const_number=number_of_eq_constraints,
            ineq_const_number=number_of_ineq_constraints,
        )

    def initial_point(self) -> np.array:
        if self.x_init.any() == None:
            return np.zeros((self._d, 1))
        return self.x_init

    def objective(self, x: np.array) -> float:
        if self.func == None:
            raise Exception("function oracle not available for " + self.name)
        return self.func(x)

    def gradient(self, x: np.array) -> np.array:
        if self.grad == None:
            raise Exception("gradient oracle not available for " + self.name)
        return self.grad(x)

    def hessian(self, x: np.array) -> np.array:
        if self.hess == None:
            raise Exception("hessian oracle not available for " + self.name)
        return self.hess(x)

    def constraints_eq(self, x: np.array) -> np.array:
        if self.eq_const == None:
            raise Exception("eq constraint oracle not available for " + self.name)
        return self.eq_const(x)

    def constraints_ineq(self, x: np.array) -> np.array:
        if self.ineq_const == None:
            raise Exception("ineq constraint oracle not available for " + self.name)
        return self.ineq_const(x)

    def constraints_eq_jacobian(self, x: np.array) -> np.array:
        if self.eq_const == None:
            raise Exception(
                "eq constraint jacobian oracle not available for " + self.name
            )
        return self.eq_jacobian(x)

    def constraints_ineq_jacobian(self, x: np.array) -> np.array:
        if self.ineq_const == None:
            raise Exception(
                "ineq constraint jacobian oracle not available for " + self.name
            )
        return self.ineq_jacobian(x)
