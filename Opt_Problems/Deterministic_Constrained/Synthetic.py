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
        number_of_eq_constraints:int = 0,
        number_of_ineq_constraints:int = 0,
    ) -> None:
        self.x_init = np.reshape(x_init, (d, 1))
        self.func = objective
        self.grad = gradient
        self.hess = hessian
        if objective == None and gradient == None and hessian == None:
            raise Exception("Need atleast one of function, gradient or hessian for the optimization problem")

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
        
        super().__init__(name = name, d = d, eq_const_number=number_of_eq_constraints, ineq_const_number=number_of_ineq_constraints)

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

    def constraints_equality(self, x: np.array) -> np.array:
        if self.eq_const == None:
            raise Exception("eq constraint oracle not available for " + self.name)
        return self.eq_const(x)

    def constraints_inequality(self, x: np.array) -> np.array:
        if self.ineq_const == None:
            raise Exception("ineq constraint oracle not available for " + self.name)
        return self.ineq_const(x)

    def constraints_equality_jacobian(self, x: np.array) -> np.array:
        if self.eq_const == None:
            raise Exception("eq constraint jacobian oracle not available for " + self.name)
        return self.eq_jacobian(x)

    def constraints_inequality_jacobian(self, x: np.array) -> np.array:
        if self.ineq_const == None:
            raise Exception("ineq constraint jacobian oracle not available for " + self.name)
        return self.ineq_jacobian(x)

# TODO : Add constraint specifying or generating ability
# class Quadratic(Constrained):

#     """
#     Generates a quadratic function with the specified constraints:
#     """

#     def __init__(self, d: int, A: np.array, b: np.array, c: np.array):
#         """
#         Inputs :
#         A   :   matrix for quadratic
#         b   :   linear term for quadratic
#         c   :   constant term for wuadratic
#         d   :   dimension of problem
#         """
#         self._A = A
#         self._b = b.reshape(d, 1)
#         self._c = c.reshape(1, 1)
#         super().__init__("Quadratic", d)

#     def initial_point(self) -> np.array:
#         return np.zeros((self._d, 1))

#     def objective(self, x: np.array) -> float:
#         val = self._c + np.dot(x.T, self._b) + 0.5 * np.dot(np.dot(x.T, self._A), x)
#         return val[0, 0]

#     def gradient(self, x: np.array) -> np.array:
#         return self._b + np.dot(self._A, x)

#     def hessian(self, x: np.array) -> np.array:
#         return self._A

#     @classmethod
#     def generate(cls, d: int, seed: int | None = None, xi: int = 2):
#         """Generates a quadratic based on the process in Numerical Experiments in:
#         A. Mokhtari, Q. Ling and A. Ribeiro, "Network Newton Distributed Optimization Methods," in IEEE Transactions on Signal Processing, vol. 65, no. 1, pp. 146-161, 1 Jan.1, 2017, doi: 10.1109/TSP.2016.2617829.

#         Inputs:

#         seed    :   seed for sampling (optional)
#         xi      :   constrols condition number, increase to increase conditon number
#                     (optional, default 2)
#         """

#         # random generator to avoid setting global generator
#         if seed is None:
#             rng = np.random.default_rng(np.random.randint(np.iinfo(np.int16).max, size=1)[0])
#         elif isinstance(seed, int):
#             rng = np.random.default_rng(seed)
#         else:
#             raise Exception("seed must be an enteger if specified")

#         s1 = 10 ** np.arange(xi)
#         s2 = 1 / 10 ** np.arange(xi)
#         if d % 2 == 0:
#             v = np.hstack((rng.choice(s1, size=int(d / 2)), rng.choice(s2, size=int(d / 2))))
#         else:
#             v = np.hstack((rng.choice(s1, size=int(d / 2) + 1), rng.choice(s2, size=int(d / 2))))
#         b = rng.random((d)) * 10 ** (int(xi / 2))
#         # print("Condition number : ", np.linalg.cond(A))
#         return cls(d=d, A=np.diag(v), b=b, c=np.array([0]))