import numpy as np
from Opt_Problems.Base import Problem
from Opt_Problems.Utilities import generate_quadratic_problem, generate_linear_constraints, create_rng

class Quadratic(Problem):

    def __init__(self, d: int, xi: int, seed: int = None, eq_contraints = 0, ineq_contraints = 0) -> None:

        '''Call super class'''
        super().__init__(name="Quadratic", d=d, eq_const_number=eq_contraints, ineq_const_number=ineq_contraints, number_of_datapoints=1)

        rng = create_rng(seed)

        '''Generate Quadratic Problem'''
        self._A, self._b = generate_quadratic_problem(d = self.d, xi=xi, rng=rng)

        '''Generate Equality Constraints'''
        self._A_eq, self._b_eq = generate_linear_constraints(m = self._number_of_eq_constraints, d = self.d, rng=rng)

        '''Generate Inequality Constraints'''
        self._A_ineq, self._b_ineq = generate_linear_constraints(m = self._number_of_ineq_constraints, d = self.d, rng=rng)

    def initial_point(self) -> np.ndarray:
        return np.zeros((self.d, 1))

    def objective(self, x: np.ndarray) -> float:
        return np.dot(x.T, np.dot(self._A, x)).squeeze() / 2 + np.dot(self._b.T, x).squeeze()
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.dot(self._A, x) + self._b
    
    def hessian(self, x: np.ndarray) -> np.ndarray:
        return self._A

    def constraints_eq(self, x: np.ndarray) -> np.ndarray:
        c_eq = super().constraints_eq(x)

        if self._number_of_eq_constraints > 0:
            c_eq = np.dot(self._A_eq, x) - self._b_eq

        return c_eq

    def constraints_ineq(self, x: np.ndarray) -> np.ndarray:
        c_ineq = super().constraints_ineq(x)

        if self._number_of_ineq_constraints > 0:
            c_ineq = np.dot(self._A_ineq, x) - self._b_ineq

        return c_ineq

    def constraints_eq_jacobian(self, x: np.ndarray) -> np.ndarray:
        J_eq = super().constraints_eq_jacobian(x)

        if self._number_of_eq_constraints > 0:
            J_eq = self._A_eq

        return J_eq
    
    def constraints_ineq_jacobian(self, x: np.ndarray) -> np.ndarray:
        J_ineq = super().constraints_ineq_jacobian(x)

        if self._number_of_ineq_constraints > 0:
            J_ineq = self._A_ineq

        return J_ineq
