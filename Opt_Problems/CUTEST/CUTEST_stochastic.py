# TODO : decreasing noise towards optimal

import numpy as np
from numpy.core.multiarray import array as array
from Opt_Problems.utils import Problem, Unconstrained_Problem
import pycutest
from Opt_Problems.CUTEST.CUTEST_deterministic import CUTEST_constrained

# TODO : increasing noise towards optimal
class CUTEST_constrained_stochastic_with_start_point(Problem):
    def __init__(self, name: str, variance):

        """import problem from cutest dataset"""
        self._cutest_object = CUTEST_constrained(name=name)
        super().__init__(name=name, d=self._cutest_object.d, eq_const_number=self._cutest_object.number_of_eq_constraints, ineq_const_number=self._cutest_object.number_of_ineq_constraints, number_of_datapoints=np.inf)

    def print_parameters(self):
        """print parameters of the cutest problem"""
        print(pycutest.problem_properties(self.name))
        print(pycutest.print_available_sif_params(self.name))

    def initial_point(self) -> np.array:
        x_init = self._cutest_object.initial_point() 
        return x_init

    def objective(self, x: np.array) -> float:
        return self._cutest_object.objective(x)

    def gradient(self, x: np.array) -> np.array:
        return self._cutest_object.lagjac(x.squeeze())[0].reshape(-1, 1)

    def hessian(self, x: np.array) -> np.array:
        return self._cutest_object.ihess(x.squeeze())

    def constraints_eq(self, x: np.array) -> np.array:
        return self._cutest_object.objcons(x.squeeze())[1][self._cutest_object.is_eq_cons].reshape(-1, 1)

    def constraints_eq_jacobian(self, x: np.array) -> np.array:
        return self._cutest_object.lagjac(x.squeeze())[1][self._cutest_object.is_eq_cons]

    def constraints_eq_hessian(self, x: np.array) -> np.array:
        hessians = []
        for i in range(self._cutest_object.m):
            if not self.is_ineq[i]:
                hessians.append(self._cutest_object.ihess(x.squeeze(), cons_index=i))

        return np.array(hessians)

    def constraints_ineq(self, x: np.array) -> np.array:
        return self._cutest_object.objcons(x.squeeze())[1][self.is_ineq].reshape(-1, 1)

    def constraints_ineq_jacobian(self, x: np.array) -> np.array:
        return self._cutest_object.lagjac(x.squeeze())[1][self.is_ineq]

    def constraints_ineq_hessian(self, x: np.array) -> np.array:
        hessians = []
        for i in range(self._cutest_object.m):
            if self.is_ineq[i]:
                hessians.append(self._cutest_object.ihess(x.squeeze(), cons_index=i))

        return np.array(hessians)