import numpy as np
from numpy.core.multiarray import array as array
from Opt_Problems.utils import Problem, Unconstrained_Problem
import pycutest

class CUTEST_constrained(Problem):
    def __init__(self, name: str):
        """import problem from cutest dataset"""
        self._cutest_object = pycutest.import_problem(name, drop_fixed_variables=True)
        if self._cutest_object.m == 0:
            raise Exception("Unconstrainted problem, use CUTEST_unconstrained")
        self.is_ineq = np.array([not value for value in self._cutest_object.is_eq_cons])
        super().__init__(name=name, d=self._cutest_object.n_free, eq_const_number=sum(self._cutest_object.is_eq_cons), ineq_const_number=self._cutest_object.m - sum(self._cutest_object.is_eq_cons))

    def print_parameters(self):
        """print parameters of the cutest problem"""
        print(pycutest.problem_properties(self.name))
        print(pycutest.print_available_sif_params(self.name))

    def initial_point(self) -> np.array:
        x_init = self._cutest_object.x0 
        return x_init.reshape(-1, 1)

    def objective(self, x: np.array) -> float:
        return self._cutest_object.obj(x.squeeze(), gradient=False)

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


class CUTEST_unconstrained(Unconstrained_Problem):
    def __init__(self, name: str):
        """import problem from cutest dataset"""
        self._cutest_object = pycutest.import_problem(name, drop_fixed_variables=True)
        if self._cutest_object.m != 0:
            raise Exception("Contrained problem, use CUTEST_constrained")
        super().__init__(name=name, d=self._cutest_object.n_free)

    def print_parameters(self):
        """print parameters of the cutest problem"""
        print(pycutest.problem_properties(self.name))
        print(pycutest.print_available_sif_params(self.name))

    def initial_point(self) -> np.array:
        x_init = self._cutest_object.x0 
        return x_init.reshape(-1, 1)

    def objective(self, x: np.array) -> float:
        return self._cutest_object.obj(x.squeeze(), gradient=False)

    def gradient(self, x: np.array) -> np.array:
        return self._cutest_object.lagjac(x.squeeze())[0].reshape(-1, 1)

    def hessian(self, x: np.array) -> np.array:
        return self._cutest_object.ihess(x.squeeze())