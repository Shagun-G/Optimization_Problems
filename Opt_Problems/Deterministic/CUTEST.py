from Opt_Problems.Base import Problem
from Opt_Problems.Deterministic.S2MPJ_support_files.cutest_parameters import CUTEST_PARAMETERS
import numpy as np
import importlib

class CUTEST(Problem):

    '''Uses the s2mpj framework and it's corresponing files are stored in the s2mpj folder.'''

    def __init__(self, name: str) -> None:
        problem = importlib.import_module("Opt_Problems.Deterministic.S2MPJ_support_files.cutest_python_problems." + name)
        problem = getattr(problem, name)
        invals = getattr(CUTEST_PARAMETERS, name).value
        exec("self._cutest_problem = problem(" + invals + ")")
        if self._cutest_problem.m == 0: 
            m_eq = 0
            m_ineq = 0
        else:
            m_eq = self._cutest_problem.neq
            m_ineq = self._cutest_problem.nle + self._cutest_problem.nge
        super().__init__(name=name, d = self._cutest_problem.n, eq_const_number=m_eq, ineq_const_number=m_ineq, number_of_datapoints=1)

        # scaling factor
        _, scaling_factor = self._cutest_problem.fgx(self._cutest_problem.x0)
        scaling_factor = np.linalg.norm(scaling_factor)
        if scaling_factor < 1e-10:
            self._scaling_factor = 1
        else:
            self._scaling_factor = 1/scaling_factor


    def initial_point(self) -> np.array:
        return self._cutest_problem.x0

    def objective(self, x: np.ndarray) -> float:
        f = self._cutest_problem.fx(x)
        return f*self._scaling_factor

    def gradient(self, x: np.ndarray) -> np.ndarray:
        _, g = self._cutest_problem.fgx(x)
        return g*self._scaling_factor

    def hessian(self, x: np.ndarray) -> np.ndarray:
        _, _, H = self._cutest_problem.fgHx(x)
        return H*self._scaling_factor

    def constraints_eq(self, x: np.ndarray) -> np.ndarray:
        c_eq = super().constraints_eq(x)
        if self.number_of_eq_constraints > 0:   
            c_eq = np.vstack([c_eq, self._cutest_problem.cIx(x, range(self._cutest_problem.nle, self._cutest_problem.nle + self._cutest_problem.neq))])

        return c_eq

    def constraints_eq_jacobian(self, x: np.ndarray) -> np.ndarray:
        J_eq = super().constraints_eq_jacobian(x)
        if self.number_of_eq_constraints > 0:   
            J_eq = np.vstack([J_eq, np.array(self._cutest_problem.cIJx(x, range(self._cutest_problem.nle, self._cutest_problem.nle + self._cutest_problem.neq))[1].todense())])

        return J_eq

    def constraints_ineq(self, x: np.ndarray) -> np.ndarray:
        c_ineq = super().constraints_ineq(x)
        if self.number_of_ineq_constraints > 0:   
            if self._cutest_problem.nle > 0:
                c_ineq = np.vstack([c_ineq, self._cutest_problem.cIx(x, range(self._cutest_problem.nle))])
            if self._cutest_problem.nge > 0:
                c_ineq = np.vstack([c_ineq, -self._cutest_problem.cIx(x, range(self._cutest_problem.nle + self._cutest_problem.neq, self._cutest_problem.nle + self._cutest_problem.neq + self._cutest_problem.nge))])

        return c_ineq

    def constraints_ineq_jacobian(self, x: np.ndarray) -> np.ndarray:
        J_ineq = super().constraints_ineq_jacobian(x)
        if self.number_of_ineq_constraints > 0:   
            if self._cutest_problem.nle > 0:
                J_ineq = np.vstack([J_ineq, np.array(self._cutest_problem.cIJx(x, range(self._cutest_problem.nle))[1].todense())])
            if self._cutest_problem.nge > 0:
                J_ineq = np.vstack([J_ineq, -np.array(self._cutest_problem.cIJx(x, range(self._cutest_problem.nle + self._cutest_problem.neq, self._cutest_problem.nle + self._cutest_problem.neq + self._cutest_problem.nge))[1].todense())])

        return J_ineq

    def variable_lower_bounds(self) -> np.ndarray:
        return self._cutest_problem.xlower

    def varable_upper_bounds(self) -> np.ndarray:
        return self._cutest_problem.xupper

    def real_problem(self) -> bool:

        classification = self._cutest_problem.pbclass
        classification_parts = classification.split('-')
        char_after_first_hyphen = classification_parts[1][0]

        return char_after_first_hyphen == 'R'

