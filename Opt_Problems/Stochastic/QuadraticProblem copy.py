import numpy as np
from Opt_Problems.Base import Problem
from Opt_Problems.Deterministic.QuadraticProblem import Quadratic
from Opt_Problems.Utilities import generate_quadratic_problem, generate_linear_constraints, create_rng
from Opt_Problems.Options import StochasticApproximationType

class QuadraticStochastic(Quadratic):

    def __init__(self, d: int, xi: int, number_of_datapoints: int, seed: int = None, eq_contraints = 0, ineq_contraints = 0) -> None:

        '''Call super class'''
        Problem.__init__(self, name="Quadratic Stochastic", d=d, eq_const_number=eq_contraints, ineq_const_number=ineq_contraints, number_of_datapoints=number_of_datapoints)

        rng = create_rng(seed)

        '''Generate Quadratic Problem'''
        self._A_list = [0]*self.number_of_datapoints
        self._b_list = [0]*self.number_of_datapoints
        for i in range(self.number_of_datapoints):
            self._A_list[i], self._b_list[i] = generate_quadratic_problem(d = self.d, xi=xi, rng=rng)

        '''Generate Equality Constraints'''
        self._A_eq, self._b_eq = generate_linear_constraints(m = self._number_of_eq_constraints, d = self.d, rng=rng)

        '''Generate Inequality Constraints'''
        self._A_ineq, self._b_ineq = generate_linear_constraints(m = self._number_of_ineq_constraints, d = self.d, rng=rng)

    def objective(self, x: np.ndarray, type: StochasticApproximationType, batch_size: int = None, seed: int = None, data_indices: list = None) -> float:

        data_points = super().generate_stochastic_batch(type = type, batch_size= batch_size, seed=seed, data_indices=data_indices)

        total = 0
        for data_index in data_points:
            total += np.dot(x.T, np.dot(self._A_list[data_index], x)) / 2 + np.dot(self._b_list[data_index].T, x)
        return total.squeeze()/len(data_points)
    
    def gradient(self, x: np.ndarray, type: StochasticApproximationType, batch_size: int = None, seed: int = None, data_indices: list = None) -> float:

        data_points = super().generate_stochastic_batch(type = type, batch_size= batch_size, seed=seed, data_indices=data_indices)

        total = 0
        for data_index in data_points:
            total += np.dot(self._A_list[data_index], x) + self._b_list[data_index]
        return total/len(data_points)
    
    def hessian(self, x: np.ndarray, type: StochasticApproximationType, batch_size: int = None, seed: int = None, data_indices: list = None) -> float:

        data_points = super().generate_stochastic_batch(type = type, batch_size= batch_size, seed=seed, data_indices=data_indices)

        total = 0
        for data_index in data_points:
            total += self._A_list[data_index]
        return total/len(data_points)