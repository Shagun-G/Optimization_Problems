from Opt_Problems.Deterministic.QuadraticProblem import Quadratic

problem = Quadratic(d=3, xi=3, seed=100, eq_contraints=2, ineq_contraints=2)
"""Calling all functions"""
print("Name: " + problem.name)
print("Dimension : ", problem.d)
print("At x = ", problem.initial_point())
print("Function: ", problem.objective(x=problem.initial_point()))
print("Gradient : ", problem.gradient(x=problem.initial_point()))
print("c_eq: ", problem.constraints_eq(x=problem.initial_point()))
print("J_eq: ", problem.constraints_eq_jacobian(x=problem.initial_point()))
print("c_ineq : ", problem.constraints_ineq(x=problem.initial_point()))
print("J_ineq : ", problem.constraints_ineq_jacobian(x=problem.initial_point()))
print("Has bound constraints : ", problem.has_bound_constraints)
print("Bounds : ", problem.variable_lower_bounds(), problem.variable_upper_bounds())

from Opt_Problems.Stochastic.QuadraticProblem import QuadraticStochastic
from Opt_Problems.Options import StochasticApproximationType

problem = QuadraticStochastic(d=3, xi=3, seed=100, eq_contraints=2, ineq_contraints=2, number_of_datapoints=1)
"""Calling all functions"""
print("Name: " + problem.name)
print("Dimension : ", problem.d)
print("At x = ", problem.initial_point())
# print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.FullBatch))
# print("Gradient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.FullBatch))
# print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.MiniBatch, batch_size=1, seed=100))
# print("Gradient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.MiniBatch, batch_size=1, seed=100))
print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.SpecifiedIndices, data_indices=[0]))
print("Gradient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.SpecifiedIndices, data_indices=[0]))
print("c_eq: ", problem.constraints_eq(x=problem.initial_point()))
print("J_eq: ", problem.constraints_eq_jacobian(x=problem.initial_point()))
print("c_ineq : ", problem.constraints_ineq(x=problem.initial_point()))
print("J_ineq : ", problem.constraints_ineq_jacobian(x=problem.initial_point()))
print("Has bound constraints : ", problem.has_bound_constraints)
print("Bounds : ", problem.variable_lower_bounds(), problem.variable_upper_bounds())