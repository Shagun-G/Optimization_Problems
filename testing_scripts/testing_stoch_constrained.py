import numpy as np
from Opt_Problems.Stochastic_Constrained.Synthetic import Quadratic_linear_norm_constraints, Quadratic_linear_constraints, Quadratic_norm_constraints

problem = Quadratic_linear_norm_constraints(d=2, n_quadratics=10, seed=1000, xi=6, m = 2)

"""Calling all functions"""
print("Name: " + problem.name)
print("Dimension : ", problem.d)
print("At x = ", problem.initial_point())
# print("Function: ", problem.objective(x=problem.initial_point()))
# print("Gradient : ", problem.gradient(x=problem.initial_point()))
# print("Hessian: ", problem.hessian(x=problem.initial_point()))
print(problem.constraints_eq(x = problem.initial_point()))
print(problem.constraints_eq_jacobian(x = problem.initial_point()))
print("------------------------------------------")


problem = Quadratic_norm_constraints(d=2, n_quadratics=10, seed=1000, xi=6)

"""Calling all functions"""
print("Name: " + problem.name)
print("Dimension : ", problem.d)
print("At x = ", problem.initial_point())
# print("Function: ", problem.objective(x=problem.initial_point()))
# print("Gradient : ", problem.gradient(x=problem.initial_point()))
# print("Hessian: ", problem.hessian(x=problem.initial_point()))
print(problem.constraints_eq(x = problem.initial_point()))
print(problem.constraints_eq_jacobian(x = problem.initial_point()))
print("------------------------------------------")


problem = Quadratic_linear_constraints(d=2, n_quadratics=10, seed=1000, xi=6, m = 2)

"""Calling all functions"""
print("Name: " + problem.name)
print("Dimension : ", problem.d)
print("At x = ", problem.initial_point())
# print("Function: ", problem.objective(x=problem.initial_point()))
# print("Gradient : ", problem.gradient(x=problem.initial_point()))
# print("Hessian: ", problem.hessian(x=problem.initial_point()))
print(problem.constraints_eq(x = problem.initial_point()))
print(problem.constraints_eq_jacobian(x = problem.initial_point()))
print("------------------------------------------")