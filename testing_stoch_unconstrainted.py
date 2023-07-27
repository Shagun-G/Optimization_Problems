import numpy as np

# * Quadratic Problem
from Stochastic_Unconstrained.Synthetic import Quadratic

problem = Quadratic(d=2, n_quadratics=10, seed=1000, xi=6)

"""Calling all functions"""
print("Name: " + problem.name)
print("Dimension : ", problem.d)
print("At x = ", problem.initial_point())
print("Function: ", problem.objective(x=problem.initial_point()))
print("Gradient : ", problem.gradient(x=problem.initial_point()))
print("Hessian: ", problem.hessian(x=problem.initial_point()))
print("------------------------------------------")
