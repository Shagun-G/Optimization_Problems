import numpy as np
from Opt_Problems.Stochastic_Constrained.Synthetic import Quadratic_linear_norm_constraints, Quadratic_linear_constraints, Quadratic_norm_constraints

problem = Quadratic_linear_norm_constraints(d=2, n_quadratics=200, seed=1000, xi=6, m = 2)

"""Calling all functions"""
# print("Name: " + problem.name)
# print("Dimension : ", problem.d)
# print("At x = ", problem.initial_point())
# print("Function: ", problem.objective(x=problem.initial_point()))
# print("Gradient : ", problem.gradient(x=problem.initial_point()))
# print("Hessian: ", problem.hessian(x=problem.initial_point()))
# print(problem.constraints_eq(x = problem.initial_point()))
# print(problem.constraints_eq_jacobian(x = problem.initial_point()))
# print("------------------------------------------")


# problem = Quadratic_norm_constraints(d=2, n_quadratics=200, seed=1000, xi=6)

# """Calling all functions"""
# print("Name: " + problem.name)
# print("Dimension : ", problem.d)
# print("At x = ", problem.initial_point())
# # print("Function: ", problem.objective(x=problem.initial_point()))
# # print("Gradient : ", problem.gradient(x=problem.initial_point()))
# # print("Hessian: ", problem.hessian(x=problem.initial_point()))
# print(problem.constraints_eq(x = problem.initial_point()))
# print(problem.constraints_eq_jacobian(x = problem.initial_point()))
# print("------------------------------------------")


# problem = Quadratic_linear_constraints(d=4, n_quadratics=200, seed=1000, xi=6, m = 2)

# """Calling all functions"""
# print("Name: " + problem.name)
# print("Dimension : ", problem.d)
# print("At x = ", problem.initial_point())
# # print("Function: ", problem.objective(x=problem.initial_point()))
# # print("Gradient : ", problem.gradient(x=problem.initial_point()))
# # print("Hessian: ", problem.hessian(x=problem.initial_point()))
# print(problem.constraints_eq(x = problem.initial_point()))
# print(problem.constraints_eq_jacobian(x = problem.initial_point()))
# print("------------------------------------------")


"""Checking Gradient and Hessian Code with autodifferentiation"""
x = np.random.rand(problem.d, 1)

import jax  # pip install jax, "jax[cpu]"
# from autograd import grad  # pip install autograd

obj_fun = lambda x: problem.objective(x, type="full", batch_size=10, seed=100)
gradient = lambda x: problem.gradient(x, type="full", batch_size=10, seed=100)
hessian = lambda x: problem.hessian(x, type="full")
# print(jax(x))
# print(grad(obj_fun)(x))
# print("--------------------")
# print(hessian(x))
# print(grad(grad(obj_fun))(x))

print(gradient(x))
print(jax.grad(obj_fun)(x))
print("--------------------")
print(hessian(x))
print(jax.jacfwd(jax.grad(obj_fun))(x))

from autograd import grad  # pip install autograd
con_eq = lambda x: problem.constraints_eq(x)
con_eq_jac = lambda x: problem.constraints_eq_jacobian(x)
con_eq_hess = lambda x: problem.constraints_eq_hessian(x)

print("-----------------------")
print(con_eq_jac(x))
print(jax.jacfwd(con_eq)(x).squeeze())
# print("--------------------")
# print(hessian(x))
# print(jax.jacfwd(jax.grad(obj_fun))(x))