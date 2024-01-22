from Opt_Problems.Machine_Learning.Constrained_Logistic_Regression import Cross_Entropy_Binary_Linear_norm_constraint, Cross_Entropy_Binary_Linear_constraint, Cross_Entropy_Binary_norm_constraint
import numpy as np

problem = Cross_Entropy_Binary_Linear_norm_constraint(name="australian", location="../Datasets/australian_scale.txt", m=10, constraint_seed=1)

# print(problem.number_of_features)
# print(problem.number_of_datapoints)
# print(problem.name)
# print(problem.objective(x=problem.initial_point(), type="stochastic", batch_size=5, seed=80))
# print(problem.gradient(x=problem.initial_point(), type="stochastic", batch_size=5, seed=80))
# print(problem.constraints_eq(x=problem.initial_point()))
# print(problem.constraints_eq_jacobian(x=problem.initial_point()))
# print(problem.constraints_eq_hessian(x=problem.initial_point()))

# problem = Cross_Entropy_Binary_norm_constraint(name="australian", location="../Datasets/australian_scale.txt")

# print(problem.number_of_features)
# print(problem.number_of_datapoints)
# print(problem.name)
# print(problem.objective(x=problem.initial_point(), type="stochastic", batch_size=5, seed=80))
# print(problem.gradient(x=problem.initial_point(), type="stochastic", batch_size=5, seed=80))
# print(problem.constraints_eq(x=problem.initial_point()))
# print(problem.constraints_eq_jacobian(x=problem.initial_point()))
# print(problem.constraints_eq_hessian(x=problem.initial_point()))

# problem = Cross_Entropy_Binary_Linear_constraint(name="australian", location="../Datasets/australian_scale.txt", m=10, constraint_seed=1)

# print(problem.number_of_features)
# print(problem.number_of_datapoints)
# print(problem.name)
# print(problem.objective(x=problem.initial_point(), type="stochastic", batch_size=5, seed=80))
# print(problem.gradient(x=problem.initial_point(), type="stochastic", batch_size=5, seed=80))
# print(problem.constraints_eq(x=problem.initial_point()))
# print(problem.constraints_eq_jacobian(x=problem.initial_point()))
# print(problem.constraints_eq_hessian(x=problem.initial_point()))

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
# print(hessian(x))
# print(jax.jacfwd(jax.grad(obj_fun))(x))

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