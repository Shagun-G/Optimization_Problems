import numpy as np

# * Quadratic Problem
from Synthetic.Quadratic import Quadratic
# Specifying
problem = Quadratic(d = 2, c = np.array([1]), b = np.array([[1, -1]]), A = np.ones((2,2)))
# Generating
problem = Quadratic.generate(d = 4, xi = 3, seed = 100)

# * Rosenbrock Problem
from Synthetic.Rosenbrock import Rosenbrock
problem = Rosenbrock(d = 3)

# * Beale Problem
from Synthetic.Beale import Beale
problem = Beale()


'''Calling all functions'''
print("Name: " + problem.name)
print("Dimension : ", problem.d)
print("At x = ", problem.initial_point())
print("Gradient : ", problem.gradient(x = problem.initial_point()))
print("Hessian: ", problem.hessian(x = problem.initial_point()))
print("------------------------------------------")

'''Checking Gradient and Hessian Code with autodifferentiation'''
x = np.random.rand(problem.d, 1)

import jax      #pip install jax, "jax[cpu]"
print(problem.gradient(x))
print(jax.grad(problem.objective)(x))
print("--------------------")
print(problem.hessian(x))
print(jax.jacfwd(jax.grad(problem.objective))(x))