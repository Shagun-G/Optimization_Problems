from Quadratic import Quadratic
import numpy as np

problem = Quadratic(n = 2, c = np.array([1]), b = np.array([[1, -1]]), A = np.ones((2,2)))

print(problem.initial_point())
print(problem.hessian(problem.initial_point()))
print(problem.gradient(problem.initial_point()))
print(problem.objective(problem.initial_point()))
print(np.shape(problem.gradient(problem.initial_point())))
