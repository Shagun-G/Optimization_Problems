import numpy as np

# * Quadratic Problem
from Synthetic.Quadratic import Quadratic
# Specifying
problem = Quadratic(d = 2, c = np.array([1]), b = np.array([[1, -1]]), A = np.ones((2,2)))
# Generating
problem = Quadratic.generate(d = 4, xi = 3, seed = 100)

# * Rosenbrock Problem
from Synthetic.Rosenbrock import Rosenbrock
problem = Rosenbrock(d = 2)

# * Beale Problem
from Synthetic.Beale import Beale
problem = Beale()
print(problem.objective(x = np.array([[3], [0.5]])))