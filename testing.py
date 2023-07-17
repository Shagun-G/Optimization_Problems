import numpy as np

# * Quadratic Problem
from Unconstrained.Quadratic import Quadratic
# Specifying
problem = Quadratic(d = 2, c = np.array([1]), b = np.array([[1, -1]]), A = np.ones((2,2)))
# Generating
problem = Quadratic.generate(d = 4, xi = 3, seed = 100)

