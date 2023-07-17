from Base_classes import Unconstrained

import numpy as np

class Rosenbrock(Unconstrained):
    '''Defines a rosenbrock function problem.'''

    def __init__(self, d: int) -> None:
        if d < 2:
            raise   Exception("Rosenbrock must be atleast dimension 2")

        super().__init__("Rosenbrock", d)    

    def initial_point(self) -> np.array:
        return np.zeros((self.d, 1))
    
    def objective(self, x: np.array) -> float:

        val = 0
        for i in range(self.d - 1):
            val = val + 100 * (x[i + 1, 0] - x[i, 0] ** 2) ** 2 + (1 - x[i, 0]) ** 2
        return val

    def gradient(self, x: np.array) -> np.array:

        grad = np.zeros((self.d, 1));
        grad[0, 0] = 200*x[0, 0]*(x[0, 0]**2 - x[1, 0]) + x[0, 0] - 1
        grad[-1, 0] = 100*(x[-1, 0] - x[-2, 0]**2)
        for i in range(1, self.d - 1):
            grad[i, 0] = 200*x[i, 0]*(x[i, 0]**2 - x[i+1, 0]) + x[i, 0] - 1 + 100*(x[i, 0] - x[i-1, 0]**2)
        grad = 2*grad
        return grad

    def hessian(self, x: np.array) -> np.array:

        H = np.zeros((self.d, self.d))
        H[0, 0] = -400*x[1, 0] + 1200*x[0, 0]**2 + 2
        H[0, 1] = -400*x[0, 0]
        H[1, 0] = H[0, 1]
        H[-1, -1] = 200
        
        for i in range(1, self.d - 1):
            H[i, i] = 202 + 1200*x[i, 0]**2 - 400*x[i+1, 0]
            H[i, i+1] = -400*x[i, 0]
            H[i+1, i] = H[i, i+1]
        return H