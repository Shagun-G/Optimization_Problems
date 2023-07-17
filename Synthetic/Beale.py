from Base_classes import Unconstrained

import numpy as np

class Beale(Unconstrained):
    '''Defines a Beale function problem.'''

    def __init__(self) -> None:
        self.data = np.array([1.5, 2.25, 2.625])
        super().__init__("Beale", 2)    

    def initial_point(self) -> np.array:
        return np.zeros((self.d, 1))
    
    def objective(self, x: np.array) -> float:
        val = 0
        for i in range(3):
            val = val + (self.data[i] - x[0, 0] * (1 - x[1, 0] ** (i + 1))) ** 2
        return val

    def gradient(self, x: np.array) -> np.array:

        grad = np.zeros((self.d, 1));
        for i in range(3):
            t = 2*(self.data[i] - x[0,0]*(1 - x[1, 0]**i))
            grad[0,0] = grad[0,0] - t*(1 - x(2)**i)
            grad[1,0] = grad[1,0] + t*x[0, 0]*i*(x[1, 0]**(i-1))
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