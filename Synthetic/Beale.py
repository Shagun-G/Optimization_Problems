from Base_classes import Unconstrained

import numpy as np

class Beale(Unconstrained):
    '''Defines a Beale function problem.'''

    def __init__(self) -> None:
        self._data = np.array([1.5, 2.25, 2.625])
        super().__init__("Beale", 2)    

    def initial_point(self) -> np.array:
        return np.zeros((self._d, 1))
    
    def objective(self, x: np.array) -> float:
        val = 0
        for i in range(3):
            val = val + (self._data[i] - x[0, 0] * (1 - x[1, 0] ** (i + 1))) ** 2
        return val

    def gradient(self, x: np.array) -> np.array:

        grad = np.zeros((self._d, 1));
        for i in range(3):
            t = 2*(self._data[i] - x[0,0]*(1 - x[1, 0]**(i+1)))
            grad[0,0] = grad[0,0] - t*(1 - x[1, 0]**(i+1))
            grad[1,0] = grad[1,0] + t*x[0, 0]*(i+1)*(x[1, 0]**(i))
        return grad

    def hessian(self, x: np.array) -> np.array:

        H = np.zeros((self._d, self._d))
        # for i in range(3)
        #     H[0,0] += 2*(1 - x[1, 0]**(i+1))*(1 - x[1, 0]**(i+1))  
        #     H[1,0] += 2*(self._data[i] - x[0,0]*(1 - x[1, 0]**(i+1)))*(-1 + x[1, 0]**(i+1))
        #     if i ==0:
        #         H[1,1] += 2*(x[0,0]*(x[1, 0]))*x[0, 0]
        #     else:
        #         H[1,1] += 2*(self._data[i] - x[0,0]*(1 - x[1, 0]**(i+1)))*x[0, 0]*(i+1)*i*(x[1, 0]**(i-1)) + 2*(i+1)*(x[0,0]*x[1, 0]**(i))*x[0, 0]*(i+1)*(x[1, 0]**(i))       
        
        # H[1,1] *= 2*x[0,0]
        H[0,1] = H[1,0]
        return H