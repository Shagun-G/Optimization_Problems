from Base_classes import Unconstrained

import numpy as np
# TODO : gradient 
# TODO : hessian
class Branin(Unconstrained):
    '''Defines a branin function problem.'''

    def __init__(self) -> None:
        super().__init__("Branin", 2)    

    def initial_point(self) -> np.array:
        return np.zeros((self._d, 1))
    
    def objective(self, x: np.array) -> float:

        a = 1.0
        b = 5.1 / (4.0 * pow(np.pi, 2.0))
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)

        val = a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s

        # offsetting to make optimal 0
        return val - 0.3978873577299371  

    def gradient(self, x: np.array) -> np.array:
        
        raise Exception("{} gradient not available".format(self.name))

    def hessian(self, x: np.array) -> np.array:

        raise Exception("{} hessian not available".format(self.name))
