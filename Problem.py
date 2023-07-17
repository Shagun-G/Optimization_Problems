'''The main abstract class that defines a problem.

All problems defined with this class have:

1. Single objective function
2. Continuous Variables
3. Use numpy arrays for input and output
'''

from abc import ABC, abstractmethod
import numpy as np

class Problem(ABC):

    '''
    Attributes

    name    :   Name of the problem
    n       :   Dimension of the problem
    '''

    def __init__(self, name : str, n : int) -> None:
        self.name = name
        self.n = n
    
    def name(self) -> str:
        return self.name
    
    def n(self) -> int:
        return self.n
    
    @abstractmethod
    def initial_point(self) -> np.array:
        '''initial point of the problem, shape (n, 1)'''        
        pass

    @abstractmethod
    def objective(self, x : np.array) -> float:
        '''return objective value'''
        pass

    @abstractmethod
    def gradient(self, x : np.array) -> np.array:
        '''return gradient of function, shape (n, 1)'''
        pass

    @abstractmethod
    def hessian(self, x : np.array) -> np.array:
        '''returns hessian of problem, shape (n, 1)'''
        pass

