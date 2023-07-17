from Problem import Problem
import numpy as np

class Quadratic(Problem):

    def __init__(self, n: int, A : np.array, b : np.array, c : np.array) -> None:
        
        self.A = A
        self.b = b.reshape(n, 1)
        self.c = c.reshape(1, 1)
        super().__init__("Quadratic", n)

    def initial_point(self) -> np.array:
        return np.zeros((self.n, 1))
    
    def objective(self, x: np.array) -> float:
        val = self.c + np.dot(x.T, self.b) + np.dot(np.dot(x.T, self.A), x)/2
        return val[0, 0]

    def gradient(self, x: np.array) -> np.array:
        return self.b + np.dot(self.A, x)
    
    def hessian(self, x: np.array) -> np.array:
        return self.A