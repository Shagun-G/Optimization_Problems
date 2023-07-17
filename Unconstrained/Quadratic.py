from Unconstrained.Problem import Problem
import numpy as np

class Quadratic(Problem):

    def __init__(self, d: int, A : np.array, b : np.array, c : np.array) -> None:
        
        self.A = A
        self.b = b.reshape(d, 1)
        self.c = c.reshape(1, 1)
        super().__init__("Quadratic", d)

    def initial_point(self) -> np.array:
        return np.zeros((self.d, 1))
    
    def objective(self, x: np.array) -> float:
        val = self.c + np.dot(x.T, self.b) + np.dot(np.dot(x.T, self.A), x)/2
        return val[0, 0]

    def gradient(self, x: np.array) -> np.array:
        return self.b + np.dot(self.A, x)
    
    def hessian(self, x: np.array) -> np.array:
        return self.A

    @classmethod
    def generate(cls, d : int, seed : int = 100, xi : int = 2):

        '''Generates a quadratic based on the process in Numerical Experiments in: 
            A. Mokhtari, Q. Ling and A. Ribeiro, "Network Newton Distributed Optimization Methods," in IEEE Transactions on Signal Processing, vol. 65, no. 1, pp. 146-161, 1 Jan.1, 2017, doi: 10.1109/TSP.2016.2617829.
        '''
        np.random.seed(seed)

        s1 = 10**np.arange(xi)
        s2 = 1/10**np.arange(xi)
        if d%2 == 0:
            v = np.hstack((np.random.choice(s1, size = int(d/2)), np.random.choice(s2, size = int(d/2))))
        else:
            v = np.hstack((np.random.choice(s1, size = int(d/2) + 1), np.random.choice(s2, size = int(d/2))))

        A = np.diag(v)
        b = np.random.uniform(0, 1, d)*10**(int(xi/2))
        print("Condition number : ", np.linalg.cond(A))
        return cls(d = d, A = np.diag(v), b = b, c = np.array([0]))
