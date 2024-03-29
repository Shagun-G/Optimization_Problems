from Opt_Problems.utils import Unconstrained_Problem
import numpy as np
from typing import Callable


class Generator(Unconstrained_Problem):
    """
    Generates a deterministic unconstrained problem using the specified, function, gradient and hessian oracles in the framework of this repository.
    """

    def __init__(
        self,
        name: str,
        d: int,
        x_init: None | np.ndarray = None,
        objective: Callable | None = None,
        gradient: Callable | None = None,
        hessian: Callable | None = None,
    ) -> None:
        self.x_init = np.reshape(x_init, (d, 1))
        self.func = objective
        self.grad = gradient
        self.hess = hessian
        if objective == None and gradient == None and hessian == None:
            raise "Need atleast one of function, gradient or hessian for the optimization problem"

        super().__init__(name, d)

    def initial_point(self) -> np.array:
        if self.x_init.any() == None:
            return np.zeros((self._d, 1))
        return self.x_init

    def objective(self, x: np.array) -> float:
        if self.func == None:
            raise Exception("function oracle not available for " + self.name)
        return self.func(x)

    def gradient(self, x: np.array) -> np.array:
        if self.grad == None:
            raise Exception("gradient oracle not available for " + self.name)
        return self.grad(x)

    def hessian(self, x: np.array) -> np.array:
        if self.hess == None:
            raise Exception("hessian oracle not available for " + self.name)
        return self.hess(x)


class Quadratic(Unconstrained_Problem):

    """
    Generates a quadratic function:
    """

    def __init__(self, d: int, A: np.array, b: np.array, c: np.array):
        """
        Inputs :
        A   :   matrix for quadratic
        b   :   linear term for quadratic
        c   :   constant term for wuadratic
        d   :   dimension of problem
        """
        self._A = A
        self._b = b.reshape(d, 1)
        self._c = c.reshape(1, 1)
        super().__init__("Quadratic", d)

    def initial_point(self) -> np.array:
        return np.zeros((self._d, 1))

    def objective(self, x: np.array) -> float:
        val = self._c + np.dot(x.T, self._b) + 0.5 * np.dot(np.dot(x.T, self._A), x)
        return val[0, 0]

    def gradient(self, x: np.array) -> np.array:
        return self._b + np.dot(self._A, x)

    def hessian(self, x: np.array) -> np.array:
        return self._A

    @classmethod
    def generate(cls, d: int, seed: int | None = None, xi: int = 2):
        """Generates a quadratic based on the process in Numerical Experiments in:
        A. Mokhtari, Q. Ling and A. Ribeiro, "Network Newton Distributed Optimization Methods," in IEEE Transactions on Signal Processing, vol. 65, no. 1, pp. 146-161, 1 Jan.1, 2017, doi: 10.1109/TSP.2016.2617829.

        Inputs:

        seed    :   seed for sampling (optional)
        xi      :   constrols condition number, increase to increase conditon number
                    (optional, default 2)
        """

        # random generator to avoid setting global generator
        if seed is None:
            rng = np.random.default_rng(np.random.randint(np.iinfo(np.int16).max, size=1)[0])
        elif isinstance(seed, int):
            rng = np.random.default_rng(seed)
        else:
            raise Exception("seed must be an enteger if specified")

        s1 = 10 ** np.arange(xi)
        s2 = 1 / 10 ** np.arange(xi)
        if d % 2 == 0:
            v = np.hstack((rng.choice(s1, size=int(d / 2)), rng.choice(s2, size=int(d / 2))))
        else:
            v = np.hstack((rng.choice(s1, size=int(d / 2) + 1), rng.choice(s2, size=int(d / 2))))
        b = rng.random((d)) * 10 ** (int(xi / 2))
        # print("Condition number : ", np.linalg.cond(A))
        return cls(d=d, A=np.diag(v), b=b, c=np.array([0]))


class Rosenbrock(Unconstrained_Problem):
    """Defines a rosenbrock function problem."""

    def __init__(self, d: int):
        if d < 2:
            raise Exception("Rosenbrock must be atleast dimension 2")

        super().__init__("Rosenbrock", d)

    def initial_point(self) -> np.array:
        return np.zeros((self._d, 1))

    def objective(self, x: np.array) -> float:
        val = 0
        for i in range(self._d - 1):
            val = val + 100 * (x[i + 1, 0] - x[i, 0] ** 2) ** 2 + (1 - x[i, 0]) ** 2
        return val

    def gradient(self, x: np.array) -> np.array:
        grad = np.zeros((self._d, 1))
        grad[0, 0] = 200 * x[0, 0] * (x[0, 0] ** 2 - x[1, 0]) + x[0, 0] - 1
        grad[-1, 0] = 100 * (x[-1, 0] - x[-2, 0] ** 2)
        for i in range(1, self._d - 1):
            grad[i, 0] = 200 * x[i, 0] * (x[i, 0] ** 2 - x[i + 1, 0]) + x[i, 0] - 1 + 100 * (x[i, 0] - x[i - 1, 0] ** 2)
        grad = 2 * grad
        return grad

    def hessian(self, x: np.array) -> np.array:
        H = np.zeros((self._d, self._d))
        H[0, 0] = -400 * x[1, 0] + 1200 * x[0, 0] ** 2 + 2
        H[0, 1] = -400 * x[0, 0]
        H[1, 0] = H[0, 1]
        H[-1, -1] = 200

        for i in range(1, self._d - 1):
            H[i, i] = 202 + 1200 * x[i, 0] ** 2 - 400 * x[i + 1, 0]
            H[i, i + 1] = -400 * x[i, 0]
            H[i + 1, i] = H[i, i + 1]
        return H


class Beale(Unconstrained_Problem):
    """Defines a Beale function problem."""

    def __init__(self):
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
        grad = np.zeros((self._d, 1))
        for i in range(3):
            t = 2 * (self._data[i] - x[0, 0] * (1 - x[1, 0] ** (i + 1)))
            grad[0, 0] = grad[0, 0] - t * (1 - x[1, 0] ** (i + 1))
            grad[1, 0] = grad[1, 0] + t * x[0, 0] * (i + 1) * (x[1, 0] ** (i))
        return grad

    def hessian(self, x: np.array) -> np.array:
        
        H = np.zeros((self._d, self._d))

        for i in range(3):
            H[0, 0] += 2*(1 - x[1, 0]**(i+1))**2
            H[1, 0] += 2*(i+1)*x[1, 0]**(i)*(self._data[i] - 2*x[0,0]*(1 - x[1, 0]**(i+1)))
        H[1, 1] = 2*x[0,0]*(4.5 + 15.75*x[1,0] + x[0,0]*(6*(x[1,0]**2 - x[1,0]) + 15*x[1,0]**4 - 1))
        H[0, 1] = H[1, 0]
        return H


# TODO : gradient
# TODO : hessian
class Branin(Unconstrained_Problem):
    """Defines a Branin function problem."""

    def __init__(self):
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


# TODO : gradient
# TODO : hessian
class hump_camel(Unconstrained_Problem):
    """Defines a 6 Hump Camel function problem."""

    def __init__(self):
        super().__init__("6 Hump Camel", 2)

    def initial_point(self) -> np.array:
        return np.zeros((self._d, 1))

    def objective(self, x: np.array) -> float:
        val = (
            (4 - 2.1 * (x[0] ** 2) + (x[0] ** 4) / 3) * (x[0] ** 2) + x[0] * x[1] + (-4 + 4 * (x[1] ** 2)) * (x[1] ** 2)
        )

        # offsetting to make optimal value 0
        return val + 1.0316284534898774

    def gradient(self, x: np.array) -> np.array:
        raise Exception("{} gradient not available".format(self.name))

    def hessian(self, x: np.array) -> np.array:
        raise Exception("{} hessian not available".format(self.name))
