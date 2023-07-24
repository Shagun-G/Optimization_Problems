from Base_classes import Unconstrained
import numpy as np


class Quadratic(Unconstrained):

    """
    Generates a quadratic function:
    A   :   matrix for quadratic
    b   :   linear term for quadratic
    c   :   constant term for wuadratic
    d   :   dimension of problem
    """

    def __init__(self, d: int, A: np.array, b: np.array, c: np.array) -> None:
        self.A = A
        self.b = b.reshape(d, 1)
        self.c = c.reshape(1, 1)
        super().__init__("Quadratic", d)

    def initial_point(self) -> np.array:
        return np.zeros((self._d, 1))

    def objective(self, x: np.array) -> float:
        val = self.c + np.dot(x.T, self.b) + 0.5 * np.dot(np.dot(x.T, self.A), x)
        return val[0, 0]

    def gradient(self, x: np.array) -> np.array:
        return self.b + np.dot(self.A, x)

    def hessian(self, x: np.array) -> np.array:
        return self.A

    @classmethod
    def generate(cls, d: int, seed: int | None = None, xi: int = 2):
        """Generates a quadratic based on the process in Numerical Experiments in:
        A. Mokhtari, Q. Ling and A. Ribeiro, "Network Newton Distributed Optimization Methods," in IEEE Transactions on Signal Processing, vol. 65, no. 1, pp. 146-161, 1 Jan.1, 2017, doi: 10.1109/TSP.2016.2617829.

        Attributes:

        seed    :   seed for sampling (optional, default 100)
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
