import numpy as np
from Base_classes import Unconstrained


# TODO : function documentation
class Quadratic(Unconstrained):

    """Creates a dataset synthetic dataset of strongly convex quadratics for a stochastic problem"""

    def __init__(self, d: int, n_quadratics: int, seed: int | None = None, xi: int = 2) -> None:
        """Generates a quadratics based on the process in Numerical Experiments in:
        A. Mokhtari, Q. Ling and A. Ribeiro, "Network Newton Distributed Optimization Methods," in IEEE Transactions on Signal Processing, vol. 65, no. 1, pp. 146-161, 1 Jan.1, 2017, doi: 10.1109/TSP.2016.2617829.

        Inputs:
        d           :   dimension of problem
        n_quadratic :   number of qaudratics or datapoints for the stochastic problem
        seed    :   seed for sampling (optional)
        xi      :   constrols condition number, increase to increase conditon number
                    (optional, default 2)
        """

        self._A_list = np.zeros((d, d, n_quadratics))
        self._b_list = np.zeros((d, n_quadratics))
        self._number_of_datapoints = n_quadratics

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
            for i in range(n_quadratics):
                v = np.hstack((rng.choice(s1, size=int(d / 2)), rng.choice(s2, size=int(d / 2))))
                self._A_list[:, :, i] = np.diag(v)
                self._b_list[:, i] = rng.random((d)) * 10 ** (int(xi / 2))
        else:
            for i in range(n_quadratics):
                v = np.hstack((rng.choice(s1, size=int(d / 2) + 1), rng.choice(s2, size=int(d / 2))))
                self._A_list[:, :, i] = np.diag(v)
                self._b_list[:, i] = rng.random((d)) * 10 ** (int(xi / 2))

        self._A_sum = np.sum(self._A_list, axis=2)
        self._b_sum = np.sum(self._b_list, axis=1).reshape((d, 1))
        print("Condition number : ", np.linalg.cond(self._A_sum))

        super().__init__("Stochastic Quadratic", d)

    def initial_point(self) -> np.array:
        return np.ones((self.d, 1))

    @property
    def number_of_datapoints(self) -> int:
        return self._number_of_datapoints

    def _determine_batch(self, type: str = "full", batch_size: int = 0, seed: int | None = None) -> np.array:
        """
        Generates an array of indices for a batch of data for calculation
        Inputs:
        type        :   "full" for full data and "stochastic" for batch
        batch_size  :   (optional) only used when stochastic type of batch
        seed        :   (optional) if not specified, random batch, else specify for the same batch of data
        """
        if type == "full":
            s = np.arange(self._number_of_datapoints)
            batch_size = self._number_of_datapoints
            return s, batch_size

        if type == "stochastic":
            if seed is None:
                rng = np.random.default_rng(np.random.randint(np.iinfo(np.int16).max, size=1)[0])
            elif isinstance(seed, int):
                rng = np.random.default_rng(seed)
            else:
                raise Exception("seed must be an integer if specified")

            if batch_size < 0:
                raise Exception(f"{type} gradient requires a batch_size > 0")

            if batch_size > self._number_of_datapoints:
                raise Exception("Batch size specified is larger than size of dataset")

            s = rng.choice(self._number_of_datapoints, size=(batch_size), replace=False)
            return s, batch_size

        raise Exception(f"{type} is not a defined type of gradient")

    def objective(self, x: np.array, type: str = "full", batch_size: int = 0, seed: int | None = None) -> float:
        s = self._determine_batch(type, batch_size, seed)
        x = x.reshape((self.d, 1))
        val = 0.5 * np.dot(np.dot(x.T, np.sum(self._A_list[:, :, s], axis=2)), x) + np.dot(
            np.sum(self._b_list[:, s], axis=1), x
        )

        return val[0, 0]

    def gradient(self, x: np.array, type: str = "full", batch_size: int = 0, seed: int | None = None) -> np.array:
        s, batch_size = self._determine_batch(type, batch_size, seed)
        x = x.reshape((self.d, 1))
        val = np.dot(np.sum(self._A_list[:, :, s], axis=2), x) + np.sum(self._b_list[:, s], axis=1).reshape((self.d, 1))

        return val

    def hessian(self, x: np.array, type: str = "full", batch_size: int = 0, seed: int | None = None) -> np.array:
        s, batch_size = self._determine_batch(type, batch_size, seed)
        return np.sum(self._A_list[:, :, s], axis=2)
