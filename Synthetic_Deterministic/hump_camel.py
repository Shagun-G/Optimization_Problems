from Base_classes import Unconstrained

import numpy as np


# TODO : gradient
# TODO : hessian
class hump_camel(Unconstrained):
    """Defines a 6 Hump Camel function problem."""

    def __init__(self) -> None:
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
