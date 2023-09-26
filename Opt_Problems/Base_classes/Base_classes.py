"""Abstract Class that define problems.

All problems defined in this class have:

1. Single objective function
2. Continuous Variables
3. Use numpy arrays for input and output
"""

from abc import ABC, abstractmethod
import numpy as np


class Unconstrained(ABC):

    """
    Defines structure of enconstrained problems.

    Required Attributes:

    name    :   Name of the problem
    d       :   Dimension of the problem
    """

    def __init__(self, name: str, d: int) -> None:
        self._name = name
        self._d = d

    @property
    def name(self) -> str:
        return self._name

    @property
    def d(self) -> int:
        return self._d

    @abstractmethod
    def initial_point(self) -> np.array:
        """initial point of the problem, shape (d, 1)"""
        pass

    @abstractmethod
    def objective(self, x: np.array) -> float:
        """return objective value"""
        pass

    @abstractmethod
    def gradient(self, x: np.array) -> np.array:
        """return gradient of function, shape (d, 1)"""
        pass

    @abstractmethod
    def hessian(self, x: np.array) -> np.array:
        """returns hessian of problem, shape (d, d)"""
        pass

class Constrained(ABC):

    """
    Defines structure of enconstrained problems.

    Required Attributes:

    name    :   Name of the problem
    d       :   Dimension of the problem
    """

    def __init__(self, name: str, d: int) -> None:
        self._name = name
        self._d = d

    @property
    def name(self) -> str:
        return self._name

    @property
    def d(self) -> int:
        return self._d

    @abstractmethod
    def initial_point(self) -> np.array:
        """initial point of the problem, shape (d, 1)"""
        pass

    @abstractmethod
    def objective(self, x: np.array) -> float:
        """return objective value"""
        pass

    @abstractmethod
    def gradient(self, x: np.array) -> np.array:
        """return gradient of function, shape (d, 1)"""
        pass

    @abstractmethod
    def hessian(self, x: np.array) -> np.array:
        """returns hessian of problem, shape (d, d)"""
        pass

    @abstractmethod
    def constraints_equality(self, x: np.array) -> float:
        """return equality constrainst value"""
        pass

    @abstractmethod
    def constraints_equality_jacobian(self, x: np.array) -> float:
        """return equality constrainsts jacobian matrix value"""
        pass

    @abstractmethod
    def constraints_inequality(self, x: np.array) -> float:
        """return inequality constrainst value"""
        pass

    @abstractmethod
    def constraints_inequality_jacobian(self, x: np.array) -> float:
        """return inequality constrainst jacobian matrix value"""
        pass