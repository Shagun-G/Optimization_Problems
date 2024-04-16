import numpy as np
from numpy.core.multiarray import array as array
from sklearn.datasets import load_svmlight_file
from abc import ABC, abstractmethod
import numpy as np
from sklearn import preprocessing


class Problem(ABC):
    """
    Defines structure of problems.

    All problems defined in this class have:

    1. Single objective function
    2. Continuous Variables
    3. Use numpy arrays for input and output

    Required Attributes:

    name    :   Name of the problem
    d       :   Dimension of the problem
    """

    def __init__(
        self,
        name: str,
        d: int,
        eq_const_number: int = 0,
        ineq_const_number: int = 0,
        number_of_datapoints: int = 1,
    ) -> None:
        self._name = name
        self._d = d
        self._number_of_eq_constraints = eq_const_number
        self._number_of_ineq_constraints = ineq_const_number
        self._number_of_datapoints = number_of_datapoints

    @property
    def name(self) -> str:
        return self._name

    @property
    def d(self) -> int:
        return self._d

    @property
    def number_of_eq_constraints(self) -> int:
        return self._number_of_eq_constraints

    @property
    def number_of_ineq_constraints(self) -> int:
        return self._number_of_ineq_constraints

    @property
    def number_of_datapoints(self) -> int:
        return self._number_of_datapoints

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
    def constraints_eq(self, x: np.array) -> np.array:
        """return equality constrainst value (m_eq, 1)"""
        pass

    @abstractmethod
    def constraints_eq_jacobian(self, x: np.array) -> np.array:
        """return equality constrainsts jacobian matrix value (m_eq, n)"""
        pass

    @abstractmethod
    def constraints_eq_hessian(self, x: np.array) -> np.array:
        """return equality constrainsts hessian tensor value (m_eq, n, n)"""
        pass

    @abstractmethod
    def constraints_ineq(self, x: np.array) -> np.array:
        """return inequality constrainst value (m_ineq, 1)"""
        pass

    @abstractmethod
    def constraints_ineq_jacobian(self, x: np.array) -> np.array:
        """return inequality constrainst jacobian matrix value (m_ineq, n)"""
        pass

    @abstractmethod
    def constraints_ineq_hessian(self, x: np.array) -> np.array:
        """return inequality constrainst jacobian matrix value (m_ineq, n, n)"""
        pass


class Unconstrained_Problem(Problem, ABC):
    """
    Defines structure of unconstrained problems.
    Required Attributes:

    name    :   Name of the problem
    d       :   Dimension of the problem
    """

    def __init__(
        self,
        name: str,
        d: int,
        number_of_datapoints: int = 1,
    ) -> None:
        self._name = name
        self._d = d
        self._number_of_datapoints = number_of_datapoints
        self._number_of_eq_constraints = 0
        self._number_of_ineq_constraints = 0

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

    """The constraints below don't exist and thus return errors"""

    def constraints_eq(self, x: np.array):
        raise Exception("No Equality Constraints")

    def constraints_eq_jacobian(self, x: np.array):
        raise Exception("No Equality Constraints")

    def constraints_eq_hessian(self, x: np.array):
        raise Exception("No Equality Constraints")

    def constraints_ineq(self, x: np.array):
        raise Exception("No Inequality Constraints")

    def constraints_ineq_jacobian(self, x: np.array):
        raise Exception("No Inequality Constraints")

    def constraints_ineq_hessian(self, x: np.array):
        raise Exception("No Inequality Constraints")


def datasets_manager(name, location):

    if name not in location:
        raise Exception("Name and file pointed to in location are different")

    X, y = load_svmlight_file(location)

    # preprocessing for specific datasets
    if "mushroom" in name.lower():
        # the target variable needs to be offset
        y = y - 1
    elif "australian" in name.lower():
        # the target has to be changed from {-1, +1} to {0, 1}
        y[y == -1] = 0
    elif "phishing" in name.lower():
        # no formatting required, {0,1} labels
        pass
    elif "sonar" in name.lower():
        # the target has to be changed from {-1, +1} to {0, 1}
        y[y == -1] = 0
    elif "gisette" in name.lower():
        # the target has to be changed from {-1, +1} to {0, 1}
        y[y == -1] = 0
    elif "a9a" in name.lower():
        # the target has to be changed from {-1, +1} to {0, 1}
        y[y == -1] = 0
    elif "w8a" in name.lower():
        # the target has to be changed from {-1, +1} to {0, 1}
        y[y == -1] = 0
    elif "ijcnn" in name.lower():
        # the target has to be changed from {-1, +1} to {0, 1}
        y[y == -1] = 0
    elif "real-sim" in name.lower():
        # the target has to be changed from {-1, +1} to {0, 1}
        y[y == -1] = 0
    else:
        raise Exception(
            "Unknown dataset, preprocessing might be required for correct format"
        )

    # normalizing dataset
    # Do not use scipy normaize due to the norm options being limited and not incorporating the negative values well.
    # X = preprocessing.normalize(X)
    # X = preprocessing.normalize(X, axis = 0, )
    # X = (X - X.min(0)) / X.ptp(0)
    '''Normalization being done now at the problem formulation'''

    return X, y