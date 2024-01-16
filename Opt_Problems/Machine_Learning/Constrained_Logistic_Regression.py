import numpy as np
from numpy.core.multiarray import array as array
from Opt_Problems.utils import datasets_manager, Problem
from Opt_Problems.Machine_Learning.Logistic_Regression import Cross_Entropy_Binary

class Cross_Entropy_Binary_Linear_norm_constraint(Problem):
    """
    Class that performs logistic regression over datasets with two constraints : 
    Ax = b_1 and ||x|| = b_2
    A and b_1 are drawn from standard normal distributions, b_2 is set by defailt to 1 following numerical procedure from "An Adaptive Sampling Sequential Quadratic Programming Method for Equality Constrained Stochastic Optimization (https://arxiv.org/pdf/2206.00712.pdf)"
    """

    def __init__(self, location: str, name: str, m : int = 10, b_2 : float = 1, constraint_seed : int | None = None):

        """
        location : location for lobsvm file,
        name : name of dataset
        m : number of linear constraints
        b_2 : constant for the norm constraint
        constraint_seed : seed for generating the linear constraint
        """

        self._number_of_classes = 2
        self._dataset_name = name
        self.logistic_function = Cross_Entropy_Binary(location=location, name=name)

        super().__init__(name=f"{name}_cross_entropy_logistic_linear_norm_constraint", d=self.logistic_function.d, number_of_datapoints=self.logistic_function.number_of_datapoints, eq_const_number=m+1, ineq_const_number=0)

        """Generating Constraints"""
        if constraint_seed is None:
            rng = np.random.default_rng(np.random.randint(np.iinfo(np.int16).max, size=1)[0])
        else:
            rng = np.random.default_rng(constraint_seed)

        self._A = rng.standard_normal(size=(m, self.d))
        self._b_1 = rng.standard_normal(size=(m, 1))
        self._b_2 = b_2
        self._m = m

    def initial_point(self) -> np.ndarray:
        return np.ones((self.d, 1))

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def number_of_classes(self) -> int:
        return self._number_of_classes

    @property
    def number_of_features(self) -> int:
        return self.logistic_function._number_of_features

    def objective(self, x: np.array, type: str, batch_size: int = 0, seed: int | None = None, data_indices : list | None = None) -> float:
        return self.logistic_function.objective(x=x, type = type, batch_size=batch_size, seed=seed, data_indices=data_indices)

    def gradient(self, x: np.array, type: str, batch_size: int = 0, seed: int | None = None, data_indices : list | None = None) -> np.ndarray:
        return self.logistic_function.gradient(x=x, type=type, batch_size=batch_size, seed=seed, data_indices=data_indices)

    def hessian(self, x: np.array, type: str, batch_size: int = 0, seed: int | None = None, data_indices : list | None = None) -> np.array:
        raise Exception(f"Can't compute hessian for {self.name}")

    def constraints_eq(self, x: np.array) -> np.array:
        return np.vstack((np.dot(self._A, x) - self._b_1, np.linalg.norm(x)**2 - self._b_2))

    def constraints_eq_jacobian(self, x: np.array) -> np.array:
        return np.vstack((self._A, 2*x.T))

    def constraints_eq_hessian(self, x: np.array) -> np.array:
        hessians = []
        for _ in range(self._m):
            hessians.append(np.zeros((self.d, self.d)))
        hessians.append(np.identity(self.d))

        return np.array(hessians)

    def constraints_ineq(self, x: np.array) -> np.array:
        raise Exception("No inequality constraints")
    
    def constraints_ineq_jacobian(self, x: np.array) -> np.array:
        raise Exception("No inequality constraints")
    
    def constraints_ineq_hessian(self, x: np.array) -> np.array:
        raise Exception("No inequality constraints")


class Cross_Entropy_Binary_Linear_constraint(Problem):
    """
    Class that performs logistic regression over datasets with linear constraints : 
    Ax = b_1
    A and b_1 are drawn from standard normal distributions
    """

    def __init__(self, location: str, name: str, m : int = 10, constraint_seed : int | None = None):

        """
        location : location for lobsvm file,
        name : name of dataset
        m : number of linear constraints
        b_2 : constant for the norm constraint
        constraint_seed : seed for generating the linear constraint
        """

        self._number_of_classes = 2
        self._dataset_name = name
        self.logistic_function = Cross_Entropy_Binary(location=location, name=name)

        super().__init__(name=f"{name}_cross_entropy_logistic_linear_constraint", d=self.logistic_function.d, number_of_datapoints=self.logistic_function.number_of_datapoints, eq_const_number=m, ineq_const_number=0)

        """Generating Constraints"""
        if constraint_seed is None:
            rng = np.random.default_rng(np.random.randint(np.iinfo(np.int16).max, size=1)[0])
        else:
            rng = np.random.default_rng(constraint_seed)

        self._A = rng.standard_normal(size=(m, self.d))
        self._b_1 = rng.standard_normal(size=(m, 1))
        self._m = m

    def initial_point(self) -> np.ndarray:
        return np.ones((self.d, 1))

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def number_of_classes(self) -> int:
        return self._number_of_classes

    @property
    def number_of_features(self) -> int:
        return self.logistic_function._number_of_features

    def objective(self, x: np.array, type: str, batch_size: int = 0, seed: int | None = None, data_indices : list | None = None) -> float:
        return self.logistic_function.objective(x=x, type = type, batch_size=batch_size, seed=seed, data_indices=data_indices)

    def gradient(self, x: np.array, type: str, batch_size: int = 0, seed: int | None = None, data_indices : list | None = None) -> np.ndarray:
        return self.logistic_function.gradient(x=x, type=type, batch_size=batch_size, seed=seed, data_indices=data_indices)

    def hessian(self, x: np.array, type: str, batch_size: int = 0, seed: int | None = None, data_indices : list | None = None) -> np.array:
        raise Exception(f"Can't compute hessian for {self.name}")

    def constraints_eq(self, x: np.array) -> np.array:
        return np.dot(self._A, x) - self._b_1

    def constraints_eq_jacobian(self, x: np.array) -> np.array:
        return self._A

    def constraints_eq_hessian(self, x: np.array) -> np.array:
        hessians = []
        for _ in range(self._m):
            hessians.append(np.zeros((self.d, self.d)))

        return np.array(hessians)

    def constraints_ineq(self, x: np.array) -> np.array:
        raise Exception("No inequality constraints")
    
    def constraints_ineq_jacobian(self, x: np.array) -> np.array:
        raise Exception("No inequality constraints")
    
    def constraints_ineq_hessian(self, x: np.array) -> np.array:
        raise Exception("No inequality constraints")


class Cross_Entropy_Binary_norm_constraint(Problem):
    """
    Class that performs logistic regression over datasets with norm constraints : 
    ||x|| = b_2
    b_2 is set by defailt to 1
    """

    def __init__(self, location: str, name: str, b_2 : float = 1):

        """
        location : location for lobsvm file,
        name : name of dataset
        m : number of linear constraints
        b_2 : constant for the norm constraint
        constraint_seed : seed for generating the linear constraint
        """

        self._number_of_classes = 2
        self._dataset_name = name
        self.logistic_function = Cross_Entropy_Binary(location=location, name=name)

        super().__init__(name=f"{name}_cross_entropy_logistic_norm_constraint", d=self.logistic_function.d, number_of_datapoints=self.logistic_function.number_of_datapoints, eq_const_number=1, ineq_const_number=0)

        self._b_2 = b_2

    def initial_point(self) -> np.ndarray:
        return np.ones((self.d, 1))

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def number_of_classes(self) -> int:
        return self._number_of_classes

    @property
    def number_of_features(self) -> int:
        return self.logistic_function._number_of_features

    def objective(self, x: np.array, type: str, batch_size: int = 0, seed: int | None = None, data_indices : list | None = None) -> float:
        return self.logistic_function.objective(x=x, type = type, batch_size=batch_size, seed=seed, data_indices=data_indices)

    def gradient(self, x: np.array, type: str, batch_size: int = 0, seed: int | None = None, data_indices : list | None = None) -> np.ndarray:
        return self.logistic_function.gradient(x=x, type=type, batch_size=batch_size, seed=seed, data_indices=data_indices)

    def hessian(self, x: np.array, type: str, batch_size: int = 0, seed: int | None = None, data_indices : list | None = None) -> np.array:
        raise Exception(f"Can't compute hessian for {self.name}")

    def constraints_eq(self, x: np.array) -> np.array:
        return np.array([[np.linalg.norm(x)**2 - self._b_2]])

    def constraints_eq_jacobian(self, x: np.array) -> np.array:
        return  2*x.T

    def constraints_eq_hessian(self, x: np.array) -> np.array:
        hessians = []
        hessians.append(np.identity(self.d))

        return np.array(hessians)

    def constraints_ineq(self, x: np.array) -> np.array:
        raise Exception("No inequality constraints")
    
    def constraints_ineq_jacobian(self, x: np.array) -> np.array:
        raise Exception("No inequality constraints")
    
    def constraints_ineq_hessian(self, x: np.array) -> np.array:
        raise Exception("No inequality constraints")