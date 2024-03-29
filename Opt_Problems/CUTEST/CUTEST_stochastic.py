# TODO : decreasing noise towards optimal

import numpy as np
from numpy.core.multiarray import array as array
from Opt_Problems.utils import Problem, Unconstrained_Problem
import pycutest
from Opt_Problems.CUTEST.CUTEST_deterministic import CUTEST_constrained, CUTEST_unconstrained

# TODO : increasing noise towards optimal
class CUTEST_constrained_stochastic_with_start_point(Problem):


    '''Constructs a stochastic constrained CUTEST problem with the added noise to the objective in the form of f(x) + xi * scale * ||x - x_0||^2/2 where xi follows normal distribution with mean 0 and specified standard deviation'''

    def __init__(self, name: str, std_dev : float, scale : float = 1):

        """import problem from cutest dataset"""
        self._cutest_object = CUTEST_constrained(name=name)
        super().__init__(name=name, d=self._cutest_object.d, eq_const_number=self._cutest_object.number_of_eq_constraints, ineq_const_number=self._cutest_object.number_of_ineq_constraints, number_of_datapoints=np.inf)
        self._std = std_dev
        self._scale = scale


    def print_parameters(self):
        """print parameters of the cutest problem"""
        print(pycutest.problem_properties(self.name))
        print(pycutest.print_available_sif_params(self.name))

    def initial_point(self) -> np.array:
        return self._cutest_object.initial_point() 

    def objective(
        self,
        x: np.array,
        type: str,
        batch_size: int = 0,
        seed: int | None = None,
        data_indices: list | None = None,
    ) -> float:
        if type == "full":
            return self._cutest_object.objective(x)
        elif type == "stochastic":
            xi = self.sample_xi(batch_size=batch_size, seed=seed, seed_list=data_indices)
            return self._cutest_object.objective(x) + xi*np.linalg.norm(x - self.initial_point())**2/2
        else:
            raise Exception("Type specified for objective function is not available")

    def gradient(
        self,
        x: np.array,
        type: str,
        batch_size: int = 0,
        seed: int | None = None,
        data_indices: list | None = None,
    ) -> np.ndarray:
        if type == "full":
            return self._cutest_object.gradient(x)
        elif type == "stochastic":
            xi = self.sample_xi(batch_size=batch_size, seed=seed, seed_list=data_indices)
            return self._cutest_object.gradient(x) + xi*(x - self.initial_point())
        else:
            raise Exception("Type specified for gradient function is not available")

    def hessian(
        self,
        x: np.array,
        type: str,
        batch_size: int = 0,
        seed: int | None = None,
        data_indices: list | None = None,
    ) -> np.array:
        if type == "full":
            return self._cutest_object.hessian(x)
        elif type == "stochastic":
            xi = self.sample_xi(batch_size=batch_size, seed=seed, seed_list=data_indices)
            return self._cutest_object.hessian(x) + xi*np.identity(self.d)
        else:
            raise Exception("Type specified for hessian function is not available")

    def constraints_eq(self, x: np.array) -> np.array:
        return self._cutest_object.constraints_eq(x)

    def constraints_eq_jacobian(self, x: np.array) -> np.array:
        return self._cutest_object.constraints_eq_jacobian(x)

    def constraints_eq_hessian(self, x: np.array) -> np.array:
        return self._cutest_object.constraints_eq_hessian(x)

    def constraints_ineq(self, x: np.array) -> np.array:
        return self._cutest_object.constraints_ineq(x)

    def constraints_ineq_jacobian(self, x: np.array) -> np.array:
        return self._cutest_object.constraints_ineq_jacobian(x)

    def constraints_ineq_hessian(self, x: np.array) -> np.array:
        return self._cutest_object.constraints_ineq_hessian(x)


    def sample_xi(
        self, batch_size: int = 0, seed: int | None = None, seed_list: list | None = None,
    ) -> np.array:
        """
        Generates an array of indices for a batch of data for calculation
        Inputs:
        batch_size  :   only used when stochastic type of batch
        seed        :   (optional) if not specified, random batch, else specify for the same batch of data
        seed_list   :   list of seeds to simulate list of data samples being spoecified
        """

        if seed_list == None:
            if seed is None:
                rng = np.random.default_rng(
                    np.random.randint(np.iinfo(np.int16).max, size=1)[0]
                )
            elif isinstance(seed, int):
                rng = np.random.default_rng(seed)
            else:
                raise Exception("seed must be an integer if specified")

            if batch_size < 0:
                raise Exception(f"{type} gradient requires a batch_size > 0")

            s =  rng.standard_normal(size=(batch_size, 1))
        else:
            xi_list = []
            for seed in seed_list:
                if isinstance(seed, int):
                    rng = np.random.default_rng(seed)
                else:
                    raise Exception("seed must be an integer if specified")

                xi_list.append(rng.standard_normal(size=(1)).squeeze())

        return self._std * self._scale * np.average(s)



# TODO : increasing noise towards optimal
class CUTEST_unconstrained_stochastic_with_start_point(Unconstrained_Problem):


    '''Constructs a stochastic unconstrained CUTEST problem with the added noise to the objective in the form of f(x) + xi * scale * ||x - x_0||^2/2 where xi follows normal distribution with mean 0 and specified standard deviation'''

    def __init__(self, name: str, std_dev : float, scale : float = 1):

        """import problem from cutest dataset"""
        self._cutest_object = CUTEST_unconstrained(name=name)
        super().__init__(name=name, d=self._cutest_object.d, number_of_datapoints=np.inf)
        self._std = std_dev
        self._scale = scale

    def print_parameters(self):
        """print parameters of the cutest problem"""
        print(pycutest.problem_properties(self.name))
        print(pycutest.print_available_sif_params(self.name))

    def initial_point(self) -> np.array:
        return self._cutest_object.initial_point() 

    def objective(
        self,
        x: np.array,
        type: str,
        batch_size: int = 0,
        seed: int | None = None,
        data_indices: list | None = None,
    ) -> float:
        if type == "full":
            return self._cutest_object.objective(x)
        elif type == "stochastic":
            xi = self.sample_xi(batch_size=batch_size, seed=seed, seed_list=data_indices)
            return self._cutest_object.objective(x) + xi*np.linalg.norm(x - self.initial_point())**2/2
        else:
            raise Exception("Type specified for objective function is not available")

    def gradient(
        self,
        x: np.array,
        type: str,
        batch_size: int = 0,
        seed: int | None = None,
        data_indices: list | None = None,
    ) -> np.ndarray:
        if type == "full":
            return self._cutest_object.gradient(x)
        elif type == "stochastic":
            xi = self.sample_xi(batch_size=batch_size, seed=seed, seed_list=data_indices)
            return self._cutest_object.gradient(x) + xi*(x - self.initial_point())
        else:
            raise Exception("Type specified for gradient function is not available")

    def hessian(
        self,
        x: np.array,
        type: str,
        batch_size: int = 0,
        seed: int | None = None,
        data_indices: list | None = None,
    ) -> np.array:
        if type == "full":
            return self._cutest_object.hessian(x)
        elif type == "stochastic":
            xi = self.sample_xi(batch_size=batch_size, seed=seed, seed_list=data_indices)
            return self._cutest_object.hessian(x) + xi*np.identity(self.d)
        else:
            raise Exception("Type specified for hessian function is not available")

    def sample_xi(
        self, batch_size: int = 0, seed: int | None = None, seed_list: list | None = None,
    ) -> np.array:
        """
        Generates an array of indices for a batch of data for calculation
        Inputs:
        batch_size  :   only used when stochastic type of batch
        seed        :   (optional) if not specified, random batch, else specify for the same batch of data
        seed_list   :   list of seeds to simulate list of data samples being spoecified
        """

        if seed_list == None:
            if seed is None:
                rng = np.random.default_rng(
                    np.random.randint(np.iinfo(np.int16).max, size=1)[0]
                )
            elif isinstance(seed, int):
                rng = np.random.default_rng(seed)
            else:
                raise Exception("seed must be an integer if specified")

            if batch_size < 0:
                raise Exception(f"{type} gradient requires a batch_size > 0")

            s =  rng.standard_normal(size=(batch_size, 1))
        else:
            xi_list = []
            for seed in seed_list:
                if isinstance(seed, int):
                    rng = np.random.default_rng(seed)
                else:
                    raise Exception("seed must be an integer if specified")

                xi_list.append(rng.standard_normal(size=(1)).squeeze())

        return self._std * self._scale * np.average(s)