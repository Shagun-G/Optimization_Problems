import numpy as np
from Opt_Problems.Options import Datasets
from sklearn.datasets import load_svmlight_file

def create_rng(seed: int):
    if seed is None:
        rng = np.random.default_rng(
            np.random.randint(np.iinfo(np.int16).max, size=1)[0]
        )
    else:
        rng = np.random.default_rng(seed)

    return rng

def generate_linear_constraints(m: int, d: int, rng: np.random.Generator):
    A = rng.standard_normal(size=(m, d))
    b = rng.standard_normal(size=(m, 1))
    return A, b

def generate_quadratic_problem(d: int, xi: int, rng: np.random.Generator):
    """Generates a quadratic based on the process in Numerical Experiments in:
    A. Mokhtari, Q. Ling and A. Ribeiro, "Network Newton Distributed Optimization Methods," in IEEE Transactions on Signal Processing, vol. 65, no. 1, pp. 146-161, 1 Jan.1, 2017, doi: 10.1109/TSP.2016.2617829.
    """
    s1 = 10 ** np.arange(xi)
    s2 = 1 / 10 ** np.arange(xi)
    if d % 2 == 0:
        v = np.hstack((rng.choice(s1, size=int(d / 2)), rng.choice(s2, size=int(d / 2))))
    else:
        v = np.hstack((rng.choice(s1, size=int(d / 2) + 1), rng.choice(s2, size=int(d / 2))))
    b = rng.random((d, 1)) * 10 ** (int(xi / 2))
    return np.diag(v), b

def generate_stochastic_batch(n, batch_size, rng):
    if n > 1e20:
        n = int(1e10)
    s = rng.choice(n, size=(batch_size), replace=False)
    return s


def datasets_manager(dataset_name: Datasets, location:str):

    if dataset_name.value not in location:
        raise Exception("Name and file pointed to in location are different")

    X, y = load_svmlight_file(location)

    if dataset_name not in Datasets.list():
        raise Exception("Dataset processing not available")

    # preprocessing for specific datasets
    if dataset_name is Datasets.Mushroom:
        # the target variable needs to be offset
        y = y - 1

    if dataset_name in [Datasets.Australian, Datasets.Phishing, Datasets.Sonar, Datasets.Gisette, Datasets.A9a, Datasets.W8a, Datasets.Ijcnn, Datasets.RealSim]:
        # the target has to be changed from {-1, +1} to {0, 1}
        y[y == -1] = 0

    '''Normalization being done now at the problem formulation'''

    return X, y