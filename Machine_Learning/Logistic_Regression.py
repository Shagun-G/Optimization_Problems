import numpy as np
from Base_classes import Unconstrained
from sklearn.datasets import load_svmlight_file
from scipy.sparse import vstack


# TODO : Function Descriptions
# TODO : Objective evalutation of binary cross entropy
# TODO : Gradient calculation change to binary corss entropy
class Cross_Entropy_Binary(Unconstrained):
    def __init__(self, location: str, name: str, sparse_format: bool = False):
        """
        # regularization with #datapoints
        Inputs:
        name        :   name of the dataset
        location    :   location of libsvm file to create logictic regression file
        sparse      :   if want data to be stored in sparse matrix format

        object attributes:
        _number_of_features : includes the bias term
        _feasures           : feature dataset (number of features x number of datapoints),
                            includes bias term constant
        _targets            : target labels ({0,1} x number of datapoints)
        _sparse             : (bool) of data stored in sparse format
        """

        self._sparse = sparse_format
        self._number_of_classes = 2

        X, y = load_svmlight_file(location)

        # preprocessing for specific datasets

        if name.lower() == "mushroom":
            # the target variable needs to be offset
            y = y - 1

        elif name.lower() == "australian":
            # the target has to be changed from {-1, +1} to {0, 1}
            y[y == -1] = 0
        elif name.lower() == "phishing":
            # no formatting required, {0,1} labels
            pass
        elif name.lower() == "sonar":
            # the target has to be changed from {-1, +1} to {0, 1}
            y[y == -1] = 0
        elif name.lower() == "gisette":
            # the target has to be changed from {-1, +1} to {0, 1}
            y[y == -1] = 0
        else:
            raise Exception("Unknown dataset, preprocessing might be required for correct format")

        self._number_of_datapoints, self._number_of_features = np.shape(X)
        y = y.reshape((self._number_of_datapoints, 1))  # reshaping target matrix
        # adding bias term to features
        if sparse_format:
            X = vstack((X.T, np.ones((1, self._number_of_datapoints))))
        else:
            X = np.vstack((X.toarray().T, np.ones((1, self._number_of_datapoints))))

        self._number_of_features += 1
        self._features = X
        self._targets = y
        super().__init__(name=f"{name}_cross_entropy_logistic", d=self._number_of_features)

    def initial_point(self) -> np.ndarray:
        return np.zeros([self._number_of_features, 1])

    def _determine_batch(self, type: str = "full", batch_size: int = 0, seed: int | None = None) -> np.array:
        if type == "full":
            s = np.arange(self._number_of_datapoints)
            batch_size = self._number_of_datapoints

            return s

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
            return s

        raise Exception(f"{type} is not a defined type of gradient")

    def objective(self, x: np.array, type: str = "full", batch_size: int = 0, seed: int | None = None) -> float:
        """Evaluates MLE loss"""

        s = self._determine_batch(type, batch_size, seed)

        # signmoid calculate

        ET = np.exp(np.dot(x.T, self._features[:, s]))

    def gradient(self, x: np.array, type: str = "full", batch_size: int = 0, seed: int | None = None) -> np.ndarray:
        """MLE Loss gradient"""

        s = self._determine_batch(type, batch_size, seed)

        if self._sparse:
            """Sparse matrix calculation"""
            x = x.reshape((self._number_of_features, 1))

            ET = np.exp(np.dot(x.T, self._features[:, s]))
            temp = np.sum(ET, axis=0) + 1
            P = ET / temp
            g = (
                np.dot(self._features[:, s], (P - self._targets[s, :].T).T) / batch_size
                + 2 * x / self._number_of_datapoints
            )

            return g.reshape((self._number_of_features, 1))

        """Non sparse calculation"""
        x = x.reshape((self._number_of_features, 1))

        ET = np.exp(np.dot(x.T, self._features[:, s]))
        temp = np.sum(ET, axis=0) + 1
        P = ET / temp
        g = (
            np.dot(self._features[:, s], (P - self._targets[s, :].T).T) / batch_size
            + 2 * x / self._number_of_datapoints
        )

        return g.reshape((self._number_of_features, 1))

    def hessian(self, x: np.array) -> np.array:
        raise Exception(f"can't compute hessian for {self.name}")


# TODO : Multi class logistic regression and mnist and some other in that
