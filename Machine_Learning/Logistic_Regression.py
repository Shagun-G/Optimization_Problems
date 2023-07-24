import numpy as np
from Base_classes import Unconstrained
from sklearn.datasets import load_svmlight_file
from scipy.sparse import vstack

# TODO : Multi class logistic regression and mnist and some other in that


# TODO : Objective evalutation
# TODO : Gradient calculation change to new format and use seed an rng generator
class Cross_Entropy_Binary(Unconstrained):
    def __init__(self, location: str, name: str | None = None, sparse_format: bool = False) -> None:
        """
        name    :   name for the dataset
        location    :   location of libsvm file to create logictic regression file
        sparse  :   if want data to be returned and stored in sparse matrix format

        object attributes:
        _number_of_features : includes +1 for bias term
        _feasures : feature dataset (number of features x number of datapoints), includes bias term constant
        _targets : target labels ({0,1} x number of datapoints)
        _sparse : (bool) of data stored in sparse format
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
        return np.zeros([self.number_features * (self.number_classes - 1), 1])

    def objective(self, x) -> float:
        """Evaluates MLE loss"""
        pass

    def gradient(self, x, type, factor=None) -> np.ndarray:
        """MLE Loss gradient"""

        if type == "full":
            s = np.arange(self.number_datapoints)
            batch_size = self.number_datapoints

        elif type == "stochastic":
            if isinstance(factor, int):
                # create a minibath of size factor
                s = np.random.choice(self.number_datapoints, size=(factor), replace=False)

            elif isinstance(factor, np.ndarray):
                # prespecified minibatch
                s = factor
            else:
                raise Exception("Specify either batch size or batch for stochastic gradient")

            batch_size = np.shape(s)[0]
            if batch_size > self.number_datapoints:
                raise Exception("Batch size specified is larger than size of dataset")

        else:
            raise Exception("{} is not a defined type of gradient".format(type))

        x = x.reshape(
            (self.number_features, self.number_classes - 1)
        )  # we consider only K-1 classes to reduce dimension in logistic regreesion

        ET = np.exp(np.dot(x.T, self.data[:, s]))
        temp = np.sum(ET, axis=0) + 1
        P = ET / temp

        g = (
            np.dot(self.data[:, s], (P - self.labels[:, s]).T) / batch_size + 2 * x / self.number_datapoints
        )  # regularization with # datapoints

        return g.reshape((self.number_features * (self.number_classes - 1), 1))

    def hessian(self, x: np.array) -> np.array:
        raise Exception(f"can't compute hessian for {self.name}")
