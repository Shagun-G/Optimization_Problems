import numpy as np
from Base_classes import Unconstrained
from sklearn.datasets import load_svmlight_file
from scipy import sparse

class Cross_Entropy_Binary(Unconstrained):
    def __init__(self, location: str, name: str):
        """
        # regularization with #datapoints
        Inputs:
        name        :   name of the dataset
        location    :   location of libsvm file to create logictic regression file

        object attributes:
        _number_of_features : number of features, includes the bias term
        _number_of_classes  : number of classes
        _features           : feature dataset (number of features x number of datapoints),
                            includes bias term constant
        _targets            : target labels ({0,1} x number of datapoints)
        """

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
        elif name.lower() == "a9a":
            # the target has to be changed from {-1, +1} to {0, 1}
            y[y == -1] = 0
        elif name.lower() == "w8a":
            # the target has to be changed from {-1, +1} to {0, 1}
            y[y == -1] = 0
        else:
            raise Exception("Unknown dataset, preprocessing might be required for correct format")

        if name not in location:
            raise Exception("Name and file pointed to in location are different")

        self._number_of_datapoints, self._number_of_features = np.shape(X)
        y = y.reshape((self._number_of_datapoints, 1))  # reshaping target matrix
        # adding bias term to features
        X = np.vstack((X.toarray().T, np.ones((1, self._number_of_datapoints))))

        self._number_of_features += 1
        self._features = X
        self._targets = y
        super().__init__(name=f"{name}_cross_entropy_logistic", d=self._number_of_features)

    def initial_point(self) -> np.ndarray:
        return np.zeros([self._number_of_features, 1])

    @property
    def number_of_datapoints(self) -> int:
        return self._number_of_datapoints

    @property
    def number_of_classes(self) -> int:
        return self._number_of_classes

    @property
    def number_of_features(self) -> int:
        return self._number_of_features

    def _determine_batch(self, type: str, batch_size: int = 0, seed: int | None = None) -> np.array:
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

    def objective(self, x: np.array, type: str, batch_size: int = 0, seed: int | None = None) -> float:
        """
        Calculates loss for full or a stochastic batch of data
        Inputs:
        x           :   logistic regression parameters
        type        :   "full" for full data and "stochastic" for batch
        batch_size  :   (optional) only used when stochastic type of batch
        seed        :   (optional) if not specified, random batch, else specify for the same batch of data
        """
        """Evaluates MLE loss"""

        s, batch_size = self._determine_batch(type, batch_size, seed)

        # signmoid calculate
        exp_neg = np.exp(-np.dot(x.T, self._features[:, s])).T
        y_hat = 1 / (1 + exp_neg)

        # cross entropy loss
        loss = -(self._targets[s, :] * np.log(y_hat) + (1 - self._targets[s, :]) * np.log(1 - y_hat))

        # replace nan with 0 to for 0*log(0) values
        loss[np.isnan(loss)] = 0
        return np.sum(loss)

    def gradient(self, x: np.array, type: str, batch_size: int = 0, seed: int | None = None) -> np.ndarray:
        """MLE Loss gradient"""
        """
        Calculates gradient of loss for full or a stochastic batch of data
        Inputs:
        x           :   logistic regression parameters
        type        :   "full" for full data and "stochastic" for batch
        batch_size  :   (optional) only used when stochastic type of batch
        seed        :   (optional) if not specified, random batch, else specify for the same batch of data
        """
        
        s, batch_size = self._determine_batch(type, batch_size, seed)
        x = x.reshape((self._number_of_features, 1))

        """Non sparse calculation"""
        exp_neg = np.exp(-np.dot(x.T, self._features[:, s])).T
        y_hat = 1 / (1 + exp_neg)
        g = (
            np.dot(self._features[:, s], (y_hat - self._targets[s, :])) / batch_size
            + 2 * x / self._number_of_datapoints
        )

        return g.reshape((self._number_of_features, 1))

    def hessian(self, x: np.array) -> np.array:
        raise Exception(f"Can't compute hessian for {self.name}")


# TODO : Constructor function
# TODO : Add mnist and iris datasets
# TODO : objective calculation
# TODO : gradient calculation
# TODO : Function descriptions update
class Cross_Entropy_Multiclass(Unconstrained):
    def __init__(self, location: str, name: str):
        """
        # regularization with #datapoints
        Inputs:
        name        :   name of the dataset
        location    :   location of libsvm file to create logictic regression file

        object attributes:
        _number_of_features : number of features, includes the bias term
        _number_of_classes  : number of classes
        _features           : feature dataset (number of features x number of datapoints),
                            includes bias term constant
        _targets            : target labels ({0,1} x number of datapoints)
        """

        self._number_of_classes = 2

        X, y = load_svmlight_file(location)

        # preprocessing for specific datasets

        if name.lower() == "mnist":
            # the target variable needs to be offset
            y = y - 1
        else:
            raise Exception("Unknown dataset, preprocessing might be required for correct format")

        self._number_of_datapoints, self._number_of_features = np.shape(X)
        y = y.reshape((self._number_of_datapoints, 1))  # reshaping target matrix
        # adding bias term to features
        X = np.vstack((X.toarray().T, np.ones((1, self._number_of_datapoints))))

        self._number_of_features += 1
        self._features = X
        self._targets = y
        super().__init__(name=f"{name}_cross_entropy_logistic", d=self._number_of_features)

    def initial_point(self) -> np.ndarray:
        return np.zeros([self._number_of_features, 1])

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
        """
        Calculates loss for full or a stochastic batch of data
        Inputs:
        x           :   logistic regression parameters
        type        :   "full" for full data and "stochastic" for batch
        batch_size  :   (optional) only used when stochastic type of batch
        seed        :   (optional) if not specified, random batch, else specify for the same batch of data
        """
        """Evaluates MLE loss"""

        s = self._determine_batch(type, batch_size, seed)

        # signmoid calculate
        exp_neg = np.exp(-np.dot(x.T, self._features[:, s])).T
        y_hat = 1 / (1 + exp_neg)

        # cross entropy loss
        loss = -(self._targets[s, :] * np.log(y_hat) + (1 - self._targets[s, :]) * np.log(1 - y_hat))

        # replace nan with 0 to for 0*log(0) values
        loss[np.isnan(loss)] = 0
        return np.sum(loss)

    def gradient(self, x: np.array, type: str = "full", batch_size: int = 0, seed: int | None = None) -> np.ndarray:
        """MLE Loss gradient"""
        """
        Calculates gradient of loss for full or a stochastic batch of data
        Inputs:
        x           :   logistic regression parameters
        type        :   "full" for full data and "stochastic" for batch
        batch_size  :   (optional) only used when stochastic type of batch
        seed        :   (optional) if not specified, random batch, else specify for the same batch of data
        """

        s = self._determine_batch(type, batch_size, seed)
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
        raise Exception(f"Can't compute hessian for {self.name}")
