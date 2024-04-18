import numpy as np
from Opt_Problems.utils import datasets_manager, Unconstrained_Problem
from scipy.special import expit


class Cross_Entropy_Binary(Unconstrained_Problem):
    def __init__(self, location: str, name: str, test_data_location: str = ""):
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
        self._dataset_name = name
        X, y = datasets_manager(name=name, location=location)
        if test_data_location != "":
            self._test_data = True
            X_test, y_test = datasets_manager(name=name, location=test_data_location)
            self._number_of_test_points, _ = np.shape(X_test)
            y_test = y_test.reshape(
                (self._number_of_test_points, 1)
            )  # reshaping target matrix
            X_test = X_test.toarray()
        else:
            self._test_data = False

        number_of_datapoints, self._number_of_features = np.shape(X)
        y = y.reshape((number_of_datapoints, 1))  # reshaping target matrix
        X = X.toarray()
        ptp = X.ptp(0)
        mins_data = X.min(0)
        X = (X - mins_data) / ptp  # normalizing
        X = np.nan_to_num(X, nan=0)  # replacing divide by 0 with 0
        X = np.vstack(
            (X.T, np.ones((1, number_of_datapoints)))
        )  # adding bias term to features
        self._number_of_features += 1
        self._features = X
        self._targets = y

        if self._test_data:
            X_test = (X_test - mins_data) / ptp  # normalizing
            X_test = np.nan_to_num(X_test, nan=0)  # replacing divide by 0 with 0
            X_test = np.vstack(
                (X_test.T, np.ones((1, self._number_of_test_points)))
            )  # adding bias term to features
            self._features_test = X_test
            self._targets_test = y_test

        super().__init__(
            name=f"{name}_cross_entropy_logistic",
            d=self._number_of_features,
            number_of_datapoints=number_of_datapoints,
        )

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
        return self._number_of_features

    def _determine_batch(
        self, type: str, batch_size: int = 0, seed: int | None = None
    ) -> np.array:
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
                rng = np.random.default_rng(
                    np.random.randint(np.iinfo(np.int16).max, size=1)[0]
                )
            elif isinstance(seed, int):
                rng = np.random.default_rng(seed)
            else:
                raise Exception("seed must be an integer if specified")

            if batch_size < 0:
                raise Exception(f"{type} gradient requires a batch_size > 0")

            if batch_size > self._number_of_datapoints:
                batch_size = self._number_of_datapoints
                # raise Exception("Batch size specified is larger than size of dataset")

            s = rng.choice(self._number_of_datapoints, size=(batch_size), replace=False)
            return s, batch_size

        raise Exception(f"{type} is not a defined type of gradient")

    def objective(
        self,
        x: np.array,
        type: str,
        batch_size: int = 0,
        seed: int | None = None,
        data_indices: list | None = None,
    ) -> float:
        """
        Calculates loss for full or a stochastic batch of data
        Inputs:
        x           :   logistic regression parameters
        type        :   "full" for full data and "stochastic" for batch
        batch_size  :   (optional) only used when stochastic type of batch
        seed        :   (optional) if not specified, random batch, else specify for the same batch of data
        """
        """Evaluates MLE loss"""

        if data_indices == None:
            s, batch_size = self._determine_batch(type, batch_size, seed)
        else:
            batch_size = len(data_indices)
            s = np.array(data_indices)

        # signmoid calculate
        y_hat = expit(np.dot(x.T, self._features[:, s])).T

        # cross entropy loss
        loss = -(
            self._targets[s, :] * np.log(y_hat)
            + (1 - self._targets[s, :]) * np.log(1 - y_hat)
        )

        # replace nan with 0 to for 0*log(0) values
        loss[np.isnan(loss)] = 0
        return np.mean(loss) + np.linalg.norm(x) ** 2 / self.number_of_datapoints

    def gradient(
        self,
        x: np.array,
        type: str,
        batch_size: int = 0,
        seed: int | None = None,
        data_indices: list | None = None,
    ) -> np.ndarray:
        """MLE Loss gradient"""
        """
        Calculates gradient of loss for full or a stochastic batch of data
        Inputs:
        x           :   logistic regression parameters
        type        :   "full" for full data and "stochastic" for batch
        batch_size  :   (optional) only used when stochastic type of batch
        seed        :   (optional) if not specified, random batch, else specify for the same batch of data
        """

        if data_indices == None:
            s, batch_size = self._determine_batch(type, batch_size, seed)
        else:
            batch_size = len(data_indices)
            s = np.array(data_indices)

        x = x.reshape((self._number_of_features, 1))

        """Non sparse calculation"""
        y_hat = expit(np.dot(x.T, self._features[:, s])).T
        g = (
            np.dot(self._features[:, s], (y_hat - self._targets[s, :])) / batch_size
            + 2 * x / self._number_of_datapoints
        )

        return g.reshape((self._number_of_features, 1))

    def hessian(
        self,
        x: np.array,
        type: str,
        batch_size: int = 0,
        seed: int | None = None,
        data_indices: list | None = None,
    ) -> np.array:
        raise Exception(f"Can't compute hessian for {self.name}")

    def objective_test(
        self,
        x: np.array,
    ) -> float:
        """
        Calculates loss for full test data
        Inputs:
        x           :   logistic regression parameters
        """
        """Evaluates MLE loss"""

        if not self._test_data:
            raise Exception("No test Data available")

        # signmoid calculate
        y_hat = expit(np.dot(x.T, self._features_test)).T

        # cross entropy loss
        loss = -(
            self._targets_test * np.log(y_hat)
            + (1 - self._targets_test) * np.log(1 - y_hat)
        )

        # replace nan with 0 to for 0*log(0) values
        loss[np.isnan(loss)] = 0
        return np.mean(loss)

    def accuracy_test(
        self,
        x: np.array,
    ) -> float:
        """
        Calculates accuracy for full test data
        Inputs:
        x           :   logistic regression parameters
        """

        if not self._test_data:
            raise Exception("No test Data available")

        # signmoid calculate
        y_hat = expit(np.dot(x.T, self._features_test)).T

        # cross entropy loss
        accuracy = 1 * (y_hat > 0.5) - self._targets_test

        return 1 - np.mean(np.abs(accuracy))


class Huber_Loss_Binary(Unconstrained_Problem):
    def __init__(self, location: str, name: str, test_data_location: str = ""):
        """
        Logistic regression over the robust regression huber loss t^2 / (1 + t^2)

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
        self._dataset_name = name
        X, y = datasets_manager(name=name, location=location)
        if test_data_location != "":
            self._test_data = True
            X_test, y_test = datasets_manager(name=name, location=test_data_location)
            self._number_of_test_points, _ = np.shape(X_test)
            y_test = y_test.reshape(
                (self._number_of_test_points, 1)
            )  # reshaping target matrix
            X_test = X_test.toarray()
        else:
            self._test_data = False

        number_of_datapoints, self._number_of_features = np.shape(X)
        y = y.reshape((number_of_datapoints, 1))  # reshaping target matrix
        X = X.toarray()
        ptp = X.ptp(0)
        mins_data = X.min(0)
        X = (X - mins_data) / ptp  # normalizing
        X = np.nan_to_num(X, nan=0)  # replacing divide by 0 with 0
        X = np.vstack(
            (X.T, np.ones((1, number_of_datapoints)))
        )  # adding bias term to features
        self._number_of_features += 1
        self._features = X
        self._targets = y

        if self._test_data:
            X_test = (X_test - mins_data) / ptp  # normalizing
            X_test = np.nan_to_num(X_test, nan=0)  # replacing divide by 0 with 0
            X_test = np.vstack(
                (X_test.T, np.ones((1, self._number_of_test_points)))
            )  # adding bias term to features
            self._features_test = X_test
            self._targets_test = y_test

        super().__init__(
            name=f"{name}_huber__logistic",
            d=self._number_of_features,
            number_of_datapoints=number_of_datapoints,
        )

    def initial_point(self) -> np.ndarray:
        return np.zeros((self.d, 1))

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def number_of_classes(self) -> int:
        return self._number_of_classes

    @property
    def number_of_features(self) -> int:
        return self._number_of_features

    def _determine_batch(
        self, type: str, batch_size: int = 0, seed: int | None = None
    ) -> np.array:
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
                rng = np.random.default_rng(
                    np.random.randint(np.iinfo(np.int16).max, size=1)[0]
                )
            elif isinstance(seed, int):
                rng = np.random.default_rng(seed)
            else:
                raise Exception("seed must be an integer if specified")

            if batch_size < 0:
                raise Exception(f"{type} gradient requires a batch_size > 0")

            if batch_size > self._number_of_datapoints:
                batch_size = self._number_of_datapoints
                # raise Exception("Batch size specified is larger than size of dataset")

            s = rng.choice(self._number_of_datapoints, size=(batch_size), replace=False)
            return s, batch_size

        raise Exception(f"{type} is not a defined type of gradient")

    def objective(
        self,
        x: np.array,
        type: str,
        batch_size: int = 0,
        seed: int | None = None,
        data_indices: list | None = None,
    ) -> float:
        """
        Calculates loss for full or a stochastic batch of data
        Inputs:
        x           :   logistic regression parameters
        type        :   "full" for full data and "stochastic" for batch
        batch_size  :   (optional) only used when stochastic type of batch
        seed        :   (optional) if not specified, random batch, else specify for the same batch of data
        """
        """Evaluates MLE loss"""

        if data_indices == None:
            s, batch_size = self._determine_batch(type, batch_size, seed)
        else:
            batch_size = len(data_indices)
            s = np.array(data_indices)

        # signmoid calculate
        error = self._targets[s, :] - expit(np.dot(x.T, self._features[:, s])).T

        # cross entropy loss
        loss = np.sum(np.square(error) / (1 + np.square(error))) / batch_size

        return loss

    def gradient(
        self,
        x: np.array,
        type: str,
        batch_size: int = 0,
        seed: int | None = None,
        data_indices: list | None = None,
    ) -> np.ndarray:
        """MLE Loss gradient"""
        """
        Calculates gradient of loss for full or a stochastic batch of data
        Inputs:
        x           :   logistic regression parameters
        type        :   "full" for full data and "stochastic" for batch
        batch_size  :   (optional) only used when stochastic type of batch
        seed        :   (optional) if not specified, random batch, else specify for the same batch of data
        """

        if data_indices == None:
            s, batch_size = self._determine_batch(type, batch_size, seed)
        else:
            batch_size = len(data_indices)
            s = np.array(data_indices)

        x = x.reshape((self._number_of_features, 1))

        """Non sparse calculation"""
        sigmoid = expit(np.dot(x.T, self._features[:, s])).T
        error = self._targets[s, :] - sigmoid
        g = (
            2
            * (
                np.dot(
                    self._features[:, s],
                    (
                        (error / np.square((1 + np.square(error))))
                        * sigmoid
                        * (sigmoid - 1)
                    ),
                )
            )
            / batch_size
        )

        return g.reshape((self._number_of_features, 1))

    def hessian(
        self,
        x: np.array,
        type: str,
        batch_size: int = 0,
        seed: int | None = None,
        data_indices: list | None = None,
    ) -> np.array:
        raise Exception(f"Can't compute hessian for {self.name}")

    def objective_test(
        self,
        x: np.array,
    ) -> float:
        """
        Calculates loss for full test data
        Inputs:
        x           :   logistic regression parameters
        """
        """Evaluates MLE loss"""

        if not self._test_data:
            raise Exception("No test Data available")

        # signmoid calculate
        error = self._targets_test - expit(np.dot(x.T, self._features_test)).T

        # cross entropy loss
        loss = (
            np.sum(np.square(error) / (1 + np.square(error)))
            / self._number_of_test_points
        )

        return loss

    def accuracy_test(
        self,
        x: np.array,
    ) -> float:
        """
        Calculates accuracy for full test data
        Inputs:
        x           :   logistic regression parameters
        """

        if not self._test_data:
            raise Exception("No test Data available")

        # signmoid calculate
        y_hat = expit(np.dot(x.T, self._features_test)).T

        # cross entropy loss
        accuracy = 1 * (y_hat > 0.5) - self._targets_test

        return 1 - np.mean(np.abs(accuracy))


# TODO : Constructor function
# TODO : Add mnist and iris datasets
# TODO : objective calculation
# TODO : gradient calculation
# TODO : Function descriptions update
# class Cross_Entropy_Multiclass(Unconstrained_Problem):
# pass
# def __init__(self, location: str, name: str):
#     """
#     # regularization with #datapoints
#     Inputs:
#     name        :   name of the dataset
#     location    :   location of libsvm file to create logictic regression file

#     object attributes:
#     _number_of_features : number of features, includes the bias term
#     _number_of_classes  : number of classes
#     _features           : feature dataset (number of features x number of datapoints),
#                         includes bias term constant
#     _targets            : target labels ({0,1} x number of datapoints)
#     """

#     self._number_of_classes = 2

#     X, y = load_svmlight_file(location)

#     # preprocessing for specific datasets

#     if name.lower() == "mnist":
#         # the target variable needs to be offset
#         y = y - 1
#     else:
#         raise Exception("Unknown dataset, preprocessing might be required for correct format")

#     self._number_of_datapoints, self._number_of_features = np.shape(X)
#     y = y.reshape((self._number_of_datapoints, 1))  # reshaping target matrix
#     # adding bias term to features
#     X = np.vstack((X.toarray().T, np.ones((1, self._number_of_datapoints))))

#     self._number_of_features += 1
#     self._features = X
#     self._targets = y
#     super().__init__(name=f"{name}_cross_entropy_logistic", d=self._number_of_features)

# def initial_point(self) -> np.ndarray:
#     return np.zeros([self._number_of_features, 1])

# def _determine_batch(self, type: str = "full", batch_size: int = 0, seed: int | None = None) -> np.array:
#     """
#     Generates an array of indices for a batch of data for calculation
#     Inputs:
#     type        :   "full" for full data and "stochastic" for batch
#     batch_size  :   (optional) only used when stochastic type of batch
#     seed        :   (optional) if not specified, random batch, else specify for the same batch of data
#     """
#     if type == "full":
#         s = np.arange(self._number_of_datapoints)
#         batch_size = self._number_of_datapoints
#         return s

#     if type == "stochastic":
#         if seed is None:
#             rng = np.random.default_rng(np.random.randint(np.iinfo(np.int16).max, size=1)[0])
#         elif isinstance(seed, int):
#             rng = np.random.default_rng(seed)
#         else:
#             raise Exception("seed must be an integer if specified")

#         if batch_size < 0:
#             raise Exception(f"{type} gradient requires a batch_size > 0")

#         if batch_size > self._number_of_datapoints:
#             raise Exception("Batch size specified is larger than size of dataset")

#         s = rng.choice(self._number_of_datapoints, size=(batch_size), replace=False)
#         return s

#     raise Exception(f"{type} is not a defined type of gradient")

# def objective(self, x: np.array, type: str = "full", batch_size: int = 0, seed: int | None = None) -> float:
#     """
#     Calculates loss for full or a stochastic batch of data
#     Inputs:
#     x           :   logistic regression parameters
#     type        :   "full" for full data and "stochastic" for batch
#     batch_size  :   (optional) only used when stochastic type of batch
#     seed        :   (optional) if not specified, random batch, else specify for the same batch of data
#     """
#     """Evaluates MLE loss"""

#     s = self._determine_batch(type, batch_size, seed)

#     # signmoid calculate
#     exp_neg = np.exp(-np.dot(x.T, self._features[:, s])).T
#     y_hat = 1 / (1 + exp_neg)

#     # cross entropy loss
#     loss = -(self._targets[s, :] * np.log(y_hat) + (1 - self._targets[s, :]) * np.log(1 - y_hat))

#     # replace nan with 0 to for 0*log(0) values
#     loss[np.isnan(loss)] = 0
#     return np.sum(loss)

# def gradient(self, x: np.array, type: str = "full", batch_size: int = 0, seed: int | None = None) -> np.ndarray:
#     """MLE Loss gradient"""
#     """
#     Calculates gradient of loss for full or a stochastic batch of data
#     Inputs:
#     x           :   logistic regression parameters
#     type        :   "full" for full data and "stochastic" for batch
#     batch_size  :   (optional) only used when stochastic type of batch
#     seed        :   (optional) if not specified, random batch, else specify for the same batch of data
#     """

#     s = self._determine_batch(type, batch_size, seed)
#     x = x.reshape(
#         (self.number_features, self.number_classes - 1)
#     )  # we consider only K-1 classes to reduce dimension in logistic regreesion

#     ET = np.exp(np.dot(x.T, self.data[:, s]))
#     temp = np.sum(ET, axis=0) + 1
#     P = ET / temp

#     g = (
#         np.dot(self.data[:, s], (P - self.labels[:, s]).T) / batch_size + 2 * x / self.number_datapoints
#     )  # regularization with # datapoints

#     return g.reshape((self.number_features * (self.number_classes - 1), 1))

# def hessian(self, x: np.array) -> np.array:
#     raise Exception(f"Can't compute hessian for {self.name}")
