import numpy as np
import jax.numpy as jnp
from jax import grad, jit
from Opt_Problems.Base import Problem
from Opt_Problems.Utilities import create_rng, datasets_manager, relu, one_hot
from Opt_Problems.ML_loss_functions import (
    CrossEntropyLoss,
    CrossEntropyLossDerivative,
    HuberLoss,
    HuberLossDerivative,
)
from Opt_Problems.Options import (
    StochasticApproximationType,
    MachineLearningLossFunctions,
    Datasets,
)
from scipy.special import expit


class FNN(Problem):

    def __init__(
        self,
        dataset_name: Datasets,
        train_location: str,
        loss_function: MachineLearningLossFunctions,
        n_Hidden: list[int],
        test_location: str = None,
    ) -> None:
        """Read Dataset"""  # datasets have each column as features)
        X_train, y_train = datasets_manager(
            dataset_name=dataset_name, location=train_location
        )
        n_train = X_train.shape[1]
        self._features_train = X_train
        self._targets_train = y_train
        self.number_of_classes = len(np.unique(y_train))

        self.test_location = test_location
        if self.test_location is not None:
            X_test, y_test = datasets_manager(
                dataset_name=dataset_name, location=self.test_location
            )
            self.n_test = X_test.shape[1]
            if dataset_name is Datasets.MNIST:
                X_test = np.vstack((X_test, np.ones((2, X_test.shape[1]))))
            self._features_test = X_test
            self._targets_test = y_test

        self.layers = [X_train.shape[0]] + n_Hidden + [self.number_of_classes]

        """Call super class"""
        super().__init__(
            name=f"{dataset_name.value}_{loss_function.value}_FNN",
            d=sum(self.layers[1:])
            + sum(
                [
                    self.layers[index] * self.layers[index + 1]
                    for index in range(len(self.layers) - 1)
                ]
            ),
            number_of_datapoints=n_train,
        )

        """Set Loss function"""
        self.loss_fuction = loss_function

    def _convert_x_to_matrix(
        self, x: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        W = []
        b = []
        index = 0
        for index_layer in range(len(self.layers) - 1):
            W.append(
                jnp.array(
                    x[
                        index : index
                        + self.layers[index_layer] * self.layers[index_layer + 1],
                        :,
                    ].reshape(self.layers[index_layer + 1], self.layers[index_layer])
                )
            )
            index += self.layers[index_layer] * self.layers[index_layer + 1]
            b.append(jnp.array(x[index : index + self.layers[index_layer + 1], :]))
            index += self.layers[index_layer + 1]

        return W, b

    def _convert_matrix_to_x(
        self, W: list[np.ndarray], b: list[np.ndarray]
    ) -> np.ndarray:
        x = np.zeros((self.d, 1))
        index = 0
        for index_layer in range(len(self.layers) - 1):
            x[
                index : index + self.layers[index_layer] * self.layers[index_layer + 1],
                :,
            ] = np.array(W[index_layer].ravel()[:, np.newaxis])
            index += self.layers[index_layer] * self.layers[index_layer + 1]
            x[index : index + self.layers[index_layer + 1], :] = np.array(
                b[index_layer]
            )
            index += self.layers[index_layer + 1]

        return x

    def initial_point(self, seed=100) -> np.ndarray:
        rng = create_rng(seed)
        return 0.1 * rng.normal(0, 1, size=(self.d, 1))

    def objective(
        self,
        x: np.ndarray,
        type: StochasticApproximationType,
        batch_size: int = None,
        seed: int = None,
        data_indices: list = None,
        W: list[np.ndarray] = None,
        b: list[np.ndarray] = None,
        in_matrix_form: bool = False,
    ) -> float:

        data_points = super().generate_stochastic_batch(
            type=type, batch_size=batch_size, seed=seed, data_indices=data_indices
        )

        if not in_matrix_form:
            W, b = self._convert_x_to_matrix(x)

        if self.loss_fuction is MachineLearningLossFunctions.MSE:
            loss = mse_loss(
                features=self._features_train[:, data_points],
                targets=self._targets_train[data_points],
                W=W,
                b=b,
            )
        else:
            raise NotImplementedError

        return float(loss)

    def gradient(
        self,
        x: np.ndarray,
        type: StochasticApproximationType,
        batch_size: int = None,
        seed: int = None,
        data_indices: list = None,
        W: list[np.ndarray] = None,
        b: list[np.ndarray] = None,
        in_matrix_form: bool = False,
    ) -> float:

        data_points = super().generate_stochastic_batch(
            type=type, batch_size=batch_size, seed=seed, data_indices=data_indices
        )

        if not in_matrix_form:
            W, b = self._convert_x_to_matrix(x)

        if self.loss_fuction is MachineLearningLossFunctions.MSE:
            grads = grad(mse_loss, argnums=(2, 3))(
                self._features_train[:, data_points],
                self._targets_train[data_points],
                W,
                b,
            )
        else:
            raise NotImplementedError

        if not in_matrix_form:
            return self._convert_matrix_to_x(grads[0], grads[1])

        return grads[0], grads[1]

    def accuracy_train(
        self,
        x: np.ndarray,
        W: list[np.ndarray] = None,
        b: list[np.ndarray] = None,
        in_matrix_form: bool = False,
    ) -> float:

        if not in_matrix_form:
            W, b = self._convert_x_to_matrix(x)

        return accuracy(self._features_train, self._targets_train, W, b)

    def accuracy_test(
        self,
        x: np.ndarray,
        W: list[np.ndarray] = None,
        b: list[np.ndarray] = None,
        in_matrix_form: bool = False,
    ) -> float:

        if self.test_location is None:
            raise Exception("No test Data available")

        if not in_matrix_form:
            W, b = self._convert_x_to_matrix(x)

        return accuracy(self._features_test, self._targets_test, W, b)

    def objective_test(
        self,
        x: np.ndarray,
        W: list[np.ndarray] = None,
        b: list[np.ndarray] = None,
        in_matrix_form: bool = False,
    ) -> float:

        if not in_matrix_form:
            W, b = self._convert_x_to_matrix(x)

        if self.loss_fuction is MachineLearningLossFunctions.MSE:
            loss = mse_loss(
                features=self._features_test,
                targets=self._targets_test,
                W=W,
                b=b,
            )
        else:
            raise NotImplementedError

        return float(loss)


"""Functions for mid steps ML"""


@jit
def predict_probabilities(y_hat, W, b):
    # make sure if a single datapoint is passed, it is a column vector
    for W_layer, b_layer in zip(W[:-1], b[:-1]):
        y_hat = jnp.dot(W_layer, y_hat) + b_layer
        y_hat = relu(y_hat)
    y_hat = jnp.exp(jnp.dot(W[-1], y_hat) + b[-1])
    return y_hat / jnp.sum(y_hat, axis=0)


@jit
def mse_loss(features, targets, W, b):
    y_hat = predict_probabilities(features, W, b)
    # convert y to one hot encoding
    targets = one_hot(targets, y_hat.shape[0])
    return jnp.mean((targets - y_hat) ** 2)


@jit
def accuracy(features, targets, W, b):
    y_hat = predict_probabilities(features, W, b)
    predictions = jnp.argmax(y_hat, axis=0)
    return jnp.mean(predictions == targets)
