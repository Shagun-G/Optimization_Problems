import numpy as np


def CrossEntropyLoss(y_hat: np.ndarray, targets: np.ndarray) -> float:

    loss = -targets * np.log(y_hat) - (1 - targets) * np.log(1 - y_hat)
    loss[np.isnan(loss)] = 0
    return np.mean(loss)


def CrossEntropyLossDerivative(
    y_hat: np.ndarray, features: np.ndarray, targets: np.ndarray
) -> np.ndarray:

    g = np.dot(features, (y_hat - targets)) / targets.shape[0]
    return g


def HuberLoss(y_hat: np.ndarray, targets: np.ndarray) -> float:

    error = targets - y_hat
    loss = np.mean(np.square(error) / (1 + np.square(error)))
    return loss


def HuberLossDerivative(
    y_hat: np.ndarray, features: np.ndarray, targets: np.ndarray
) -> np.ndarray:

    error = targets - y_hat
    g = (
        2
        * (
            np.dot(
                features,
                ((error / np.square((1 + np.square(error)))) * y_hat * (y_hat - 1)),
            )
        )
        / targets.shape[0]
    )
    return g
