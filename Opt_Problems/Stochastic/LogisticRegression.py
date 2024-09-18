import numpy as np
from Opt_Problems.Base import Problem
from Opt_Problems.Utilities import generate_linear_constraints, create_rng, datasets_manager
from Opt_Problems.ML_loss_functions import CrossEntropyLoss, CrossEntropyLossDerivative, HuberLoss, HuberLossDerivative
from Opt_Problems.Options import StochasticApproximationType, MachineLearningLossFunctions, Datasets
from scipy.special import expit

class BinaryLogisticRegression(Problem):

    def __init__(self, dataset_name: Datasets, train_location: str, loss_function : MachineLearningLossFunctions, test_location: str = None, number_of_linear_constraints : int = 0, linear_constraint_seed = None, norm_equality_constant : float = None, norm_inequality_constant : float = None) -> None:

        '''Read Dataset'''
        X, y = datasets_manager(dataset_name=dataset_name, location=train_location)
        n_train, d = np.shape(X)
        y = y.reshape((n_train, 1))  # reshaping target matrix
        X = X.toarray()
        ptp = X.ptp(0)
        mins_data = X.min(0)
        X = (X - mins_data) / ptp  # normalizing
        X = np.nan_to_num(X, nan=0)  # replacing divide by 0 with 0
        X = np.vstack(
            (X.T, np.ones((1, n_train)))
        )  # adding bias term to features
        d += 1
        self._features = X
        self._targets = y

        self.test_location = test_location
        if self.test_location is not None:
            X_test, y_test = datasets_manager(dataset_name=dataset_name, location=self.test_location)
            n_test, _ = np.shape(X_test)
            y_test = y_test.reshape(
                (n_test, 1)
            )  # reshaping target matrix
            X_test = X_test.toarray()

            X_test = (X_test - mins_data) / ptp  # normalizing
            X_test = np.nan_to_num(X_test, nan=0)  # replacing divide by 0 with 0
            X_test = np.vstack(
                (X_test.T, np.ones((1, n_test)))
            )  # adding bias term to features
            self._features_test = X_test
            self._targets_test = y_test        

        '''Generate Constraints'''
        m_eq = 0
        m_ineq = 0
        self.number_of_linear_constraints = number_of_linear_constraints
        self.norm_eq_constraint = norm_equality_constant is not None
        self.norm_ineq_constraint = norm_inequality_constant is not None
        self.regularize = not (self.norm_eq_constraint or self.norm_ineq_constraint  )

        if self.number_of_linear_constraints > 0:
            m_eq += self.number_of_linear_constraints
            rng = create_rng(linear_constraint_seed)
            self._A_eq, self._b_eq = generate_linear_constraints(m = self.number_of_linear_constraints, d = d, rng=rng)
        if self.norm_eq_constraint:
            m_eq += 1
            self._b_eq_norm = norm_equality_constant
        if self.norm_ineq_constraint:
            m_ineq += 1
            self._b_ineq_norm = norm_inequality_constant
            
        '''Call super class'''
        super().__init__(name=f"{dataset_name.value}_{loss_function.value}", d=d, eq_const_number=m_eq, ineq_const_number=m_ineq, number_of_datapoints=n_train)

        if self.norm_ineq_constraint and self.norm_eq_constraint:
            raise ValueError('Cannot have both norm_equality and norm_inequality constraints')

        '''Set Loss function'''
        self.loss_fuction = loss_function

    def initial_point(self) -> np.ndarray:
        return np.ones((self.d, 1))

    def objective(self, x: np.ndarray, type: StochasticApproximationType, batch_size: int = None, seed: int = None, data_indices: list = None) -> float:

        data_points = super().generate_stochastic_batch(type = type, batch_size= batch_size, seed=seed, data_indices=data_indices)

        y_hat = self._predicted_probability(x, self._features[:, data_points])

        if self.loss_fuction is MachineLearningLossFunctions.CrossEntropy:
            loss = CrossEntropyLoss(y_hat, self._targets[data_points, :])

        if self.loss_fuction is MachineLearningLossFunctions.HuberLoss:
            loss = HuberLoss(y_hat, self._targets[data_points, :])

        if self.regularize:
            loss += np.linalg.norm(x) ** 2 / self.number_of_datapoints
        
        return loss
    
    def gradient(self, x: np.ndarray, type: StochasticApproximationType, batch_size: int = None, seed: int = None, data_indices: list = None) -> float:

        data_points = super().generate_stochastic_batch(type = type, batch_size= batch_size, seed=seed, data_indices=data_indices)

        y_hat = self._predicted_probability(x, self._features[:, data_points])
        if self.loss_fuction is MachineLearningLossFunctions.CrossEntropy:
            g = CrossEntropyLossDerivative(y_hat, self._features[:, data_points], self._targets[data_points, :])

        if self.loss_fuction is MachineLearningLossFunctions.HuberLoss:
            g = HuberLossDerivative(y_hat, self._features[:, data_points], self._targets[data_points, :])

        if self.regularize:
            g += 2 * x / self.number_of_datapoints
        return g

    def accuracy_train(
        self,
        x: np.array,
    ) -> float:

        return self._accuracy(x, self._features, self._targets)

    def accuracy_test(
        self,
        x: np.array,
    ) -> float:

        if self.test_location is None:
            raise Exception("No test Data available")

        return self._accuracy(x, self._features_test, self._targets_test)

    def objective_test(self, x: np.ndarray) -> float:

        if self.test_location is None:
            raise Exception("No test Data available")

        y_hat = self._predicted_probability(x, self._features_test)

        if self.loss_fuction is MachineLearningLossFunctions.CrossEntropy:
            loss = CrossEntropyLoss(y_hat, self._targets_test)

        if self.loss_fuction is MachineLearningLossFunctions.HuberLoss:
            loss = HuberLoss(y_hat, self._targets_test)
        
        return loss

    '''Functions for mid steps ML'''
    def _predicted_probability(self, x: np.ndarray, features: np.ndarray) -> np.ndarray:

        return expit(np.dot(x.T, features)).T

    def _accuracy(
        self,
        x: np.array,
        features: np.ndarray,
        targets: np.ndarray
    ) -> float:

        y_hat = self._predicted_probability(x, features)
        accuracy = 1 * (y_hat > 0.5) - targets

        return 1 - np.mean(np.abs(accuracy))

    '''Constraint Functions'''
    
    def constraints_eq(self, x: np.ndarray) -> np.ndarray:
        c_eq = super().constraints_eq(x)

        if self.number_of_linear_constraints > 0:
            c_eq = np.vstack([c_eq, np.dot(self._A_eq, x) - self._b_eq])

        if self.norm_eq_constraint:
            c_eq = np.vstack([c_eq, np.linalg.norm(x)**2 - self._b_eq_norm])

        return c_eq

    def constraints_ineq(self, x: np.ndarray) -> np.ndarray:
        c_ineq = super().constraints_ineq(x)

        if self.norm_ineq_constraint:
            c_ineq = np.vstack([c_ineq, np.linalg.norm(x)**2 - self._b_ineq_norm])

        return c_ineq

    def constraints_eq_jacobian(self, x: np.ndarray) -> np.ndarray:
        J_eq = super().constraints_eq_jacobian(x)

        if self.number_of_linear_constraints > 0:
            J_eq = np.vstack([J_eq, self._A_eq])

        if self.norm_eq_constraint:
            J_eq = np.vstack([J_eq, 2*x.T])

        return J_eq
    
    def constraints_ineq_jacobian(self, x: np.ndarray) -> np.ndarray:
        J_ineq = super().constraints_ineq_jacobian(x)

        if self.norm_ineq_constraint:
            J_ineq = np.vstack([J_ineq, 2*x.T])

        return J_ineq