import numpy as np
import torch
from Opt_Problems.Base import Problem
from Opt_Problems.Utilities import create_rng, datasets_manager
from Opt_Problems.Options import (
    StochasticApproximationType,
    MachineLearningLossFunctions,
    Datasets,
    PytorchClassificationModelOptions,
)
from Opt_Problems.PytorchModels.FNN import FNN


"""
Important Notices:

1. The data has to be organized as one datapoint in each row (datapoints x features).

"""


class PytorchModelsClassification(Problem):

    def __init__(
        self,
        dataset_name: Datasets,
        train_location: str,
        pytorch_model: PytorchClassificationModelOptions,
        test_location: str = None,
        **kwargs: dict[PytorchClassificationModelOptions, list],
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

        """Device"""
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        """Model"""
        self.pytorch_model = pytorch_model
        if self.pytorch_model is PytorchClassificationModelOptions.FNN:
            if "Hidden_Layers" not in kwargs:
                raise ValueError("Missing number of hidden layers for FNN")
            if "Activation" not in kwargs:
                raise ValueError("Missing number of Activations layers for FNN")

            self.layers = kwargs["Hidden_Layers"]
            self.activation = kwargs["Activation"]
            if self.activation is PytorchClassificationModelOptions.ReLU:
                self.activation = torch.nn.ReLU()
            elif self.activation is PytorchClassificationModelOptions.Sigmoid:
                self.activation = torch.nn.Sigmoid()
            elif self.activation is PytorchClassificationModelOptions.Tanh:
                self.activation = torch.nn.Tanh()
            else:
                raise ValueError("Unknown activation function")

            self.model = FNN(
                input_dim=self._features_train.shape[0],
                hidden_layers=self.layers,
                output_dim=self.number_of_classes,
                activation=self.activation,
            )
        else:
            raise ValueError("Unknown pytorch model")

        """Send model and data to device"""
        self.model.to(self.device)
        self._features_train = (
            torch.from_numpy(self._features_train.T).to(torch.float32).to(self.device)
        )
        self._targets_train = (
            torch.from_numpy(self._targets_train).to(torch.int64).to(self.device)
        )
        if self.test_location is not None:
            self._features_test = (
                torch.from_numpy(self._features_test.T)
                .to(torch.float32)
                .to(self.device)
            )
            self._targets_test = (
                torch.from_numpy(self._targets_test).to(torch.int64).to(self.device)
            )

        """Call super class"""
        super().__init__(
            name=f"{dataset_name.value}_{self.pytorch_model.value}",
            d=sum([param.numel() for param in self.model.parameters()]),
            number_of_datapoints=n_train,
        )

        """Set Loss function"""
        self.loss_fuction = torch.nn.CrossEntropyLoss()

    def initial_point(self, seed=100) -> np.ndarray:
        rng = create_rng(seed)
        return 0.1 * rng.normal(0, 1, size=(self.d, 1))

    def _numpy_vector_assign_to_model(self, x: np.ndarray):

        x = torch.from_numpy(x).to(torch.float32).to(self.device)

        index = 0
        state_dict = self.model.state_dict()
        for key in state_dict:
            num_param = state_dict[key].numel()
            state_dict[key] = x[index : index + num_param].view(state_dict[key].shape)
            index += num_param

        self.model.load_state_dict(state_dict)

    def _parameter_grad_to_numpy_vector(self) -> np.ndarray:
        x = np.zeros((self.d, ))
        index = 0
        for param in self.model.parameters():
            x[index : index + param.numel()] = param.grad.view(-1).to("cpu").numpy()
            index += param.numel()
        return x.reshape(-1, 1)

    def objective(
        self,
        x: np.ndarray,
        type: StochasticApproximationType,
        batch_size: int = None,
        seed: int = None,
        data_indices: list = None,
    ) -> float:

        data_points = super().generate_stochastic_batch(
            type=type, batch_size=batch_size, seed=seed, data_indices=data_indices
        )

        # convert vector to model parameters
        self._numpy_vector_assign_to_model(x)

        # calculate loss in eval mode
        self.model.eval()
        with torch.inference_mode():
            loss = self.loss_fuction(
                self.model(self._features_train[data_points]),
                self._targets_train[data_points],
            )

        return float(loss)

    def gradient(
        self,
        x: np.ndarray,
        type: StochasticApproximationType,
        batch_size: int = None,
        seed: int = None,
        data_indices: list = None,
    ) -> float:

        data_points = super().generate_stochastic_batch(
            type=type, batch_size=batch_size, seed=seed, data_indices=data_indices
        )

        # convert vector to model parameters
        self._numpy_vector_assign_to_model(x)

        # calculate loss in train mode
        self.model.train()
        loss = self.loss_fuction(
            self.model(self._features_train[data_points]),
            self._targets_train[data_points],
        )

        # calculate gradients
        loss.backward()

        # convert gradients to vector
        grad = self._parameter_grad_to_numpy_vector()

        self.model.zero_grad()

        return grad

    def accuracy_train(
        self,
        x: np.ndarray,
    ) -> float:

        if self.test_location is None:
            raise Exception("No test Data available")

        # convert vector to model parameters
        self._numpy_vector_assign_to_model(x)

        return self._accuracy(self._features_train, self._targets_train)

    def accuracy_test(
        self,
        x: np.ndarray,
        W: list[np.ndarray] = None,
        b: list[np.ndarray] = None,
        in_matrix_form: bool = False,
    ) -> float:

        if self.test_location is None:
            raise Exception("No test Data available")

        # convert vector to model parameters
        self._numpy_vector_assign_to_model(x)

        return self._accuracy(self._features_test, self._targets_test)

    def objective_test(
        self,
        x: np.ndarray,
        W: list[np.ndarray] = None,
        b: list[np.ndarray] = None,
        in_matrix_form: bool = False,
    ) -> float:
        pass

        if self.test_location is None:
            raise Exception("No test Data available")

        # convert vector to model parameters
        self._numpy_vector_assign_to_model(x)

        # calculate loss in eval mode
        self.model.eval()
        with torch.inference_mode():
            loss = self.loss_fuction(
                self.model(self._features_test),
                self._targets_test,
            )

        return float(loss)

    def _accuracy(self, features: torch.Tensor, targets: torch.Tensor) -> float:

        output = torch.softmax(self.model(features), dim=1)
        output = torch.argmax(output, dim=1)
        return float(torch.sum(output == targets) / targets.shape[0])


