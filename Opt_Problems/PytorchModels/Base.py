import numpy as np
import torch
from Opt_Problems.Base import Problem
from Opt_Problems.Utilities import (
    create_rng,
    datasets_manager,
    pytorch_datasets_manager,
)
from Opt_Problems.Options import (
    StochasticApproximationType,
    PytorchClassificationModelOptions,
)
from Opt_Problems.PytorchModels.BaseModels import FNN, TinyVGG
from Opt_Problems.PytorchModels.PytorchSampler import SubsetSampler
from torch.utils.data import DataLoader


"""
Important Notices:

1. The data has to be organized as one datapoint in each row (datapoints x features) when flattened.
2. Datasets are downloaded from Pytorch inbuilt datasets.
3. Primarily for image classification.

"""


class PytorchModelsImageClassification(Problem):

    def __init__(
        self,
        dataset_name: PytorchClassificationModelOptions,
        pytorch_model: PytorchClassificationModelOptions,
        **kwargs: dict[PytorchClassificationModelOptions, list],
    ) -> None:
        """Read Dataset"""  # datasets have each column as features)
        (
            self.train_features,
            self.train_labels,
            self.test_features,
            self.test_labels,
            self.number_of_classes,
        ) = pytorch_datasets_manager(dataset_name=dataset_name)

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
                input_dim=self.train_features[0].numel(),
                hidden_layers=self.layers,
                output_dim=self.number_of_classes,
                activation=self.activation,
            )
        elif self.pytorch_model is PytorchClassificationModelOptions.TinyVGG:
            if dataset_name is not PytorchClassificationModelOptions.FASHION_MNIST:
                raise ValueError("TinyVGG is only available for Fashion_MNIST")

            self.train_features = self.train_features.reshape(-1, 1, 28, 28)
            self.test_features = self.test_features.reshape(-1, 1, 28, 28)

            self.model = TinyVGG(
                input_shape=1, hidden_units=10, output_shape=self.number_of_classes
            )

        elif self.pytorch_model in [
            PytorchClassificationModelOptions.ResNet50,
            PytorchClassificationModelOptions.ResNet18,
            PytorchClassificationModelOptions.ResNet34,
        ]:
            if dataset_name not in [
                PytorchClassificationModelOptions.CIFAR10,
                PytorchClassificationModelOptions.CIFAR100,
            ]:
                raise ValueError("ResNet is only available for CIFAR10 and CIFAR100")

            self.train_features = self.train_features.reshape(-1, 3, 32, 32)
            self.test_features = self.test_features.reshape(-1, 3, 32, 32)
            if self.pytorch_model is PytorchClassificationModelOptions.ResNet18:
                from torchvision.models import resnet18
                self.model = resnet18(num_classes=self.number_of_classes)
            if self.pytorch_model is PytorchClassificationModelOptions.ResNet34:
                from torchvision.models import resnet34
                self.model = resnet34(num_classes=self.number_of_classes)
            if self.pytorch_model is PytorchClassificationModelOptions.ResNet50:
                from torchvision.models import resnet50
                self.model = resnet50(num_classes=self.number_of_classes)
        else:
            raise ValueError("Unknown pytorch model")

        """Send model and data to device"""
        self.model.to(self.device)
        self.train_features = self.train_features.to(self.device)
        self.train_labels = self.train_labels.to(self.device)
        self.test_features = self.test_features.to(self.device)
        self.test_labels = self.test_labels.to(self.device)

        """Call super class"""
        super().__init__(
            name=f"{dataset_name.value}_{self.pytorch_model.value}",
            d=sum([param.numel() for param in self.model.parameters()]),
            number_of_datapoints=self.train_features.size(0),
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
        for name, param in self.model.named_parameters():
            state_dict[name] = x[index : index + param.numel()].view(param.shape)
            index += param.numel()
        self.model.load_state_dict(state_dict)

    def _parameter_grad_to_numpy_vector(self) -> np.ndarray:
        x = np.zeros((self.d,))
        index = 0
        for _, param in self.model.named_parameters():
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

        if self.pytorch_model is PytorchClassificationModelOptions.ResNet18 or self.pytorch_model is PytorchClassificationModelOptions.ResNet34 or self.pytorch_model is PytorchClassificationModelOptions.ResNet50:
            if len(data_points) > 1000:
                return self._calculate_loss_batch_wise(self.train_features[data_points], self.train_labels[data_points])

        # calculate loss in eval mode
        self.model.eval()
        with torch.inference_mode():
            loss = self.loss_fuction(
                self.model(self.train_features[data_points]),
                self.train_labels[data_points],
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
            self.model(self.train_features[data_points]), self.train_labels[data_points]
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

        # convert vector to model parameters
        self._numpy_vector_assign_to_model(x)

        return self._accuracy(self.train_features, self.train_labels)

    def accuracy_test(
        self,
        x: np.ndarray,
    ) -> float:

        # convert vector to model parameters
        self._numpy_vector_assign_to_model(x)

        return self._accuracy(self.test_features, self.test_labels)

    def objective_test(
        self,
        x: np.ndarray,
    ) -> float:
        pass

        # convert vector to model parameters
        self._numpy_vector_assign_to_model(x)

        # calculate loss in eval mode
        self.model.eval()
        with torch.inference_mode():
            loss = self.loss_fuction(self.model(self.test_features), self.test_labels)

        return float(loss)

    def _accuracy(self, features: torch.Tensor, targets: torch.Tensor) -> float:

        self.model.eval()
        acc = 0
        with torch.inference_mode():
            index = 0
            while index < features.shape[0]:
                output = torch.softmax(self.model(features[index : index + 100]), dim=1)
                acc += torch.sum(torch.argmax(output, dim=1) == targets[index : index + 100])
                index += 100
        
        return float(acc / targets.shape[0])


    def _calculate_loss_batch_wise(self, features: torch.Tensor, targets: torch.Tensor) -> float:

        loss = 0
        self.model.eval()
        with torch.inference_mode():
            index = 0
            while index < features.shape[0]:
                loss += 100*self.loss_fuction(self.model(features[index : index + 100]), targets[index : index + 100])
                index += 100
        
        return float(loss / features.shape[0])