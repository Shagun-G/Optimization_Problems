from Opt_Problems.PytorchModels.Base import PytorchModelsClassification
from Opt_Problems.Options import (
    StochasticApproximationType,
    PytorchClassificationModelOptions,
)

problem = PytorchModelsClassification(
    dataset_name=PytorchClassificationModelOptions.CIFAR100,
    pytorch_model=PytorchClassificationModelOptions.FNN,
    Hidden_Layers=[512],
    Activation=PytorchClassificationModelOptions.ReLU,
)

# # """Calling all functions"""
print("Name: " + problem.name)
print("Dimension : ", problem.d)
# print("At x = ", problem.initial_point())
print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.FullBatch))
# print("Gradient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.FullBatch))
# print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.MiniBatch, batch_size=4, seed=100))
# print("adient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.MiniBatch, batch_size=4, seed=100))
# print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.SpecifiedIndices, data_indices=[0, 1, 2, 3, 4, 5]))
# print("Gradient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.SpecifiedIndices, data_indices=[0, 1, 2, 3, 4, 5]))

print("Function: ", problem.objective_test(x=problem.initial_point()))
print("Accuracy train: ", problem.accuracy_train(x=problem.initial_point()))
print("Accuracy test: ", problem.accuracy_test(x=problem.initial_point()))

### Training Loop to Test gradients and Function Value ###

x = problem.initial_point()
for i in range(1, 10000):
    x = x - 0.01 * problem.gradient(
        x=x, type=StochasticApproximationType.MiniBatch, batch_size=32
    )

    if i % 100 == 0:
        print("------------------{i}------------------".format(i=i))
        print(
            f"Train loss: ",
            problem.objective(x=x, type=StochasticApproximationType.FullBatch),
        )
        print(f"Test loss: ", problem.objective_test(x=x))
        print("Accuracy train: ", problem.accuracy_train(x=x))
        print("Accuracy test: ", problem.accuracy_test(x=x))
