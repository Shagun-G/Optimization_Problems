from Opt_Problems.Stochastic.Nueral_networks import FNN
from Opt_Problems.Options import StochasticApproximationType, Datasets, MachineLearningLossFunctions

train_location="../Datasets/cifar10.bz2"
test_location="../Datasets/cifar10.t.bz2"
problem = FNN(dataset_name=Datasets.CIFAR10, train_location=train_location, test_location=test_location, loss_function=MachineLearningLossFunctions.MSE, n_Hidden=[512])

"""Calling all functions"""
print("Name: " + problem.name)
print("Dimension : ", problem.d)
# print("At x = ", problem.initial_point())
print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.FullBatch))
# print("Gradient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.FullBatch))
# print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.MiniBatch, batch_size=4, seed=100))
# print("Gradient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.MiniBatch, batch_size=4, seed=100))
# print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.SpecifiedIndices, data_indices=[0, 1, 2, 3, 4, 5]))
# print("Gradient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.SpecifiedIndices, data_indices=[0, 1, 2, 3, 4, 5]))

print("Function: ", problem.objective_test(x=problem.initial_point()))
print("Accuracy train: ", problem.accuracy_train(x=problem.initial_point()))
print("Accuracy test: ", problem.accuracy_test(x=problem.initial_point()))