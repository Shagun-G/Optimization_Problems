from Opt_Problems.Stochastic.LogisticRegression import BinaryLogisticRegression
from Opt_Problems.Options import StochasticApproximationType, Datasets, MachineLearningLossFunctions

train_location="../Datasets/ijcnn1.bz2"
test_location="../Datasets/ijcnn1.t.bz2"
problem = BinaryLogisticRegression(dataset_name=Datasets.Ijcnn, train_location=train_location, test_location=test_location, loss_function=MachineLearningLossFunctions.CrossEntropy, number_of_linear_constraints=2, linear_constraint_seed=100)

"""Calling all functions"""
print("Name: " + problem.name)
print("Dimension : ", problem.d)
print("Regularized: ", problem.regularize)
# print("At x = ", problem.initial_point())
# print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.FullBatch))
print("Gradient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.FullBatch))
# print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.MiniBatch, batch_size=4, seed=100))
print("Gradient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.MiniBatch, batch_size=49990, seed=100))
# print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.SpecifiedIndices, data_indices=[0, 1, 2, 3, 4, 5]))
# print("Gradient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.SpecifiedIndices, data_indices=[0, 1, 2, 3, 4, 5]))
# print("c_eq: ", problem.constraints_eq(x=problem.initial_point()))
# print("J_eq: ", problem.constraints_eq_jacobian(x=problem.initial_point()))
# print("c_ineq : ", problem.constraints_ineq(x=problem.initial_point()))
# print("J_ineq : ", problem.constraints_ineq_jacobian(x=problem.initial_point()))
# print("Has bound constraints : ", problem.has_bound_constraints)

# print("Accuracy train: ", problem.accuracy_train(x=problem.initial_point()))
# print("Accuracy test: ", problem.accuracy_test(x=problem.initial_point()))
# print("Has linear:", problem.number_of_linear_constraints )
# print("Has norm eq:", problem.norm_eq_constraint )
# print("Has norm ineq:", problem.norm_ineq_constraint )

# Compare numerical gradient with actual gradient
import numpy as np
def gradient_FD(f, x, epsilon=1e-9):
    """
    Compute numerical gradient of a function using Finite Differences.
    """
    n = x.shape[0]
    gradient = np.zeros((n,1))
    for i in range(n):
        e = np.zeros((n,))
        e[i] = epsilon
        gradient[i] = (f(x + e) - f(x)) / (epsilon)
    return gradient


# print("Gradient : ", gradient_FD(lambda x: problem.objective(x, type=StochasticApproximationType.FullBatch), problem.initial_point()))