from Opt_Problems.Stochastic.LogisticRegression import BinaryLogisticRegression
from Opt_Problems.Options import StochasticApproximationType, Datasets, MachineLearningLossFunctions

train_location = "../Datasets/w8a.txt"
test_location = "../Datasets/w8a.t"
problem = BinaryLogisticRegression(dataset_name=Datasets.W8a, train_location=train_location, test_location=test_location, loss_function=MachineLearningLossFunctions.HuberLoss, number_of_linear_constraints=2, linear_constraint_seed=100)

"""Calling all functions"""
print("Name: " + problem.name)
print("Dimension : ", problem.d)
print("Regularized: ", problem.regularize)
# print("At x = ", problem.initial_point())
# print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.FullBatch))
# print("Gradient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.FullBatch))
# print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.MiniBatch, batch_size=4, seed=100))
# print("Gradient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.MiniBatch, batch_size=4, seed=100))
print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.SpecifiedIndices, data_indices=[0, 1, 2, 3, 4, 5]))
print("Gradient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.SpecifiedIndices, data_indices=[0, 1, 2, 3, 4, 5]))
# print("c_eq: ", problem.constraints_eq(x=problem.initial_point()))
# print("J_eq: ", problem.constraints_eq_jacobian(x=problem.initial_point()))
# print("c_ineq : ", problem.constraints_ineq(x=problem.initial_point()))
# print("J_ineq : ", problem.constraints_ineq_jacobian(x=problem.initial_point()))
print("Has bound constraints : ", problem.has_bound_constraints)

print("Accuracy train: ", problem.accuracy_train(x=problem.initial_point()))
print("Accuracy test: ", problem.accuracy_test(x=problem.initial_point()))
print("Has linear:", problem.number_of_linear_constraints )
print("Has norm eq:", problem.norm_eq_constraint )
print("Has norm ineq:", problem.norm_ineq_constraint )