from Opt_Problems.Stochastic.LogisticRegression import MultiClassLogisticRegression
from Opt_Problems.Options import (
    StochasticApproximationType,
    Datasets,
    MachineLearningLossFunctions,
)

# train_location = "../Datasets/mnist.bz2"
# test_location = "../Datasets/mnist.t.bz2"
# train_location = "../Datasets/cifar10.bz2"
# test_location = "../Datasets/cifar10.t.bz2"
train_location = "../Datasets/covtype.scale01.bz2"
problem = MultiClassLogisticRegression(
    dataset_name=Datasets.COVTYPE,
    train_location=train_location,
    # test_location=test_location,
    # number_of_linear_constraints=2,
    # linear_constraint_seed=100,
    # norm_equality_constant=1,
    norm_inequality_constant=1
)

"""Calling all functions"""
print("Name: " + problem.name)
print("Dimension : ", problem.d)
print("Regularized: ", problem.regularize)
# print("At x = ", problem.initial_point())
x = problem.initial_point()
print("Function: ", problem.objective(x=x, type=StochasticApproximationType.FullBatch))
# print(
#     "Gradient : ",
#     problem.gradient(
#         x=x, type=StochasticApproximationType.FullBatch
#     ),
# )
# print("Function: ", problem.objective(x=x, type=StochasticApproximationType.MiniBatch, batch_size=4, seed=100))
# print(
#     "Gradient : ",
#     problem.gradient(
#         x=problem.initial_point(),
#         type=StochasticApproximationType.MiniBatch,
#         batch_size=49990,
#         seed=100,
#     ),
# )
# print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.SpecifiedIndices, data_indices=[0, 1, 2, 3, 4, 5]))
# print("Gradient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.SpecifiedIndices, data_indices=[0, 1, 2, 3, 4, 5]))
# print("c_eq: ", problem.constraints_eq(x=x))
# print("J_eq: ", problem.constraints_eq_jacobian(x=problem.initial_point()))
# print("c_ineq : ", problem.constraints_ineq(x=problem.initial_point()))
# print("J_ineq : ", problem.constraints_ineq_jacobian(x=problem.initial_point()))
print("Has bound constraints : ", problem.has_bound_constraints)

print("Has linear:", problem.number_of_linear_constraints )
print("Has norm eq:", problem.norm_eq_constraint )
print("Has norm ineq:", problem.norm_ineq_constraint )

# for _ in range(100):
    # x = x - 0.01*problem.gradient(x=x, type=StochasticApproximationType.FullBatch)
    # print("Loss: ", problem.objective(x=x, type=StochasticApproximationType.FullBatch))