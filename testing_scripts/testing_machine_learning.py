from Opt_Problems.Machine_Learning import Logistic_Regression
from sklearn import preprocessing
# * Cross Entropy logistic regression problems

problem = Logistic_Regression.Cross_Entropy_Binary(
    name="australian", location="../Datasets/australian_scale.txt"
)


# print(problem._targets[problem._targets == 0])
# print(problem._targets[problem._targets == -1])
# print(problem._targets[problem._targets == 1])
print(problem._features)
# print(problem._number_of_features)
# print(problem._number_of_datapoints)
# print(problem.name)
# print(problem.objective(x=problem.initial_point(), type="stochastic", batch_size=5, seed=80))
# print(problem.gradient(x=problem.initial_point(), type="stochastic", batch_size=5, seed=80))

