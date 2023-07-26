from Machine_Learning.Logistic_Regression import Cross_Entropy_Binary

# * Cross Entropy logistic regression problems

problem = Cross_Entropy_Binary(name="gisette", location="../Datasets/gisette_scale.bz2", sparse_format=False)

# print(problem._targets)
# print(problem._features)
# print(problem._number_of_features)
# print(problem._number_of_datapoints)
# print(problem.name)
print(problem.gradient(x=problem.initial_point(), type="stochastic", batch_size=10, seed=80))
print(problem.gradient(x=problem.initial_point(), type="stochastic", batch_size=10, seed=80))
