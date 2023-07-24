from Machine_Learning.Logistic_Regression import Cross_Entropy_Binary

# * Cross Entropy logistic regression problems

problem = Cross_Entropy_Binary(name = "australian", location="../../Datasets/australian_scale.txt", sparse_format=False)

print(problem._targets)
