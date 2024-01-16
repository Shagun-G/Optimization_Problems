from Opt_Problems.Machine_Learning.Constrained_Logistic_Regression import Cross_Entropy_Binary_Linear_norm_constraint, Cross_Entropy_Binary_Linear_constraint, Cross_Entropy_Binary_norm_constraint

problem = Cross_Entropy_Binary_Linear_norm_constraint(name="australian", location="../Datasets/australian_scale.txt", m=10, constraint_seed=1)

# print(problem.number_of_features)
# print(problem.number_of_datapoints)
# print(problem.name)
# print(problem.objective(x=problem.initial_point(), type="stochastic", batch_size=5, seed=80))
# print(problem.gradient(x=problem.initial_point(), type="stochastic", batch_size=5, seed=80))
print(problem.constraints_eq(x=problem.initial_point()))
# print(problem.constraints_eq_jacobian(x=problem.initial_point()))
# print(problem.constraints_eq_hessian(x=problem.initial_point()))

problem = Cross_Entropy_Binary_norm_constraint(name="australian", location="../Datasets/australian_scale.txt")

# print(problem.number_of_features)
# print(problem.number_of_datapoints)
# print(problem.name)
# print(problem.objective(x=problem.initial_point(), type="stochastic", batch_size=5, seed=80))
# print(problem.gradient(x=problem.initial_point(), type="stochastic", batch_size=5, seed=80))
print(problem.constraints_eq(x=problem.initial_point()))
# print(problem.constraints_eq_jacobian(x=problem.initial_point()))
# print(problem.constraints_eq_hessian(x=problem.initial_point()))

problem = Cross_Entropy_Binary_Linear_constraint(name="australian", location="../Datasets/australian_scale.txt", m=10, constraint_seed=1)

# print(problem.number_of_features)
# print(problem.number_of_datapoints)
# print(problem.name)
# print(problem.objective(x=problem.initial_point(), type="stochastic", batch_size=5, seed=80))
# print(problem.gradient(x=problem.initial_point(), type="stochastic", batch_size=5, seed=80))
print(problem.constraints_eq(x=problem.initial_point()))
# print(problem.constraints_eq_jacobian(x=problem.initial_point()))
# print(problem.constraints_eq_hessian(x=problem.initial_point()))