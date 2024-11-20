from Opt_Problems.Deterministic.CUTEST import CUTEST

name = "ELEC"
# name = "DISC2"
problem = CUTEST(name=name)
# """Calling all functions"""
print("Name : " + problem.name)
print("Dimension : ", problem.d)
# print("At x = ", problem.initial_point())
print("Function : ", problem.objective(x=problem.initial_point()))
print("Gradient : ", problem.gradient(x=problem.initial_point()))
# print("c_eq : ", problem.constraints_eq(x=problem.initial_point()))
# print("J_eq : ", problem.constraints_eq_jacobian(x=problem.initial_point()))
print("c_ineq : ", problem.constraints_ineq(x=problem.initial_point()))
print("J_ineq : ", problem.constraints_ineq_jacobian(x=problem.initial_point()))
# print("Has bound constraints : ", problem.has_bound_constraints)
# print("Bounds : ", problem.variable_lower_bounds(), problem.varable_upper_bounds())

# from Opt_Problems.Stochastic.CUTEST import CUTESTStochastic
# from Opt_Problems.Options import StochasticApproximationType, CUTESTNoiseOptions

# problem = CUTESTStochastic(name=name, noise_option=CUTESTNoiseOptions.NormalNoisyGradient, noise_level=1)
"""Calling all functions"""
# print("Name: " + problem.name)
# print("Dimension : ", problem.d)
# print("At x = ", problem.initial_point())
# print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.FullBatch))
# print("Gradient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.FullBatch))
# print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.MiniBatch, batch_size=10, seed=100))
# print("Gradient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.MiniBatch, batch_size=10, seed=100))
# print("Function: ", problem.objective(x=problem.initial_point(), type=StochasticApproximationType.SpecifiedIndices, data_indices=range(10)))
# print("Gradient : ", problem.gradient(x=problem.initial_point(), type=StochasticApproximationType.SpecifiedIndices, data_indices=[0]))
# print("c_eq: ", problem.constraints_eq(x=problem.initial_point()))
# print("J_eq: ", problem.constraints_eq_jacobian(x=problem.initial_point()))
# print("c_ineq : ", problem.constraints_ineq(x=problem.initial_point()))
# print("J_ineq : ", problem.constraints_ineq_jacobian(x=problem.initial_point()))
# print("Has bound constraints : ", problem.has_bound_constraints)
# print("Bounds : ", problem.variable_lower_bounds(), problem.varable_upper_bounds())