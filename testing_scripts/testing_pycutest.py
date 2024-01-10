import numpy as np
import pycutest

'''Testing CUTEST Constrained'''
print("----------------------------------------------------")
p = pycutest.import_problem('BT9')

print(p.name)
print(p.m)
print(p.n)
print(p.n_free)
print(p.cl)
print(p.cu)
print(type(p.is_eq_cons))

x = p.x0
f = p.obj(x)  
print(f)
g, J = p.lagjac(x)
print(g)
print(p.is_eq_cons)
print(J)
print(J[p.is_eq_cons])
g = p.ihess(x)
print(g)
g = p.objcons(x)[1][np.array([True, False])]
print(g)
print(p.ihess(x, cons_index=0))
print(p.ihess(x, cons_index=1))

from Opt_Problems.CUTEST.CUTEST_deterministic import CUTEST_constrained

print("----------------------------------------------------")
problem = CUTEST_constrained(name="BT9")
print(problem.name)
print(problem.d)
print(problem.number_of_eq_constraints)
print(problem.number_of_ineq_constraints)
print(problem.initial_point())
print(problem.objective(problem.initial_point()))
print(problem.gradient(problem.initial_point()))
print(problem.hessian(problem.initial_point()))
print(problem.constraints_eq(problem.initial_point()))
print(problem.constraints_ineq(problem.initial_point()))
print(problem.constraints_eq_jacobian(problem.initial_point()))
print(problem.constraints_ineq_jacobian(problem.initial_point()))
print(problem.constraints_eq_hessian(problem.initial_point()))
print(problem.constraints_ineq_hessian(problem.initial_point()))

'''Testing CUTEST Unconstrained'''
print("----------------------------------------------------")
p = pycutest.import_problem('ROSENBR')

print(p.name)
print(p.n)

x = p.x0
f, g = p.obj(x, gradient=True)  
print(f)
print(g)
g = p.ihess(x)
print(g)

from Opt_Problems.CUTEST.CUTEST_deterministic import CUTEST_unconstrained

print("----------------------------------------------------")
problem = CUTEST_unconstrained(name="ROSENBR")
print(problem.name)
print(problem.d)
print(problem.initial_point())
print(problem.objective(problem.initial_point()))
print(problem.gradient(problem.initial_point()))
print(problem.hessian(problem.initial_point()))