import numpy as np
import pycutest

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

from Opt_Problems.CUTEST import CUTEST_problem

problem = CUTEST_problem(name="BT9")
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