from Opt_Problems.Deterministic.CUTEST import CUTEST
from tqdm import tqdm
import pandas as pd

with open("list_of_python_problems") as f:
    list_of_python_problems = [
        line.strip().replace(".py", "") for line in f.readlines()
    ]

names = []
dimension = []
n_constraints = []
n_equality = []
n_inequality = []
has_bounds = []
is_real = []

for name in tqdm(list_of_python_problems):

    try:
        problem = CUTEST(name=name)
    except Exception as e:
        print(f"error with {name}", e)
        continue

    print("Name : " + problem.name)
    names.append(problem.name)
    dimension.append(problem.d)
    n_constraints.append(problem.number_of_eq_constraints + problem.number_of_ineq_constraints)
    n_equality.append(problem.number_of_eq_constraints)
    n_inequality.append(problem.number_of_ineq_constraints)
    has_bounds.append(problem.has_bound_constraints)
    is_real.append(problem.real_problem())

total_intex = pd.DataFrame(
    {
        "name": names,
        "dimension" : dimension,
        "n_constraints": n_constraints,
        "n_equality": n_equality,
        "n_inequality": n_inequality,
        "has_bounds": has_bounds
    }
)
total_intex.to_csv("CUTEST_index.csv")