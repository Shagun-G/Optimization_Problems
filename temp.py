from Opt_Problems.Deterministic.CUTEST import CUTEST
from tqdm import tqdm
import pandas as pd

df = pd.read_csv('Equality.csv')

df['is_real'] = None
for i in tqdm(range(len(df))):
    name = df.iloc[i]['name']
    problem = CUTEST(name)
    df.loc[i, 'is_real'] = problem.real_problem()

df.to_csv('Equality.csv', index=False)
