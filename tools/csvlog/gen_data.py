import pandas as pd
import numpy as np

names = ['name', 'time', 'value']

t1 = np.linspace(0, 5, 100)
t2 = np.linspace(0, 3, 200)
t3 = np.linspace(1, 6, 50)

f1 = np.cos(t1)
f2 = np.sin(t2)
f3 = np.log(t3)

l1 = np.repeat('sin', len(f1))
l2 = np.repeat('cos', len(f2))
l3 = np.repeat('log', len(f3))

data = np.block([[l1, l2, l3], [t1, t2, t3], [f1, f2, f3]]).T
data = data[data[:, 1].argsort()]

df = pd.DataFrame(data, columns=names)
df.to_csv('data.csv', index=False)

print(df)