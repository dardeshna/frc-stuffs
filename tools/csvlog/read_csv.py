import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""Reads and plots data from a csv log file with the following format

name, timestamp, value
"ch1", 0, 1
"ch2", 0, 2
"ch1", 1, 2
"ch2", 1, 3
...

"""

# file (str): file path
file = 'data.csv'

# t_default ([float, float]): default time interval to plot
t_default = []

# channels in csv file
# plot (bool): whether to plot
# subplot (int): which subplot to use
# t ([float, float]): time interval to plot
# ls, color, ... : pyplot args
channels = {
    'sin' : {
        'plot': True,
    },
    'cos' : {
        'plot': True,
        'subplot': 0,
        'ls': '--',
        'color': 'orange',
    },
    'log' : {
        'plot': True,
        'subplot': 1,
        't': [1, 4],
    },
}


df = pd.read_csv(file, skiprows=1, header=None, skipinitialspace=True)
_, subplots = np.unique([v.get('subplot', 0) for v in channels.values()], return_inverse=True)
fig, axs = plt.subplots(np.max(subplots)+1, squeeze=False, sharex=True)
axs = axs[:, 0]

for i, (c, v) in enumerate(channels.items()):

    if not v.get('plot', True):
        continue

    t = v.get('t', t_default)
    
    mask = (df[0] == c)
    if len(t) == 2:
        mask = mask & (df[1] >= t[0]) & (df[1] <= t[1])

    kwargs = {a:b for a,b in v.items() if a not in ('plot', 'subplot', 't')}
    
    df.loc[mask].plot(1, 2, label=c, ax=axs[subplots[i]], **kwargs)

plt.xlabel('time (s)')
plt.show()