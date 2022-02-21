import numpy as np
from matplotlib import pyplot as plt

METERS_TO_FEET = 3.28084

def get_2022_goal():

    return np.array([
        [-2, 8.67],
        [-1.25, 7],
        [1.25, 7],
        [2, 8.67],
    ])

def plot_traj(x, y, x_actual=None, y_actual=None):

    plt.figure()
    plt.plot(x * METERS_TO_FEET, y * METERS_TO_FEET, label='optimized')

    if x_actual is not None and y_actual is not None:
        plt.gca().set_prop_cycle(None)
        plt.plot(x_actual * METERS_TO_FEET, y_actual * METERS_TO_FEET, '--', alpha=0.4, label='re-simulated')
    
    plt.plot(*(get_2022_goal().T))
    plt.plot([np.min(x) * METERS_TO_FEET, np.max(x) * METERS_TO_FEET], [0, 0], '-.')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('optimized ball trajectory')
    plt.legend()