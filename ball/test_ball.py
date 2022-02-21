import sys
import os
sys.path.insert(1, os.path.realpath(os.path.pardir))

import numpy as np
from matplotlib import pyplot as plt

from ball import Ball

from ball_util import *

from util import Sim

m = 0.26932025
d = 0.2413
C_d = 0.5
rho = 1.2
g = 9.80665

ball = Ball(m, d, C_d, rho, g)

v_0, theta_0 = 7.0104, 1.3439041
x_0 = np.array([-5 / METERS_TO_FEET, 0.9144, v_0*np.cos(theta_0), v_0*np.sin(theta_0), 0])

sim = Sim(ball, x_0)

t = np.linspace(0, 1.5, 101)

sim.solve(lambda x, t: np.array([]), t)

xs, us, ts = sim.get_result()

print(np.c_[ts, METERS_TO_FEET*xs[:, :2]])

plot_traj(*(xs[:, :2].T))
plt.show()