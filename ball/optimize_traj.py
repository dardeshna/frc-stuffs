import sys
import os
sys.path.insert(1, os.path.realpath(os.path.pardir))

import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import minimize

from ball import Ball
from ball_util import *

from util import Sim

m = 0.26932025 # ball mass (kg)
d = 0.2413 # ball diameter (m)

ball = Ball(m, d)

# initial and final positions (ft -> m)
x_i = np.array([-10, 3]) / METERS_TO_FEET
x_f = np.array([0, 8.67]) / METERS_TO_FEET

# min and max initial angles (deg)
theta_0_min = np.deg2rad(50)
theta_0_max = np.deg2rad(90)

# maximum motor and ball velocities
w_motor_max = 5676 * 2 * np.pi / 60 # (rpm -> rad/s)
G = 2
d_wheel = 4 / 12 / METERS_TO_FEET
v_tangential_max = w_motor_max / G * (d_wheel / 2)

# maximum height (ft -> m)
y_max = 14 / METERS_TO_FEET

# number of points to optimize over
n = 101

# naive straight line trajectory initialization
x_0 = np.append(np.r_[np.linspace(x_i, x_f, n).T, np.ones((2, n))].ravel(), [0,1])

# helper function to unpack flattened trajectory
def unpack(x):

    t = x[-1]
    w = x[-2]
    x,y,v_x,v_y = x[:-2].reshape((4, -1))

    return x, y, v_x, v_y, w, t

# objective function
def f(x_mat):
    x, y, v_x, v_y, w, t = unpack(x_mat)
    theta = np.arctan2(v_y, v_x)
    return np.abs(theta[-1] + np.pi/2) # make final angle as close to -90 as possible

# equality constraints
def c_eq(x_mat):

    x, y, v_x, v_y, w, t = unpack(x_mat)
    dt = t/n

    x_dot = ball.f((x, y, v_x, v_y, w), None, None)[:4] # compute derivatives at each trajectory point

    return np.r_[
        (x_mat[:-2].reshape((4,-1))[:,1:] - (x_mat[:-2].reshape((4,-1))[:,:-1] + x_dot[:,:-1] * dt)).ravel(), # continuity
        x[0]-x_i[0], # initial position
        y[0]-x_i[1],
        x[-1]-x_f[0], # final position
        y[-1]-x_f[1],
        w, # force zero ball spin
    ]

# inequality constraints (non-negative)
def c_ineq(x_mat):
    x, y, v_x, v_y, w, t = unpack(x_mat)

    v = np.sqrt(v_x**2 + v_y**2)
    theta = np.arctan2(v_y, v_x)

    return np.r_[
        y_max - np.max(y), # max height
        theta[0] - theta_0_min, # min initial angle
        theta_0_max - theta[0], # max initial angle
        v_tangential_max - (v[0] + np.abs(w * 2 * np.pi * d / 2)), # limit linear velocity and spin based on max flywheel speed
    ]

# run optimization
res = minimize(f, x_0, constraints=[{'type':'eq','fun':c_eq}, {'type':'ineq','fun':c_ineq}])

x, y, v_x, v_y, w, t = unpack(res['x'])
v = np.sqrt(v_x**2 + v_y**2)
theta = np.arctan2(v_y, v_x)

print("t: ", t)
# print("x_0: ", x[0] * METERS_TO_FEET)
# print("y_0: ", y[0] * METERS_TO_FEET)
# print("x_f: ", x[-1] * METERS_TO_FEET)
# print("y_f: ", y[-1] * METERS_TO_FEET)
print("v_0: ", v[0] * METERS_TO_FEET)
print("theta_0: ", np.rad2deg(theta[0]))
print("v_f: ", v[-1] * METERS_TO_FEET)
print("theta_f: ", np.rad2deg(theta[-1]))
print("w: ", w)
print("top motor rpm: ", (v[0] + w * 2 * np.pi * d / 2) / (d_wheel / 2) * G * 60 / (2 * np.pi))
print("bottom motor rpm: ", (v[0] - w * 2 * np.pi * d / 2) / (d_wheel / 2) * G * 60 / (2 * np.pi))

# resimulate with integration
sim = Sim(ball, np.array([*x_i, v_x[0], v_y[0], w]))
ts = np.linspace(0, t, n)
sim.solve(lambda x, t: np.array([]), ts)
xs, us, ts = sim.get_result()

# plot trajectory
plot_traj(x,y, *(xs[:, :2].T))
plt.show()