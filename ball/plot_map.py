import sys
import os
sys.path.insert(1, os.path.realpath(os.path.pardir))

import numpy as np
from matplotlib import pyplot as plt

from ball import Ball

from ball_util import *

from util import Sim

from scipy.optimize import minimize

m_ball = 0.26932025
d_ball = 0.2413

ball = Ball(m_ball, d_ball)

d_wheel = 4 / 12 / METERS_TO_FEET # (in -> m)

h = 29.5 # shooter height (in)

w_motor_max = 6380 # (rpm)
G = 9/4

# interpolating map entries (target distance, w_bottom w_top, hood angle)
map = np.array([
    [74.38, 1500.0, 2000.0, 20.0],
    [82.38, 1650.0, 2150.0, 24.0],
    [90.38, 1651.0, 2151.0, 24.5],
    [98.95, 1800.0, 2300.0, 27.0],
    [120.0, 1900.0, 2400.0, 30.0],
    [133.5, 1950.0, 2450.0, 33.0],
    [159.7, 2050.0, 2450.0, 33.5],
    [171.1, 2150.0, 2650.0, 37.0],
    [181.1, 2154.2, 2656.9, 39.0],
    [200.1, 2350.0, 2800.0, 40.0],
    [220.1, 2450.0, 2900.0, 43.0],
])

d, w_bottom, w_top, phi = map.T

phi = np.deg2rad(phi)
theta_0 = np.pi/2 - phi

v_0, w_0 = get_vel_spin(w_top, w_bottom, d_ball, d_wheel)

# initial conditions for each map entry
x_0s = np.c_[
    get_ball_exit(d, 29.5, phi, camera_to_pivot=-10.6, flywheel_spacing=11.5).T,
    v_0*np.cos(theta_0),
    v_0*np.sin(theta_0),
    w_0]

# calculate amount to scale velocity (shooter efficiency) for ball to actually score in goal
def get_efficiency():

    efficiency = []

    for i in range(x_0s.shape[0]):

        # initial and final positions (ft -> m)
        x_i = x_0s[i][0:2]
        x_f = np.array([0, get_2022_goal()[0,1]]) / METERS_TO_FEET

        # number of points to optimize over
        n = 101

        # naive straight line trajectory initialization
        x_0 = np.append(np.r_[np.linspace(x_i, x_f, n).T, np.ones((2, n))].ravel(), [0,1])

        # objective function
        def f(x_mat):
            x, y, v_x, v_y, w, t = unpack(x_mat)
            return np.sqrt((x[-1]-x_f[0])**2 + (y[-1]-x_f[1])**2)

        # equality constraints
        def c_eq(x_mat):

            x, y, v_x, v_y, w, t = unpack(x_mat)
            dt = t/n

            v = np.sqrt(v_x**2 + v_y**2)
            theta = np.arctan2(v_y, v_x)

            x_dot = ball.f((x, y, v_x, v_y, w), None, None)[:4] # compute derivatives at each trajectory point

            return np.r_[
                (x_mat[:-2].reshape((4,-1))[:,1:] - (x_mat[:-2].reshape((4,-1))[:,:-1] + x_dot[:,:-1] * dt)).ravel(), # continuity
                x[0]-x_i[0], # initial position
                y[0]-x_i[1],
                theta[0] - theta_0[i],
                w/v[0] - w_0[i]/v_0[i],
            ]

        # run optimization
        res = minimize(f, x_0, constraints=[{'type':'eq','fun':c_eq}])

        x, y, v_x, v_y, w, t = unpack(res['x'])
        v = np.sqrt(v_x**2 + v_y**2)
        theta = np.arctan2(v_y, v_x)

        print("d: ", map[i,0])
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
        print("top motor rpm:  {}\nbottom motor rpm:  {}".format(*get_top_bottom(v[0], w, d_ball, d_wheel, G)))
        print("efficiency: ", v[0]/v_0[i])
        print("-------")

        efficiency.append(v[0]/v_0[i])
    
    return efficiency

efficiency = get_efficiency()
print(efficiency)
# efficiency = [0.8902330808701645, 0.9150868146812376, 0.7630654811726282, 0.7607101629145113, 0.7584153140375083, 0.8194389464192788, 0.7541686335546248, 0.7656953548276755, 0.7433674358929608, 0.7426189748395449]

# plot unscaled and scaled trajectories
for i, x in enumerate(x_0s):

    t = np.linspace(0, 1.5, 101)

    x_scaled = x.copy()
    x_scaled[2:] *= efficiency[i]

    for j, x_ in enumerate((x, x_scaled)):
    
        sim = Sim(ball, x_)
        sim.solve(lambda x, t: np.array([]), t)

        xs, us, ts = sim.get_result()

        x, y = xs[:, :2].T * METERS_TO_FEET

        plt.figure(j)
        plt.plot(x[x <= 0], y[x <= 0], label=f"d: {map[i,0]}")

for i, t in enumerate(('unscaled trajectories', 'scaled trajectories')):
    plt.figure(i)
    plt.plot(*(get_2022_goal().T))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(t)

# plot map velocity vs estimated velocity
plt.figure()
plt.plot(w_top * G, efficiency * w_top * G, label='top motor')
plt.plot(w_bottom * G, efficiency * w_bottom * G, label='bottom motor')
plt.axvline(w_motor_max, ls='-.')
plt.xlabel('commanded velocity')
plt.ylabel('estimated velocity')
plt.legend()

# plot distance vs efficiency
plt.figure()
plt.plot(d, efficiency)
plt.xlabel('distance')
plt.ylabel('efficiency')

plt.show()