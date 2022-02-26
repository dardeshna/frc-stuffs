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

# helper function to unpack flattened trajectory
def unpack(x):

    t = x[-1]
    w = x[-2]
    x,y,v_x,v_y = x[:-2].reshape((4, -1))

    return x, y, v_x, v_y, w, t

def get_vel_spin(w_top, w_bottom, d_ball, d_wheel, G=1):

    v_top = w_top / G * 2 * np.pi / 60 * (d_wheel / 2)
    v_bottom = w_bottom / G * 2 * np.pi / 60 * (d_wheel / 2)

    v = (v_top + v_bottom) / 2
    w = (v_top - v_bottom) / (2 * np.pi * d_ball)

    return v, w

def get_top_bottom(v, w, d_ball, d_wheel, G=1):

     w_top = (v + w * 2 * np.pi * d_ball / 2) / (d_wheel / 2) * G * 60 / (2 * np.pi)
     w_bottom = (v - w * 2 * np.pi * d_ball / 2) / (d_wheel / 2) * G * 60 / (2 * np.pi)
     
     return w_top, w_bottom

def get_ball_exit(d, h, phi, camera_to_pivot, flywheel_spacing):

    return np.array([
        (get_2022_goal()[0,0] - (d + camera_to_pivot + flywheel_spacing/2 * np.cos(phi)) / 12) / METERS_TO_FEET,
        (h + flywheel_spacing/2 * np.sin(phi)) / 12 / METERS_TO_FEET,
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