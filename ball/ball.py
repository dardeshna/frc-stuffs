import sys
import os
sys.path.insert(1, os.path.realpath(os.path.pardir))

import numpy as np

from util import System

class Ball(System):
    """Ball

    state: [x, y, v_x, v_y, w]

    x = ball x position
    y = ball y position
    v_x = ball x velocity
    v_y = ball y velocity
    w = ball topspin frequency

    equations of motion:

    v = sqrt(v_x**2 + v_y**2)
    theta = arctan2(v_y, v_x)
    x_dot = v_x
    y_dot = v_y
    v_x_dot (-D * v^2 * cos(theta)) / m
    v_y_dot = (-D * v^2 * sin(theta) - rho * d^3 * w * v) / m - g

    v = ball speed
    theta = ball tangent angle

    D = 1/2 * rho * C_d * (pi * (d/2)^2) = drag constant
    m = ball mass
    d = ball diameter
    rho = air density
    C_d = coefficient of drag
    g = gravitational acceleration
    """
    
    n_states = 5
    m_inputs = 0

    def __init__(self, m, d, C_d=0.5, rho=1.2, g=9.81):

        self.m = m
        self.d = d
        self.C_d = C_d
        self.rho = rho
        self.g = g

        self.D = 1/2 * rho * C_d * (np.pi * (d/2)**2)

    def f(self, x, u, t):

        x,y,v_x,v_y,w = x
        
        v = np.sqrt(v_x**2 + v_y**2)
        theta = np.arctan2(v_y, v_x)

        a_x = (- self.D * v**2 * np.cos(theta)) / self.m
        a_y = (- self.D * v**2 * np.sin(theta) - self.rho * self.d**3 * w * v) / self.m - self.g

        return np.array([
            v_x,
            v_y,
            a_x,
            a_y,
            np.zeros_like(v_x),
        ])