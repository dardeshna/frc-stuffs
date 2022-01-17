import sys
import os
sys.path.insert(1, os.path.realpath(os.path.pardir))

import numpy as np

from util import System

class Flywheel(System):
    """Flywheel

    state: [theta, w]

    theta = flywheel position
    w = flywheel speed

    inputs: [V, tau_ext]

    V = motor voltage
    tau_ext = external torque

    equations of motion:

    x_dot = w
    w_dot = - (G ** 2) / J * (k_t * k_b / R + k_F) * w  +  G / J * k_t / R * V  -  tau_ext / J

    R = motor resistance (multiple motors combine in parallel)
    k_t = motor torque constant
    k_b = motor back-emf constant
    k_F = motor viscous friction constant
    G = gear ratio
    J = flywheel moment of inertia
    """
    
    n = 2
    m = 2

    def __init__(self, motor, G, J):
        
        self.A = np.array([
            [0, 1],
            [0, - (G ** 2) / J * (motor.k_t * motor.k_b / motor.R + motor.k_F)]
        ])
        self.B = np.array([
            [0, 0],
            [G / J * motor.k_t / motor.R, -1 / J],
        ])

    def f(self, x, u):

        return self.A @ x + self.B @ u