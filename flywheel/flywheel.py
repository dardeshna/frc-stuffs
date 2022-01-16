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
    w_dot = -(G ** 2) * k_t / (1 / k_b * R * J) * w  +  G * k_t / (R * J) * V  -  tau_ext / J

    R = motor resistance (multiple motors combine in parallel)
    k_t = motor torque constant
    k_b = motor back-emf constant
    G = gear ratio
    J = flywheel moment of inertia
    """
    
    n = 2
    m = 2

    def __init__(self, motor, G, J):
        
        self.A = np.array([
            [0, 1],
            [0, -(G ** 2) * motor.k_t / (1 / motor.k_b * motor.R * J)]
        ])
        self.B = np.array([
            [0, 0],
            [G * motor.k_t / (motor.R * J), -1 / J],
        ])

    def f(self, x, u):

        return self.A @ x + self.B @ u