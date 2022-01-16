import sys
import os
sys.path.insert(1, os.path.realpath(os.path.pardir))

import copy
import numpy as np
from matplotlib import pyplot as plt

from flywheel import Flywheel
from util import Motor, Encoder, Talon, TalonControlMode, Sim

SIM_DT = 0.001
CTRL_DT = 0.02

# motor and system constants

V_nominal = 12 # V
w_free =  5676 * (2 * np.pi / 60) # RPM -> rad/s
T_stall = 2.6 # Nm
I_stall = 105 # A
I_free = 1.8 # A
n_motors = 1

G_ratio = 1
J_flywheel = 3.6e-4 # kg*m^2

motor = Motor(V_nominal, w_free, T_stall, I_stall, I_free, n_motors)
flywheel = Flywheel(motor, G_ratio, J_flywheel)
encoder = Encoder(G=G_ratio, dt=SIM_DT)
base_talon = Talon(encoder, dt=SIM_DT)
ticks_per_rev, ticks_per_100ms_per_rev_per_second = encoder.get_conversions()

print(ticks_per_100ms_per_rev_per_second)

# PID Gains
base_talon.P = 0
base_talon.I = 0
base_talon.D = 0
base_talon.F = 1023 / (w_free / G_ratio * ticks_per_100ms_per_rev_per_second)
base_talon.izone = 0

def sim_offboard(sim_ts, x_0, w_goal, tau_ext):

    sim = Sim(flywheel, x_0)
    talon = copy.deepcopy(base_talon)
    talon.set(TalonControlMode.Velocity, w_goal * ticks_per_100ms_per_rev_per_second)

    # run simulation
    for i, t in enumerate(sim_ts):

        talon.update()
        V = talon.get_motor_output_voltage()
        sim.step((V, tau_ext[i]), SIM_DT)
        talon.encoder.push_reading(sim.xs[-1][0] * ticks_per_rev)

    return sim


# def sim_onboard(sim_ts, x_0, w_goal, tau_ext):

#     sim = Sim(flywheel, x_0)
#     talon = copy.deepcopy(base_talon)

#     ctrl_ts = np.arange(int(t/CTRL_DT)) * CTRL_DT

#     update_ctrl = np.zeros_like(sim_ts, dtype=bool)
#     update_ctrl[np.searchsorted(sim_ts, ctrl_ts)] = True

#     j = 0

#     # run simulation
#     for i, t in enumerate(sim_ts):

#         if update_ctrl[i]:
#             w = talon.encoder.sensor_rate / ticks_per_100ms_per_rev_per_second
#             talon.set(TalonControlMode.PercentOutput, 0)
#             j += 1

#         talon.update()
#         V = talon.get_motor_output_voltage()
#         sim.step((V, tau_ext[i]), SIM_DT)
#         talon.pushReading(sim.xs[-1][0] * ticks_per_rev)


def sim_spinup(sim_func, t, w_goal):

    sim_ts = np.arange(int(t/SIM_DT)) * SIM_DT
    tau_ext = np.zeros_like(sim_ts)

    sim = sim_func(sim_ts, np.array([0, 0]), w_goal, tau_ext)

    return sim


if __name__ == "__main__":

    sim = sim_spinup(sim_offboard, 5, w_free/2)
    xs, us, ts = sim.get_result()

    plt.plot(ts, xs[:,1] * 60 / (2*np.pi))
    plt.show()

