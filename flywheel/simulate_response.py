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
r_flywheel = 0.076 # m
m_flywheel = 0.23 # kg
n_flywheel = 2
J_flywheel = 1/2 * m_flywheel * r_flywheel**2 * n_flywheel # kg*m^2, solid disk

motor = Motor(V_nominal, w_free, T_stall, I_stall, I_free, n_motors)
flywheel = Flywheel(motor, G_ratio, J_flywheel)
encoder = Encoder(G=G_ratio, dt=SIM_DT)
base_talon = Talon(encoder, dt=SIM_DT)
ticks_per_rev, ticks_per_100ms_per_rev_per_second = encoder.get_conversions()

# Current limits
base_talon.stator_limit_config = (10, 20, 0)
base_talon.stator_limit_enable = False
base_talon.supply_limit_config = (np.inf, np.inf, 0)
base_talon.supply_limit_enable = False

# PID Gains
base_talon.P = 0.1
base_talon.I = 0
base_talon.D = 0
base_talon.F = 1023 / (w_free / G_ratio * ticks_per_100ms_per_rev_per_second)
base_talon.izone = 0

# LQR gains

def sim_offboard(sim_ts, x_0, w_goal, tau_ext):

    sim = Sim(flywheel, x_0)
    talon = copy.deepcopy(base_talon)
    talon.encoder.set_rate(x_0[1] * ticks_per_100ms_per_rev_per_second)
    talon.set(TalonControlMode.Velocity, w_goal * ticks_per_100ms_per_rev_per_second)

    # run simulation
    for i, t in enumerate(sim_ts):

        talon.update()
        V = talon.get_motor_output_voltage()
        sim.step((V, tau_ext[i]), SIM_DT)
        talon.encoder.push_reading(sim.xs[-1][0] * ticks_per_rev)
        talon.current = np.abs(motor.get_I(V, sim.xs[-1][1]))

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

def sim_shoot(sim_func, t, w_goal):

    tau_dist, steps_dist = calc_ball_ext_torque(w_goal)

    idx_start = int(0.2*t / SIM_DT)
    idx_end = idx_start + steps_dist

    t = idx_end * SIM_DT if idx_end * SIM_DT > t else t

    sim_ts = np.arange(int(t/SIM_DT)) * SIM_DT
    tau_ext = np.zeros_like(sim_ts)
    tau_ext[idx_start:idx_end] = tau_dist
    
    sim = sim_func(sim_ts, np.array([0, w_goal]), w_goal, tau_ext)

    return sim

# Ball disturbance

m_ball = 0.27 # kg
r_ball = 0.12 # m
J_ball = 2/3 * m_ball * r_ball ** 2 # kg*m^2, thin spherical shell
wrap_angle = np.pi / 4 # rad

def calc_ball_ext_torque(w_flywheel):

    v_ball = 1/2 * w_flywheel * r_flywheel
    dir = np.sign(w_flywheel)

    K_trans = 1/2 * m_ball * v_ball ** 2
    K_rot = 1/2 * J_ball * (v_ball / r_ball) ** 2
    K = K_trans + K_rot

    w_flywheel_final = dir * np.sqrt(w_flywheel ** 2 - 2 / J_flywheel * K)

    print("initial speed:", w_flywheel * 60 / (2 * np.pi))
    print("final speed:", w_flywheel_final * 60 / (2 * np.pi))

    t = np.abs(wrap_angle / ((w_flywheel + w_flywheel_final) / 2))
    steps = int(t / SIM_DT)
    
    tau_ext = dir * K / (wrap_angle * steps * SIM_DT / t)

    print("actual disturbance: ", t, K/wrap_angle)
    print("adjusted disturbance: ", steps * SIM_DT, tau_ext)

    return tau_ext, steps

if __name__ == "__main__":
    
    w_goal = w_free / 2

    print("flywheel inertia:", J_flywheel)
    print("flywheel energy:", 1/2 * J_flywheel * w_goal**2)

    tau_ext, t = calc_ball_ext_torque(w_goal)
    print("ball inertia:", J_ball)
    print("ball energy: ",  tau_ext * wrap_angle)

    print("accel time: ", t * SIM_DT)

    sim = sim_shoot(sim_offboard, 5, w_goal)
    xs, us, ts = sim.get_result()

    plt.figure()
    plt.plot(ts, xs[:,1] * 60 / (2*np.pi))

    plt.figure()
    plt.plot(ts, us[:, 0])
    plt.plot(ts[1:], motor.get_I(us[:-1, 0], xs[1:, 1]))
    plt.plot(ts[1:], motor.get_I(us[:-1, 0], xs[1:, 1])*us[:-1, 0]/12)

    plt.show()

